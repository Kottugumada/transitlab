"""
Offline tests for the classifier training logic.

The critical invariant: split MUST be by star (kepid), never by row.
A multi-KOI host that spans both partitions would leak stellar-parameter
features and invalidate the validation numbers.

All tests run offline against the committed fixture CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.train import (
    ALL_FEATURE_COLUMNS,
    HONEST_FEATURE_COLUMNS,
    SPLIT_SEED,
    make_binary_labels,
    split_by_star,
    load_training_data_from_csv,
)
from app.catalogue_sync import DISPOSITION_DERIVED_FEATURES

FIXTURES = Path(__file__).parent / "fixtures"


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def _synthetic_df(n_stars: int = 20, kois_per_star: int = 2) -> pd.DataFrame:
    """
    Build a synthetic DataFrame that mimics what load_training_data returns.
    Each star has `kois_per_star` KOIs with identical kepid.
    Half the stars have CONFIRMED KOIs, half FALSE POSITIVE.
    """
    rng = np.random.default_rng(0)
    rows = []
    for star_i in range(n_stars):
        kepid = 10000 + star_i
        disposition = "CONFIRMED" if star_i < n_stars // 2 else "FALSE POSITIVE"
        for koi_j in range(kois_per_star):
            rows.append({
                "kepoi_name": f"K{star_i:05d}.0{koi_j+1}",
                "kepid": kepid,
                "disposition": disposition,
                "koi_period": rng.uniform(1, 100),
                "koi_model_snr": rng.uniform(10, 1000),
                "koi_depth": rng.uniform(100, 20000),
                "koi_duration": rng.uniform(1, 10),
                "koi_steff": rng.uniform(4000, 7000),
                "koi_srad": rng.uniform(0.5, 2.5),
                "koi_slogg": rng.uniform(3.5, 5.0),
                "koi_kepmag": rng.uniform(9, 16),
                "koi_ror": rng.uniform(0.01, 0.2),
                "koi_impact": rng.uniform(0, 1),
                "koi_max_mult_ev": rng.uniform(5, 500),
                "koi_bin_oedp_sig": rng.uniform(0, 5),
                "koi_time0bk": rng.uniform(60, 80),
                "koi_score": rng.uniform(0, 1),
                "koi_fpflag_nt": int(rng.integers(0, 2)),
                "koi_fpflag_ss": int(rng.integers(0, 2)),
                "koi_fpflag_co": int(rng.integers(0, 2)),
                "koi_fpflag_ec": int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


# ── split_by_star correctness ─────────────────────────────────────────────────

def test_no_star_in_both_partitions():
    """Core invariant: no kepid appears in both train and test."""
    df = _synthetic_df(n_stars=40, kois_per_star=3)
    y = make_binary_labels(df["disposition"])
    train_idx, test_idx = split_by_star(df, y)

    train_stars = set(df.iloc[train_idx]["kepid"])
    test_stars = set(df.iloc[test_idx]["kepid"])
    overlap = train_stars & test_stars
    assert not overlap, (
        f"{len(overlap)} kepid(s) appear in both train and test: {overlap}\n"
        "This is a leakage bug — split must be by star, not by row."
    )


def test_split_covers_all_rows():
    df = _synthetic_df()
    y = make_binary_labels(df["disposition"])
    train_idx, test_idx = split_by_star(df, y)
    assert len(train_idx) + len(test_idx) == len(df)


def test_split_approximate_size():
    """Test partition is roughly 80/20."""
    df = _synthetic_df(n_stars=100)
    y = make_binary_labels(df["disposition"])
    train_idx, test_idx = split_by_star(df, y)
    test_frac = len(test_idx) / len(df)
    # Allow wide tolerance: GroupShuffleSplit snaps to star boundaries
    assert 0.10 <= test_frac <= 0.35, f"Test fraction {test_frac:.2f} outside expected range"


def test_split_is_deterministic():
    """Same seed → same split."""
    df = _synthetic_df()
    y = make_binary_labels(df["disposition"])
    t1, v1 = split_by_star(df, y, seed=SPLIT_SEED)
    t2, v2 = split_by_star(df, y, seed=SPLIT_SEED)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(v1, v2)


def test_different_seeds_differ():
    df = _synthetic_df(n_stars=50)
    y = make_binary_labels(df["disposition"])
    t1, _ = split_by_star(df, y, seed=0)
    t2, _ = split_by_star(df, y, seed=99)
    assert not np.array_equal(t1, t2), "Different seeds should produce different splits"


# ── make_binary_labels ────────────────────────────────────────────────────────

def test_confirmed_is_planet_like():
    s = pd.Series(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE", "CONFIRMED"])
    labels = make_binary_labels(s)
    assert list(labels) == [1, 1, 0, 1]


def test_candidate_is_planet_like():
    """CANDIDATE is planet-like in the binary framing."""
    s = pd.Series(["CANDIDATE"])
    assert make_binary_labels(s).iloc[0] == 1


def test_false_positive_is_zero():
    s = pd.Series(["FALSE POSITIVE"])
    assert make_binary_labels(s).iloc[0] == 0


# ── Feature-set invariants (leakage audit) ────────────────────────────────────

def test_honest_and_all_features_differ():
    """The all-features model must add at least the disposition-derived columns."""
    extra = set(ALL_FEATURE_COLUMNS) - set(HONEST_FEATURE_COLUMNS)
    assert extra, "ALL_FEATURE_COLUMNS must be a strict superset of HONEST_FEATURE_COLUMNS"


def test_koi_score_in_all_but_not_honest():
    """koi_score encodes the Robovetter result — must NOT be an honest feature."""
    assert "koi_score" not in HONEST_FEATURE_COLUMNS, (
        "koi_score is disposition-derived — must be excluded from honest-features model"
    )
    assert "koi_score" in ALL_FEATURE_COLUMNS, (
        "koi_score must be present in all-features model for leakage comparison"
    )


def test_fp_flags_not_in_honest_features():
    """Robovetter flags must be absent from honest features."""
    flags = {c for c in DISPOSITION_DERIVED_FEATURES if c.startswith("koi_fpflag")}
    for flag in flags:
        assert flag not in HONEST_FEATURE_COLUMNS, (
            f"{flag} is a Robovetter output flag and must not appear in honest features"
        )


def test_all_honest_features_in_all_features():
    """Every honest feature must also be in the all-features set."""
    for col in HONEST_FEATURE_COLUMNS:
        assert col in ALL_FEATURE_COLUMNS, f"{col} missing from ALL_FEATURE_COLUMNS"


# ── Fixture-based smoke test ──────────────────────────────────────────────────

def test_load_from_fixture():
    df = load_training_data_from_csv(FIXTURES / "koi_excerpt.csv")
    assert len(df) == 10
    assert "disposition" in df.columns
    assert "kepid" in df.columns


def test_fixture_has_both_classes():
    df = load_training_data_from_csv(FIXTURES / "koi_excerpt.csv")
    y = make_binary_labels(df["disposition"])
    assert 0 in y.values
    assert 1 in y.values


def test_split_on_fixture_no_overlap():
    """Even with just 10 rows, split-by-star must not overlap kepids."""
    df = load_training_data_from_csv(FIXTURES / "koi_excerpt.csv")
    y = make_binary_labels(df["disposition"])
    train_idx, test_idx = split_by_star(df, y, test_size=0.3)
    train_stars = set(df.iloc[train_idx]["kepid"])
    test_stars = set(df.iloc[test_idx]["kepid"])
    assert not (train_stars & test_stars), "Split-by-star violated on fixture"
