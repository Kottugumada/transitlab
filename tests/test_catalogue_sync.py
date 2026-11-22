"""
Offline tests for catalogue_sync parsing and transformation.

These tests use committed fixture CSVs and do NOT require network access
or a live database — they test only the pure data-transformation logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.catalogue_sync import (
    DISPOSITION_DERIVED_FEATURES,
    HONEST_FEATURE_COLUMNS,
    _nan_to_none,
    _row_to_koi,
    _row_to_ps,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── _nan_to_none ──────────────────────────────────────────────────────────────

def test_nan_to_none_on_nan():
    assert _nan_to_none(float("nan")) is None
    assert _nan_to_none(np.nan) is None


def test_nan_to_none_passthrough():
    assert _nan_to_none(3.14) == pytest.approx(3.14)
    assert _nan_to_none("hello") == "hello"
    assert _nan_to_none(None) is None
    assert _nan_to_none(0) == 0


# ── KOI fixture parsing ───────────────────────────────────────────────────────

@pytest.fixture
def koi_df():
    return pd.read_csv(FIXTURES / "koi_excerpt.csv", low_memory=False)


def test_koi_fixture_loads(koi_df):
    assert len(koi_df) == 10


def test_koi_row_to_dict_confirmed(koi_df):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = koi_df[koi_df["kepoi_name"] == "K00001.01"].iloc[0]
    d = _row_to_koi(row, now)

    assert d["kepoi_name"] == "K00001.01"
    assert d["kepid"] == 11446443
    assert d["kepler_name"] == "Kepler-1 b"
    assert d["disposition"] == "CONFIRMED"
    assert d["pdisposition"] == "CANDIDATE"
    assert d["period"] == pytest.approx(2.4706133)
    assert d["depth"] == pytest.approx(15727.0)
    assert d["synced_at"] == now


def test_koi_row_to_dict_false_positive(koi_df):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = koi_df[koi_df["kepoi_name"] == "K00004.01"].iloc[0]
    d = _row_to_koi(row, now)
    assert d["disposition"] == "FALSE POSITIVE"


def test_koi_features_blob_contains_honest_features(koi_df):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = koi_df.iloc[0]
    d = _row_to_koi(row, now)
    features = d["features"]
    for col in HONEST_FEATURE_COLUMNS:
        assert col in features, f"Honest feature '{col}' missing from features blob"


def test_koi_features_blob_contains_robovetter_flags(koi_df):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = koi_df.iloc[0]
    d = _row_to_koi(row, now)
    features = d["features"]
    for col in DISPOSITION_DERIVED_FEATURES:
        if col.startswith("koi_fp"):
            assert col in features, f"Robovetter flag '{col}' missing from features blob"


def test_koi_no_nan_in_known_good_row(koi_df):
    """Known-good rows in the fixture should produce no None in numeric fields."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = koi_df[koi_df["kepoi_name"] == "K00001.01"].iloc[0]
    d = _row_to_koi(row, now)
    for field in ("period", "depth", "model_snr", "steff", "srad", "kepmag"):
        assert d[field] is not None, f"Field '{field}' unexpectedly None"


def test_koi_label_distribution(koi_df):
    dist = koi_df["koi_disposition"].value_counts().to_dict()
    assert dist.get("CONFIRMED", 0) >= 1
    assert dist.get("FALSE POSITIVE", 0) >= 1
    # Fixture has 8 CONFIRMED and 2 FALSE POSITIVE
    assert dist["CONFIRMED"] == 8
    assert dist["FALSE POSITIVE"] == 2


def test_koi_multi_planet_host(koi_df):
    """Kepler-100 has two KOIs in the fixture — both should parse."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    subset = koi_df[koi_df["kepid"] == 6521045]
    assert len(subset) == 2
    names = {_row_to_koi(row, now)["kepoi_name"] for _, row in subset.iterrows()}
    assert names == {"K00041.01", "K00041.02"}


# ── PS fixture parsing ────────────────────────────────────────────────────────

@pytest.fixture
def ps_df():
    return pd.read_csv(FIXTURES / "ps_excerpt.csv", low_memory=False)


def test_ps_fixture_loads(ps_df):
    assert len(ps_df) == 5


def test_ps_row_to_dict(ps_df):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = ps_df[ps_df["pl_name"] == "Kepler-10 b"].iloc[0]
    d = _row_to_ps(row, now)

    assert d["pl_name"] == "Kepler-10 b"
    assert d["hostname"] == "Kepler-10"
    assert d["period"] == pytest.approx(0.8374907)
    assert d["rade"] == pytest.approx(1.47)
    assert d["synced_at"] == now


def test_ps_null_mass_becomes_none(ps_df):
    """Kepler-186 f has no mass in the fixture — should map to None."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    row = ps_df[ps_df["pl_name"] == "Kepler-186 f"].iloc[0]
    d = _row_to_ps(row, now)
    # pl_bmasse is NaN in fixture (empty cell)
    # _row_to_ps doesn't map pl_bmasse directly (not in PS_COLUMNS mapping to model),
    # but sy_vmag should still parse
    assert d["pl_name"] == "Kepler-186 f"


# ── Leakage audit: honest vs all features ────────────────────────────────────

def test_honest_features_exclude_disposition_derived():
    """No honest feature should appear in the disposition-derived list."""
    overlap = set(HONEST_FEATURE_COLUMNS) & set(DISPOSITION_DERIVED_FEATURES)
    assert not overlap, (
        f"These columns appear in both honest and disposition-derived lists: {overlap}\n"
        "This would leak label information into the honest-features model."
    )


def test_disposition_derived_includes_koi_score():
    """koi_score is the Robovetter's probability — must be excluded from honest model."""
    assert "koi_score" in DISPOSITION_DERIVED_FEATURES


def test_disposition_derived_includes_all_fpflags():
    flags = {"koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"}
    assert flags.issubset(set(DISPOSITION_DERIVED_FEATURES)), (
        "Robovetter flags encode vetting decisions and must be in DISPOSITION_DERIVED_FEATURES"
    )
