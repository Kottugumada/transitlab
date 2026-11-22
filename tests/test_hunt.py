"""
Offline tests for the §6.2 hunt batch.

All network calls (MAST, Neon) are mocked; no .env required.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from app.hunt import period_recovered, run_hunt_batch
from app.engine.pipeline import hunt_pipeline

FIXTURES = Path(__file__).parent / "fixtures"
CADENCE_DAYS = 29.4 / 60 / 24


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _load_fixture(name: str) -> dict:
    """Load a CSV fixture and return arrays matching fetch_light_curve output."""
    import csv
    rows = list(csv.DictReader(open(FIXTURES / name)))
    # Fixture uses "time_bkjd" column name
    time_col = "time_bkjd" if "time_bkjd" in rows[0] else "time"
    time = np.array([float(r[time_col]) for r in rows])
    flux = np.array([float(r["flux"]) for r in rows])
    flux_err = np.array([float(r["flux_err"]) for r in rows])
    quality = np.zeros(len(time), dtype=np.int32)
    return {"time": time, "flux": flux, "flux_err": flux_err, "quality": quality}


def _make_settings(tmp_path=None, **overrides):
    """Return a mock Settings object with sensible defaults."""
    s = MagicMock()
    s.hunt_max_targets = 3
    s.hunt_n_periods = 500          # fast for tests
    s.hunt_period_max_days = 20.0
    s.bls_period_min_days = 0.5
    s.detrend_window_factor = 101
    s.hunt_recovery_tolerance = 0.01
    if tmp_path is not None:
        s._cache_dir = tmp_path
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ── period_recovered unit tests ───────────────────────────────────────────────

def test_period_recovered_exact():
    rec, alias = period_recovered(5.000, 5.000)
    assert rec is True
    assert alias == "exact"


def test_period_recovered_exact_within_tolerance():
    # 0.5% off — within the 1% default tolerance
    rec, alias = period_recovered(5.025, 5.000)
    assert rec is True
    assert alias == "exact"


def test_period_recovered_half():
    # BLS finds P/2 — common aliasing for grazing EBs
    rec, alias = period_recovered(2.5, 5.0)
    assert rec is True
    assert alias == "half"


def test_period_recovered_double():
    # BLS finds 2P
    rec, alias = period_recovered(10.0, 5.0)
    assert rec is True
    assert alias == "double"


def test_period_not_recovered():
    # BLS finds a completely different period
    rec, alias = period_recovered(3.14, 5.0)
    assert rec is False
    assert alias is None


def test_period_recovered_respects_tolerance():
    # 2% off — outside the 1% default tolerance
    rec, _ = period_recovered(5.11, 5.0, tolerance=0.01)
    assert rec is False


# ── hunt_pipeline on fixture ──────────────────────────────────────────────────

def test_hunt_pipeline_on_planet_fixture():
    """hunt_pipeline on the 5-day planet fixture should recover the period."""
    lc = _load_fixture("kic_planet_5d.csv")
    result = hunt_pipeline(
        lc["time"], lc["flux"], lc["flux_err"], lc["quality"],
        period_min=0.5, period_max=15.0, n_periods=500,
    )

    assert "bls" in result
    assert "fold_data" in result
    assert "fit" in result
    assert "vetting" in result

    bls = result["bls"]
    assert bls["period"] is not None
    # BLS should recover ~5 d period (within 5% for a short grid)
    rec, _ = period_recovered(bls["period"], 5.0, tolerance=0.05)
    assert rec, f"Expected ~5 d period, got {bls['period']:.4f} d"


def test_hunt_pipeline_bls_snr_positive():
    lc = _load_fixture("kic_planet_5d.csv")
    result = hunt_pipeline(
        lc["time"], lc["flux"], lc["flux_err"], lc["quality"],
        period_min=0.5, period_max=15.0, n_periods=500,
    )
    assert result["bls"]["snr"] > 0


def test_hunt_pipeline_fold_data_shape():
    lc = _load_fixture("kic_planet_5d.csv")
    result = hunt_pipeline(
        lc["time"], lc["flux"], lc["flux_err"], lc["quality"],
        period_min=0.5, period_max=15.0, n_periods=200,
    )
    fd = result["fold_data"]
    assert len(fd["phase"]) == 100
    assert all(-0.5 <= p <= 0.5 for p in fd["phase"])


def test_hunt_pipeline_cadence_counts():
    lc = _load_fixture("kic_planet_5d.csv")
    result = hunt_pipeline(
        lc["time"], lc["flux"], lc["flux_err"], lc["quality"],
        period_min=0.5, period_max=15.0, n_periods=200,
    )
    assert result["n_cadences_raw"] == len(lc["time"])
    assert result["n_cadences_clean"] <= result["n_cadences_raw"]


# ── run_hunt_batch mocked ─────────────────────────────────────────────────────

def _make_mock_koi(kepoi_name="K00001.01", kepid=757450, period=5.0, model_snr=37.0):
    koi = MagicMock()
    koi.kepoi_name = kepoi_name
    koi.kepid = kepid
    koi.period = period
    koi.model_snr = model_snr
    koi.disposition = "CONFIRMED"
    # features dict with the honest-feature columns
    koi.features = {
        "koi_period": period,
        "koi_time0bk": 120.4,
        "koi_duration": 2.0,
        "koi_depth": 1500.0,
        "koi_ror": 0.038,
        "koi_impact": 0.1,
        "koi_model_snr": model_snr,
        "koi_max_mult_ev": 10.0,
        "koi_bin_oedp_sig": 1.2,
        "koi_steff": 5800.0,
        "koi_srad": 1.0,
        "koi_slogg": 4.4,
        "koi_kepmag": 11.3,
    }
    return koi


def _make_synthetic_lc(period=5.0, n=1500, seed=42) -> dict:
    """Synthetic LC with an injected transit at the given period."""
    rng = np.random.default_rng(seed)
    time = 120.0 + np.arange(n) * CADENCE_DAYS
    flux = 1.0 + rng.normal(0, 200e-6, n)
    flux_err = np.full(n, 200e-6)
    quality = np.zeros(n, dtype=np.int32)
    # Inject transit
    epoch = 121.3
    dur = 2.0 / 24.0
    for nt in range(-2, int((time[-1] - time[0]) / period) + 2):
        mask = np.abs(time - (epoch + nt * period)) < dur / 2
        flux[mask] -= 1500e-6
    return {"time": time, "flux": flux, "flux_err": flux_err, "quality": quality}


def _make_mock_db(kois):
    """Return a mock SQLAlchemy Session that returns kois from _select_targets."""
    db = MagicMock()

    # Make query chain work for _select_targets
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.group_by.return_value = mock_query
    mock_query.subquery.return_value = MagicMock()
    mock_query.join.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = kois

    # For Target lookup (filter_by), return None → new Target created
    target_query = MagicMock()
    target_query.filter_by.return_value = target_query
    target_query.first.return_value = None
    target_query.order_by.return_value = target_query

    db.query.side_effect = lambda model: (
        mock_query if model.__name__ in ("KoiCatalogue",) else target_query
    )

    # flush just increments a mock id
    flush_counter = [0]
    def _flush():
        flush_counter[0] += 1
    db.flush.side_effect = _flush
    db.commit = MagicMock()
    db.add = MagicMock()

    return db


class _FakeModel:
    """Minimal classifier stand-in: always predicts probability 0.95."""
    def predict_proba(self, X):
        return np.array([[0.05, 0.95]] * len(X))


def test_run_hunt_batch_basic(tmp_path):
    """Full batch run with mocked DB, MAST, and joblib model."""
    kois = [_make_mock_koi()]
    settings = _make_settings(tmp_path=tmp_path)
    db = _make_mock_db(kois)

    lc = _make_synthetic_lc()

    mock_run = MagicMock()
    mock_run.id = 42

    with (
        patch("app.hunt._select_targets", return_value=kois),
        patch("app.hunt.fetch_light_curve", return_value=lc),
        patch("app.hunt.joblib.load", return_value=_FakeModel()),
        patch("app.hunt.hunt_pipeline") as mock_pipe,
        patch("app.hunt.HuntRun", return_value=mock_run),
    ):
        # Return a minimal but valid pipeline result
        mock_pipe.return_value = {
            "n_cadences_raw": 1500,
            "n_cadences_clean": 1499,
            "n_dropped": 1,
            "bls": {
                "period": 5.001,
                "epoch": 121.3,
                "duration_days": 2.0 / 24,
                "depth_ppm": 1500.0,
                "power": 0.42,
                "snr": 25.0,
                "odd_even_sigma": 0.8,
            },
            "fold_data": {"phase": [0.0] * 100, "flux": [1.0] * 100, "flux_err": [0.001] * 100},
            "fit": {
                "depth": 0.0015, "duration": 0.083, "ingress_frac": 0.3,
                "center": 0.0, "fit_depth_ppm": 1480.0, "fit_depth_sigma": 22.0,
                "shape_ratio": 0.3, "success": True,
            },
            "vetting": {
                "bls_snr": 25.0, "bls_power": 0.42, "odd_even_sigma": 0.8,
                "secondary_depth_ppm": 50.0, "secondary_sigma": 0.4,
                "shape_ratio": 0.3, "n_transits": 10.0, "depth_scatter": 0.1,
                "fit_depth_ppm": 1480.0, "fit_depth_sigma": 22.0,
            },
        }

        result = run_hunt_batch(db, settings, cache_dir=tmp_path)

    assert result["hunt_run_id"] is not None
    assert len(result["results"]) == 1
    r = result["results"][0]
    assert r["kepoi_name"] == "K00001.01"
    assert r["bls_period"] == pytest.approx(5.001)
    assert r["recovered"] is True      # 5.001 ≈ 5.000 within 1%
    assert r["score"] == pytest.approx(0.95)
    assert result["n_error"] == 0


def test_run_hunt_batch_lc_error(tmp_path):
    """When MAST raises, target is marked error and result has error key."""
    kois = [_make_mock_koi()]
    settings = _make_settings(tmp_path=tmp_path)
    db = _make_mock_db(kois)

    with (
        patch("app.hunt._select_targets", return_value=kois),
        patch("app.hunt.fetch_light_curve", side_effect=ValueError("No data")),
        patch("app.hunt.joblib.load", return_value=_FakeModel()),
    ):
        result = run_hunt_batch(db, settings, cache_dir=tmp_path)

    assert result["n_error"] == 1
    assert result["results"][0]["recovered"] is False
    assert "error" in result["results"][0]


def test_run_hunt_batch_idempotent_skip(tmp_path):
    """Targets already completed are skipped on re-run."""
    koi = _make_mock_koi()
    settings = _make_settings(tmp_path=tmp_path)
    db = _make_mock_db([koi])

    # Simulate target already completed
    completed_target = MagicMock()
    completed_target.hunt_status = "completed"
    completed_target.target_id = 99

    existing_detection = MagicMock()
    existing_detection.period = 5.001
    existing_detection.bls_snr = 25.0
    existing_detection.catalogue_model_score = 0.95

    det_query = MagicMock()
    det_query.filter_by.return_value = det_query
    det_query.order_by.return_value = det_query
    det_query.first.return_value = existing_detection

    target_query = MagicMock()
    target_query.filter_by.return_value = target_query
    target_query.first.return_value = completed_target

    hunt_run_q = MagicMock()

    def _side_effect(model):
        name = model.__name__ if hasattr(model, "__name__") else str(model)
        if "Detection" in str(model):
            return det_query
        if "Target" in str(model):
            return target_query
        return hunt_run_q

    db.query.side_effect = _side_effect

    with (
        patch("app.hunt._select_targets", return_value=[koi]),
        patch("app.hunt.joblib.load", return_value=_FakeModel()),
        patch("app.hunt.fetch_light_curve") as mock_fetch,
    ):
        result = run_hunt_batch(db, settings, force=False, cache_dir=tmp_path)

    # fetch_light_curve must NOT have been called
    mock_fetch.assert_not_called()
    assert result["n_error"] == 0
