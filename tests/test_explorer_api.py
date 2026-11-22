"""
Offline tests for GET /api/v1/explorer/{kepoi_name}.

The database and MAST fetcher are both mocked so no network calls are made.
app.dependency_overrides replaces get_db so the endpoint never touches Neon.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.api.main import app
from app.db import get_db

CADENCE_DAYS = 29.4 / 60 / 24


def _synthetic_lc(seed: int = 7, with_transit: bool = True) -> dict:
    """Return a synthetic LC dict matching what mast_client.fetch_light_curve returns."""
    rng = np.random.default_rng(seed)
    n = 1500
    time = 120.0 + np.arange(n) * CADENCE_DAYS
    flux = 1.0 + rng.normal(0, 200e-6, n)
    flux_err = np.full(n, 200e-6)
    quality = np.zeros(n, dtype=np.int32)

    if with_transit:
        # Inject a simple transit: P=5d, depth=1500ppm, duration=2h
        period = 5.0
        epoch = 121.3
        dur = 2.0 / 24.0
        half = dur / 2.0
        for n_t in range(-2, int((time[-1] - time[0]) / period) + 2):
            t_mid = epoch + n_t * period
            mask = np.abs(time - t_mid) < half
            flux[mask] -= 1500e-6

    return {"time": time, "flux": flux, "flux_err": flux_err, "quality": quality}


def _make_koi_row(
    kepoi_name: str = "K00001.01",
    kepid: int = 757450,
    period: float = 2.4706133,
    time0: float = 120.4,
    duration: float = 2.175,  # hours
    disposition: str = "CONFIRMED",
    kepler_name: str = "Kepler-1 b",
    depth: float = 15800.0,
):
    koi = MagicMock()
    koi.kepoi_name = kepoi_name
    koi.kepid = kepid
    koi.period = period
    koi.time0 = time0
    koi.duration = duration
    koi.disposition = disposition
    koi.kepler_name = kepler_name
    koi.depth = depth
    return koi


def _db_override(koi_row):
    """Return a get_db dependency override that yields a mock DB returning koi_row."""
    def override():
        mock_db = MagicMock()
        mock_db.get.return_value = koi_row
        yield mock_db
    return override


@pytest.fixture(autouse=True)
def clear_overrides():
    """Ensure dependency overrides don't leak between tests."""
    yield
    app.dependency_overrides.clear()


def test_explorer_returns_200():
    koi = _make_koi_row()
    app.dependency_overrides[get_db] = _db_override(koi)
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    assert res.status_code == 200


def test_explorer_response_shape():
    app.dependency_overrides[get_db] = _db_override(_make_koi_row())
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    data = res.json()
    expected_keys = {
        "kepoi_name", "kepid", "kepler_name", "disposition",
        "period", "epoch", "duration_days", "depth_ppm",
        "n_cadences_raw", "n_cadences_clean",
        "fold_data", "fit", "vetting",
    }
    assert expected_keys <= set(data.keys()), f"Missing keys: {expected_keys - set(data.keys())}"


def test_explorer_fold_data_shape():
    app.dependency_overrides[get_db] = _db_override(_make_koi_row())
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    fd = res.json()["fold_data"]
    assert len(fd["phase"]) == 100
    assert len(fd["flux"]) == 100
    assert len(fd["flux_err"]) == 100


def test_explorer_phase_range():
    app.dependency_overrides[get_db] = _db_override(_make_koi_row())
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    phases = res.json()["fold_data"]["phase"]
    assert all(-0.5 <= p <= 0.5 for p in phases), "Phase values out of [-0.5, 0.5]"


def test_explorer_vetting_has_expected_keys():
    app.dependency_overrides[get_db] = _db_override(_make_koi_row())
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    v = res.json()["vetting"]
    for key in ("odd_even_sigma", "shape_ratio", "n_transits", "depth_scatter",
                "fit_depth_ppm", "fit_depth_sigma"):
        assert key in v, f"Vetting key {key!r} missing"


def test_explorer_404_for_unknown_koi():
    app.dependency_overrides[get_db] = _db_override(None)  # db.get returns None → 404
    res = TestClient(app).get("/api/v1/explorer/K99999.01")
    assert res.status_code == 404


def test_explorer_422_for_null_period():
    koi = _make_koi_row()
    koi.period = None
    app.dependency_overrides[get_db] = _db_override(koi)
    res = TestClient(app).get("/api/v1/explorer/K00001.01")
    assert res.status_code == 422


def test_explorer_n_cadences_clean_leq_raw():
    app.dependency_overrides[get_db] = _db_override(_make_koi_row())
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    data = res.json()
    assert data["n_cadences_clean"] <= data["n_cadences_raw"]


def test_explorer_duration_days_correct():
    """duration_days in response should be koi.duration / 24."""
    koi = _make_koi_row(duration=4.8)  # 4.8 hours
    app.dependency_overrides[get_db] = _db_override(koi)
    with patch("app.api.routes.fetch_light_curve", return_value=_synthetic_lc()):
        res = TestClient(app).get("/api/v1/explorer/K00001.01")
    assert abs(res.json()["duration_days"] - 4.8 / 24) < 1e-9
