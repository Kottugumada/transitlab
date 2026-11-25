"""
Offline tests for the §7 vetting API endpoints.

GET /api/v1/candidates  — ranked detection list
GET /api/v1/candidates/{id} — full vetting data
GET /api/v1/model — model card JSON

All DB calls and MAST/cache calls are mocked; no network required.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db import get_db


CADENCE_DAYS = 29.4 / 60 / 24


# ── Helpers ────────────────────────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc)


def _make_detection(
    det_id: int = 1,
    target_id: int = 1,
    period: float = 5.0,
    depth_ppm: float = 1500.0,
    bls_snr: float = 25.0,
    score: float = 0.92,
) -> MagicMock:
    det = MagicMock()
    det.id = det_id
    det.target_id = target_id
    det.period = period
    det.epoch_bjd = 121.3
    det.duration_h = 2.0
    det.depth_ppm = depth_ppm
    det.bls_snr = bls_snr
    det.bls_power = 0.42
    det.fit_depth_ppm = 1480.0
    det.fit_depth_sigma = 22.0
    det.odd_even_sigma = 0.9
    det.secondary_depth_ppm = 50.0
    det.secondary_sigma = 0.3
    det.shape_ratio = 0.25
    det.n_transits = 10
    det.depth_scatter = 0.08
    det.catalogue_model_score = score
    det.created_at = _now()
    det.fold_data = {
        "phase": list(np.linspace(-0.5, 0.5, 100)),
        "flux": [1.0] * 100,
        "flux_err": [1e-4] * 100,
    }
    det.vetting = {
        "bls_snr": bls_snr, "bls_power": 0.42,
        "odd_even_sigma": 0.9, "secondary_depth_ppm": 50.0,
        "secondary_sigma": 0.3, "shape_ratio": 0.25,
        "n_transits": 10.0, "depth_scatter": 0.08,
        "fit_depth_ppm": 1480.0, "fit_depth_sigma": 22.0,
    }
    return det


def _make_target(kepid: int = 757450, disposition: str = "CONFIRMED") -> MagicMock:
    t = MagicMock()
    t.target_id = 1
    t.kepid = kepid
    t.mission = "Kepler"
    t.label_disposition = disposition
    return t


def _make_koi(kepoi_name: str = "K00001.01", kepid: int = 757450) -> MagicMock:
    k = MagicMock()
    k.kepoi_name = kepoi_name
    k.kepid = kepid
    k.kepler_name = "Kepler-1 b"
    return k


def _db_override_factory(det=None, target=None, koi=None, dets_list=None):
    """Build a get_db override with configurable query results."""
    def override():
        db = MagicMock()

        # db.get(Detection, id) → det
        db.get.return_value = det

        # Provide a full mock query chain
        q = MagicMock()
        q.filter.return_value = q
        q.filter_by.return_value = q
        q.order_by.return_value = q
        q.outerjoin.return_value = q
        q.join.return_value = q
        q.group_by.return_value = q
        q.subquery.return_value = MagicMock()
        q.scalar.return_value = None
        q.first.return_value = koi
        q.offset.return_value = q
        q.limit.return_value = q
        q.count.return_value = len(dets_list) if dets_list is not None else 0
        # all() returns list of (Detection, Target, kepoi_name) tuples
        q.all.return_value = (
            [(d, target, koi.kepoi_name if koi else None) for d in dets_list]
            if dets_list is not None
            else []
        )

        db.query.return_value = q

        # Attach target to det so det.target works
        if det and target:
            det.target = target

        yield db

    return override


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


# ── GET /candidates ────────────────────────────────────────────────────────

def test_candidates_returns_200():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi, dets_list=[det])
    res = TestClient(app).get("/api/v1/candidates")
    assert res.status_code == 200


def test_candidates_response_shape():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi, dets_list=[det])
    data = TestClient(app).get("/api/v1/candidates").json()
    assert "items" in data and "total" in data
    assert isinstance(data["items"], list)


def test_candidates_item_has_required_keys():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi, dets_list=[det])
    data = TestClient(app).get("/api/v1/candidates").json()
    if data["items"]:
        item = data["items"][0]
        for key in ("id", "period", "depth_ppm", "bls_snr", "catalogue_model_score", "disposition"):
            assert key in item, f"Missing key: {key}"


def test_candidates_min_score_filter():
    """min_score param should be accepted without 422."""
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi, dets_list=[det])
    res = TestClient(app).get("/api/v1/candidates?min_score=0.8")
    assert res.status_code == 200


def test_candidates_sort_snr():
    app.dependency_overrides[get_db] = _db_override_factory(dets_list=[])
    res = TestClient(app).get("/api/v1/candidates?sort=snr")
    assert res.status_code == 200


def test_candidates_invalid_sort_rejected():
    res = TestClient(app).get("/api/v1/candidates?sort=bogus")
    assert res.status_code == 422


def test_candidates_empty_list_ok():
    app.dependency_overrides[get_db] = _db_override_factory(dets_list=[])
    data = TestClient(app).get("/api/v1/candidates").json()
    assert data["items"] == []
    assert data["total"] == 0


# ── GET /candidates/{id} ──────────────────────────────────────────────────

def test_candidate_detail_returns_200():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi)
    with patch("app.api.routes._try_odd_even_fold", return_value=(None, None)):
        res = TestClient(app).get("/api/v1/candidates/1")
    assert res.status_code == 200


def test_candidate_detail_response_shape():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi)
    with patch("app.api.routes._try_odd_even_fold", return_value=(None, None)):
        data = TestClient(app).get("/api/v1/candidates/1").json()
    for key in ("id", "period", "depth_ppm", "bls_snr", "fold_data", "vetting", "catalogue_model_score"):
        assert key in data, f"Missing key: {key}"


def test_candidate_detail_fold_data_shape():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi)
    with patch("app.api.routes._try_odd_even_fold", return_value=(None, None)):
        data = TestClient(app).get("/api/v1/candidates/1").json()
    fd = data["fold_data"]
    assert len(fd["phase"]) == 100
    assert all(-0.5 <= p <= 0.5 for p in fd["phase"])


def test_candidate_detail_404():
    app.dependency_overrides[get_db] = _db_override_factory(det=None)
    res = TestClient(app).get("/api/v1/candidates/9999")
    assert res.status_code == 404


def test_candidate_detail_odd_even_returned_when_available():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi)

    fake_fold = {
        "phase": list(np.linspace(-0.5, 0.5, 50)),
        "flux": [1.0] * 50,
        "flux_err": [1e-4] * 50,
    }
    with patch("app.api.routes._try_odd_even_fold", return_value=(fake_fold, fake_fold)):
        data = TestClient(app).get("/api/v1/candidates/1").json()

    assert data["odd_fold"] is not None
    assert data["even_fold"] is not None
    assert len(data["odd_fold"]["phase"]) == 50


def test_candidate_detail_odd_even_null_when_no_cache():
    det = _make_detection()
    tgt = _make_target()
    koi = _make_koi()
    app.dependency_overrides[get_db] = _db_override_factory(det=det, target=tgt, koi=koi)
    with patch("app.api.routes._try_odd_even_fold", return_value=(None, None)):
        data = TestClient(app).get("/api/v1/candidates/1").json()
    assert data["odd_fold"] is None
    assert data["even_fold"] is None


# ── GET /model ─────────────────────────────────────────────────────────────

def test_model_returns_200():
    app.dependency_overrides[get_db] = _db_override_factory()
    with patch("app.api.routes._MODEL_CARD") as mc:
        mc.exists.return_value = False
        res = TestClient(app).get("/api/v1/model")
    assert res.status_code == 200


def test_model_response_shape():
    app.dependency_overrides[get_db] = _db_override_factory()
    with patch("app.api.routes._MODEL_CARD") as mc:
        mc.exists.return_value = False
        data = TestClient(app).get("/api/v1/model").json()
    for key in ("trained_at", "data_snapshot", "honest_features", "metrics", "model_path", "model_card_text"):
        assert key in data, f"Missing key: {key}"


# ── odd_even_fold_data (pure function) ─────────────────────────────────────

def test_odd_even_fold_data_shapes():
    from app.engine.features import odd_even_fold_data
    from app.engine.clean import clean
    from app.engine.detrend import detrend
    from pathlib import Path

    # Load the planet fixture
    import pandas as pd
    df = pd.read_csv(Path("tests/fixtures/kic_planet_5d.csv"))
    time = df["time_bkjd"].to_numpy()
    flux = df["flux"].to_numpy()
    flux_err = df["flux_err"].to_numpy()
    quality = np.zeros(len(time), dtype=np.int32)

    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    defl, _, de = detrend(tc, fc, fec)

    odd, even = odd_even_fold_data(tc, defl, de, period=5.0, epoch=121.3, n_bins=50)
    assert odd is not None
    assert even is not None
    assert len(odd["phase"]) == 50
    assert len(even["phase"]) == 50
    assert all(-0.5 <= p <= 0.5 for p in odd["phase"])


def test_odd_even_fold_data_different_depths():
    """Injected EB-like signal (alternating depths) should give odd ≠ even."""
    from app.engine.features import odd_even_fold_data

    rng = np.random.default_rng(42)
    n = 1500
    period = 5.0
    epoch = 121.3
    dur = 2.0 / 24.0
    time = 120.0 + np.arange(n) * CADENCE_DAYS
    flux = 1.0 + rng.normal(0, 100e-6, n)

    # Inject alternating depths: odd transits 2000 ppm, even 500 ppm
    n_min = int(np.floor((time[0] - epoch) / period))
    n_max = int(np.ceil((time[-1] - epoch) / period))
    for i in range(n_min, n_max + 1):
        t_mid = epoch + i * period
        mask = np.abs(time - t_mid) < dur / 2
        depth = 2000e-6 if i % 2 != 0 else 500e-6
        flux[mask] -= depth

    flux_err = np.full(n, 150e-6)

    odd, even = odd_even_fold_data(time, flux, flux_err, period, epoch, n_bins=50)
    # Odd transits should be deeper (more negative flux, more positive depth)
    odd_min = min(f for f in odd["flux"] if f is not None)
    even_min = min(f for f in even["flux"] if f is not None)
    assert odd_min < even_min, (
        f"Odd minimum flux {odd_min:.6f} should be less than even {even_min:.6f}"
    )
