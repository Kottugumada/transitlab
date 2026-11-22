"""
Offline tests for app/mast_client.py.

Lightkurve is mocked so no network calls are made.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_mock_lc(n: int = 1500, seed: int = 42):
    """Return a mock Lightkurve LightCurve-like object."""
    rng = np.random.default_rng(seed)
    mock_lc = MagicMock()
    mock_lc.time.value = 120.0 + np.arange(n) * (29.4 / 60 / 24)
    mock_lc.flux.value = 1.0 + rng.normal(0, 200e-6, n)
    mock_lc.flux_err.value = np.full(n, 200e-6)
    mock_lc.quality.value = np.zeros(n, dtype=int)
    mock_lc.quality.value[5] = 16  # inject one bad cadence flag
    return mock_lc


def _make_mock_collection(n: int = 1500):
    """Return a mock SearchResult / LightCurveCollection."""
    mock_lc = _make_mock_lc(n)

    mock_coll = MagicMock()
    mock_coll.stitch.return_value = mock_lc

    mock_search = MagicMock()
    mock_search.__len__.return_value = 4   # 4 Kepler quarters found
    mock_search.download_all.return_value = mock_coll

    return mock_search, mock_lc


def test_fetch_returns_required_keys(tmp_path):
    mock_search, _ = _make_mock_collection()
    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        result = fetch_light_curve(757450, cache_dir=tmp_path)

    assert set(result.keys()) == {"time", "flux", "flux_err", "quality"}


def test_fetch_arrays_are_numpy(tmp_path):
    mock_search, _ = _make_mock_collection()
    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        result = fetch_light_curve(757450, cache_dir=tmp_path)

    for k in ("time", "flux", "flux_err", "quality"):
        assert isinstance(result[k], np.ndarray), f"{k!r} is not an ndarray"


def test_fetch_caches_to_disk(tmp_path):
    mock_search, _ = _make_mock_collection()
    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        fetch_light_curve(757450, cache_dir=tmp_path)

    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 1, f"Expected 1 cache file, found {len(cache_files)}"


def test_fetch_uses_cache_on_second_call(tmp_path):
    """Second call must NOT invoke search_lightcurve again."""
    mock_search, _ = _make_mock_collection()
    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        fetch_light_curve(757450, cache_dir=tmp_path)
        fetch_light_curve(757450, cache_dir=tmp_path)
        # search_lightcurve should have been called exactly once
        assert mock_lk.search_lightcurve.call_count == 1


def test_fetch_force_refresh_re_downloads(tmp_path):
    mock_search, _ = _make_mock_collection()
    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        fetch_light_curve(757450, cache_dir=tmp_path)
        fetch_light_curve(757450, cache_dir=tmp_path, force_refresh=True)
        assert mock_lk.search_lightcurve.call_count == 2


def test_fetch_raises_when_no_data(tmp_path):
    mock_search = MagicMock()
    mock_search.__len__.return_value = 0
    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        with pytest.raises(ValueError, match="No.*light curves found"):
            fetch_light_curve(999999, cache_dir=tmp_path)


def test_fetch_handles_missing_flux_err(tmp_path):
    """When flux_err is absent, the client should estimate it, not crash."""
    mock_lc = _make_mock_lc()
    mock_lc.flux_err = None   # simulate no error column

    mock_coll = MagicMock()
    mock_coll.stitch.return_value = mock_lc
    mock_search = MagicMock()
    mock_search.__len__.return_value = 1
    mock_search.download_all.return_value = mock_coll

    with patch("app.mast_client.lk") as mock_lk:
        mock_lk.search_lightcurve.return_value = mock_search
        from app.mast_client import fetch_light_curve
        result = fetch_light_curve(12345, cache_dir=tmp_path)

    assert np.all(np.isfinite(result["flux_err"]))
    assert np.all(result["flux_err"] > 0)
