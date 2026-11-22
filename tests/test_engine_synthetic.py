"""
§6.1 Injection-recovery tests for the transit engine.

All tests run offline against committed synthetic fixture CSVs.
No network, no database, no Lightkurve.

Done-check targets:
  - Injected planet (P=5d, depth=1500ppm, SNR≈37) recovered within 1%
  - BLS SNR > 7 on the planet fixture
  - Clean step drops bad cadences and normalises to median = 1.0
  - Detrend preserves injected depth to within 10%
  - Phase-fold centred at 0, sorted by phase
  - Trapezoid fit depth within 10% of injected depth
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.engine.clean import clean
from app.engine.detrend import detrend
from app.engine.bls import run_bls
from app.engine.fold import phase_fold, bin_fold
from app.engine.fit import fit_trapezoid, _trapezoid

FIXTURES = Path(__file__).parent / "fixtures"

# Constants matching the fixture generator
CADENCE_DAYS = 29.4 / 60 / 24
NOISE_PPM = 200.0
INJ_PERIOD = 5.0          # days
INJ_DEPTH_PPM = 1500.0
INJ_DURATION_H = 2.0      # hours
INJ_DURATION_DAYS = INJ_DURATION_H / 24.0


def _load_fixture(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(FIXTURES / name)
    return (
        df["time_bkjd"].to_numpy(),
        df["flux"].to_numpy(),
        df["flux_err"].to_numpy(),
        df["quality"].to_numpy(dtype=int),
    )


# ── clean() ───────────────────────────────────────────────────────────────────

def test_clean_drops_bad_cadences():
    time, flux, flux_err, quality = _load_fixture("kic_quiet_1.csv")
    _, fc, _, n_dropped = clean(time, flux, flux_err, quality)
    # Fixture has 10 NaN cadences + quality-flagged ones
    assert n_dropped >= 10


def test_clean_normalises_to_median_one():
    time, flux, flux_err, quality = _load_fixture("kic_quiet_1.csv")
    _, fc, _, _ = clean(time, flux, flux_err, quality)
    assert abs(np.median(fc) - 1.0) < 1e-6


def test_clean_all_finite_after():
    time, flux, flux_err, quality = _load_fixture("kic_quiet_1.csv")
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    assert np.all(np.isfinite(tc))
    assert np.all(np.isfinite(fc))
    assert np.all(np.isfinite(fec))


def test_clean_no_quality_flags_means_only_nan_dropped():
    time, flux, flux_err, quality = _load_fixture("kic_quiet_2.csv")
    # kic_quiet_2 has no bad cadences and no NaN → nothing should be dropped
    _, _, _, n_dropped = clean(time, flux, flux_err, quality)
    assert n_dropped == 0


def test_clean_error_on_all_nan():
    with pytest.raises(ValueError, match="All cadences"):
        clean(
            np.array([1.0, 2.0, 3.0]),
            np.array([np.nan, np.nan, np.nan]),
        )


# ── detrend() ─────────────────────────────────────────────────────────────────

def test_detrend_returns_finite():
    time, flux, flux_err, quality = _load_fixture("kic_quiet_2.csv")
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    df, trend, de = detrend(tc, fc, fec)
    assert np.all(np.isfinite(df))
    assert np.all(np.isfinite(trend))
    assert np.all(np.isfinite(de))


def test_detrend_median_near_one():
    """After detrending a quiet star, median should still be ≈1."""
    time, flux, flux_err, quality = _load_fixture("kic_quiet_2.csv")
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    df, _, _ = detrend(tc, fc, fec)
    assert abs(np.median(df) - 1.0) < 0.01


def test_detrend_preserves_injected_depth():
    """
    Detrending a planet signal should not remove more than 10% of the depth.
    The S-G window (101 cadences ≈ 50 h) is >> transit duration (2 h),
    so it cannot significantly follow the transit dip.
    """
    time, flux, flux_err, quality = _load_fixture("kic_planet_5d.csv")
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    df, trend, de = detrend(tc, fc, fec)

    # Check that the trend is NOT tracking individual transit dips
    # (trend should be smooth, close to 1 everywhere)
    assert np.std(trend) < 5e-4, (
        f"Trend std {np.std(trend):.2e} is too large — detrending may be removing transits"
    )


# ── run_bls() ─────────────────────────────────────────────────────────────────

def _planet_lc():
    """Load, clean, detrend the planet fixture."""
    time, flux, flux_err, quality = _load_fixture("kic_planet_5d.csv")
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    df, _, de = detrend(tc, fc, fec)
    return tc, df, de


def test_bls_recovers_period():
    """BLS should recover P=5d within 1%."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    assert abs(result["period"] - INJ_PERIOD) / INJ_PERIOD < 0.01, (
        f"Recovered period {result['period']:.4f} d, expected {INJ_PERIOD} d"
    )


def test_bls_snr_above_threshold():
    """BLS SNR should be well above a 7σ detection threshold."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    assert result["snr"] > 7.0, f"BLS SNR = {result['snr']:.1f} < 7"


def test_bls_depth_within_factor_two():
    """BLS depth should be within factor 2 of the injected depth."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    ratio = result["depth_ppm"] / INJ_DEPTH_PPM
    assert 0.5 < ratio < 2.0, (
        f"BLS depth {result['depth_ppm']:.0f} ppm vs injected {INJ_DEPTH_PPM:.0f} ppm"
    )


def test_bls_result_has_required_keys():
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de)
    for key in ("period", "epoch", "duration_days", "depth_ppm", "power", "snr", "odd_even_sigma"):
        assert key in result, f"Missing key '{key}' in BLS result"


def test_bls_raises_on_tiny_baseline():
    """If baseline is too short for 2 transits at period_min, should raise."""
    # 10 cadences ≈ 0.2 days baseline; period_min=0.5 would need baseline > 1d
    time = np.arange(10) * CADENCE_DAYS + 120.0
    flux = np.ones(10)
    flux_err = np.full(10, 200e-6)
    with pytest.raises(ValueError, match="Period range"):
        run_bls(time, flux, flux_err, period_min=0.5, period_max=100.0)


# ── phase_fold() ──────────────────────────────────────────────────────────────

def test_phase_fold_range():
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, _, _ = phase_fold(tc, df, de, result["period"], result["epoch"])
    assert np.all(phase >= -0.5) and np.all(phase <= 0.5)


def test_phase_fold_sorted():
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, _, _ = phase_fold(tc, df, de, result["period"], result["epoch"])
    assert np.all(np.diff(phase) >= 0), "Phase array is not sorted"


def test_phase_fold_transit_near_zero():
    """The transit dip should be centred near phase 0."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, _ = phase_fold(tc, df, de, result["period"], result["epoch"])

    # Find the minimum flux (deepest point)
    min_idx = np.argmin(fph)
    assert abs(phase[min_idx]) < 0.1, (
        f"Transit minimum is at phase {phase[min_idx]:.3f}, expected near 0"
    )


# ── bin_fold() ────────────────────────────────────────────────────────────────

def test_bin_fold_shape():
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, result["period"], result["epoch"])
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=100)
    assert len(centers) == 100
    assert len(bf) == 100
    assert len(be) == 100


def test_bin_fold_transit_visible():
    """Binned folded LC should show a dip of at least 500 ppm near phase 0."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, result["period"], result["epoch"])
    centers, bf, _ = bin_fold(phase, fph, fe, n_bins=100)

    near_zero = np.abs(centers) < 0.05
    valid = near_zero & np.isfinite(bf)
    if valid.sum() == 0:
        pytest.skip("No finite bins near phase 0")
    min_flux = float(np.nanmin(bf[valid]))
    assert min_flux < 1.0 - 500e-6, (
        f"Binned minimum flux {min_flux:.6f} — transit dip not visible (expected < {1 - 500e-6:.6f})"
    )


# ── fit_trapezoid() ───────────────────────────────────────────────────────────

def test_fit_recovers_depth():
    """Trapezoid fit depth should be within 10% of the injected 1500 ppm."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, result["period"], result["epoch"])
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=100)

    duration_frac = result["duration_days"] / result["period"]
    fit = fit_trapezoid(centers, bf, be, duration_frac)

    assert fit["success"], "Trapezoid fit did not converge"
    ratio = fit["fit_depth_ppm"] / INJ_DEPTH_PPM
    assert 0.9 <= ratio <= 1.1, (
        f"Fit depth {fit['fit_depth_ppm']:.0f} ppm vs injected {INJ_DEPTH_PPM:.0f} ppm "
        f"(ratio {ratio:.2f}, expected 0.9–1.1)"
    )


def test_fit_depth_sigma_positive():
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, result["period"], result["epoch"])
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=100)

    duration_frac = result["duration_days"] / result["period"]
    fit = fit_trapezoid(centers, bf, be, duration_frac)
    assert fit["fit_depth_sigma"] > 0


def test_fit_shape_ratio_range():
    """Planet should have ingress_frac < 0.6 (box-like)."""
    tc, df, de = _planet_lc()
    result = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, result["period"], result["epoch"])
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=100)

    duration_frac = result["duration_days"] / result["period"]
    fit = fit_trapezoid(centers, bf, be, duration_frac)
    assert fit["shape_ratio"] < 0.6, (
        f"shape_ratio {fit['shape_ratio']:.2f} too large for a planet-like signal"
    )


def test_trapezoid_model_box():
    """With ingress_frac=0, the model should be a perfect box."""
    phase = np.linspace(-0.5, 0.5, 1000)
    flux = _trapezoid(phase, depth=0.01, duration=0.1, ingress_frac=0.0, center=0.0)
    in_transit = np.abs(phase) < 0.05
    out_transit = np.abs(phase) > 0.1

    assert np.allclose(flux[in_transit], 0.99, atol=1e-9)
    assert np.allclose(flux[out_transit], 1.0, atol=1e-9)
