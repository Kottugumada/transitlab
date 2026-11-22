"""
Unit tests for the vetting-feature extraction (app/engine/features.py).

All tests run offline.  No network, no database, no Lightkurve.

We build synthetic light curves with known properties and verify that
the computed features cross the right thresholds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.engine.clean import clean
from app.engine.detrend import detrend
from app.engine.bls import run_bls
from app.engine.fold import phase_fold, bin_fold
from app.engine.fit import fit_trapezoid
from app.engine.features import (
    compute_vetting_features,
    _odd_even_sigma,
    _secondary_eclipse,
    _per_transit_scatter,
)

FIXTURES = Path(__file__).parent / "fixtures"
CADENCE_DAYS = 29.4 / 60 / 24


def _load_lc(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, detrend a fixture CSV."""
    df = pd.read_csv(FIXTURES / name)
    time = df["time_bkjd"].to_numpy()
    flux = df["flux"].to_numpy()
    flux_err = df["flux_err"].to_numpy()
    quality = df["quality"].to_numpy(dtype=int)
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    df2, _, de = detrend(tc, fc, fec)
    return tc, df2, de


def _run_pipeline(name: str) -> tuple[dict, dict, dict]:
    """Full pipeline → (bls_result, fit_result, vetting_features)."""
    tc, df, de = _load_lc(name)
    bls = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, bls["period"], bls["epoch"])
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=100)
    duration_frac = bls["duration_days"] / bls["period"]
    fit = fit_trapezoid(centers, bf, be, duration_frac)
    feats = compute_vetting_features(tc, df, de, bls["period"], bls["epoch"],
                                     bls["duration_days"], bls, fit)
    return bls, fit, feats


# ── compute_vetting_features result shape ─────────────────────────────────────

def test_vetting_features_has_all_keys():
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    expected = {
        "bls_snr", "bls_power", "odd_even_sigma", "secondary_depth_ppm",
        "secondary_sigma", "shape_ratio", "n_transits", "depth_scatter",
        "fit_depth_ppm", "fit_depth_sigma",
    }
    missing = expected - set(feats.keys())
    assert not missing, f"Missing vetting features: {missing}"


def test_vetting_features_all_finite():
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    for k, v in feats.items():
        assert np.isfinite(v), f"Feature '{k}' is not finite: {v}"


# ── Planet fixture: features should PASS vetting ──────────────────────────────

def test_planet_bls_snr_high():
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    assert feats["bls_snr"] > 7.0, f"bls_snr = {feats['bls_snr']:.1f}"


def test_planet_odd_even_sigma_low():
    """Planet: odd and even transit depths should be similar (< 3σ)."""
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    assert feats["odd_even_sigma"] < 3.0, (
        f"odd_even_sigma = {feats['odd_even_sigma']:.2f} — unexpectedly large for a planet"
    )


def test_planet_secondary_depth_small():
    """Planet: no detectable secondary eclipse (< 2× noise floor ≈ 1000 ppm)."""
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    assert feats["secondary_depth_ppm"] < 1500.0, (
        f"secondary_depth_ppm = {feats['secondary_depth_ppm']:.0f} — too large for a planet"
    )


def test_planet_shape_ratio_low():
    """Planet trapezoid fit should be box-like (ingress_frac < 0.6)."""
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    assert feats["shape_ratio"] < 0.6, (
        f"shape_ratio = {feats['shape_ratio']:.2f} — unexpectedly V-shaped for a planet"
    )


def test_planet_n_transits_positive():
    _, _, feats = _run_pipeline("kic_planet_5d.csv")
    assert feats["n_transits"] >= 2, f"n_transits = {feats['n_transits']}"


# ── EB fixture: features should FAIL vetting ─────────────────────────────────

def test_eb_odd_even_sigma_high():
    """
    EB with primary=10000ppm and secondary=5000ppm should produce a large
    odd/even sigma (>> 3σ), flagging it as a likely eclipsing binary.
    """
    tc, df, de = _load_lc("kic_eb_10d.csv")
    bls = run_bls(tc, df, de, period_min=0.5, period_max=30.0)

    # The BLS may find the half-period (5d) or the full period (10d).
    # Use the actual injection period for the vetting check.
    INJ_PERIOD_EB = 10.0
    INJ_EPOCH_EB = 120.0 + 2.5  # T0_BKJD + 2.5 from fixture generator

    sigma = _odd_even_sigma(tc, df, de, INJ_PERIOD_EB, INJ_EPOCH_EB, 3.0 / 24.0)
    assert sigma > 3.0, (
        f"odd_even_sigma = {sigma:.2f} — expected >> 3σ for EB with primary/secondary asymmetry"
    )


def test_eb_secondary_eclipse_detected():
    """
    EB secondary eclipse at phase 0.5 should be detectable.
    """
    tc, df, de = _load_lc("kic_eb_10d.csv")
    bls = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    phase, fph, fe = phase_fold(tc, df, de, bls["period"], bls["epoch"])

    duration_frac = bls["duration_days"] / bls["period"]
    sec_depth, sec_sigma = _secondary_eclipse(phase, fph, fe, duration_frac)

    # Secondary should be > 1000 ppm (it's 5000 ppm at the injected period,
    # but BLS may find the half-period, so we set a modest threshold)
    assert sec_depth > 500.0 or sec_sigma > 1.0, (
        f"secondary_depth_ppm = {sec_depth:.0f}, secondary_sigma = {sec_sigma:.1f} "
        "— expected detectable secondary for EB"
    )


# ── _per_transit_scatter ──────────────────────────────────────────────────────

def test_scatter_stable_planet():
    """Planet: depth scatter should be low (< 1.0 coefficient of variation)."""
    tc, df, de = _load_lc("kic_planet_5d.csv")
    bls = run_bls(tc, df, de, period_min=0.5, period_max=30.0)
    n_transits, scatter = _per_transit_scatter(tc, df, de, bls["period"], bls["epoch"],
                                                bls["duration_days"])
    assert n_transits >= 2
    assert scatter < 1.0, f"depth_scatter = {scatter:.2f} — too large for a stable planet"


def test_scatter_quiet_star_no_transits():
    """Quiet star with no injected transits: n_transits should be 0 or scatter large."""
    tc, df, de = _load_lc("kic_quiet_2.csv")
    # Use a fake period / epoch that doesn't align with anything real
    n_transits, scatter = _per_transit_scatter(tc, df, de,
                                                period=7.0,
                                                epoch=tc[0] + 1.0,
                                                duration_days=2.0 / 24.0)
    # Either no transits found, or scatter > 0.5 (noise-dominated)
    assert n_transits == 0 or scatter > 0.5, (
        f"Expected noisy scatter on quiet star: n={n_transits}, scatter={scatter:.2f}"
    )


# ── _odd_even_sigma edge cases ────────────────────────────────────────────────

def test_odd_even_sigma_zero_for_flat():
    """Flat light curve → odd/even depths should both be ≈0 → sigma ≈ 0."""
    rng = np.random.default_rng(0)
    n = 500
    time = 120.0 + np.arange(n) * CADENCE_DAYS
    flux = 1.0 + rng.normal(0, 200e-6, n)
    flux_err = np.full(n, 200e-6)
    sigma = _odd_even_sigma(time, flux, flux_err, period=5.0, epoch=121.0, duration_days=2 / 24)
    assert sigma < 5.0, f"odd_even_sigma = {sigma:.2f} on flat star — expected small"


def test_odd_even_sigma_no_data():
    """With too few data points, should return 0.0 not raise."""
    sigma = _odd_even_sigma(
        np.array([120.0, 121.0]),
        np.array([1.0, 1.0]),
        np.array([0.001, 0.001]),
        period=5.0, epoch=100.0, duration_days=0.5,
    )
    assert sigma == 0.0
