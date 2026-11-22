"""
Orchestrate the transit engine pipeline.

Two entry points:
  fold_at_period — Explorer mode: use archive period/epoch, skip BLS
  hunt_pipeline  — Hunt mode (§6.2): run BLS, then fold at best period
"""

import numpy as np
from app.engine.bls import run_bls
from app.engine.clean import clean
from app.engine.detrend import detrend
from app.engine.fold import phase_fold, bin_fold
from app.engine.fit import fit_trapezoid
from app.engine.features import compute_vetting_features


def fold_at_period(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
    n_bins: int = 100,
    window_length: int = 101,
) -> dict:
    """
    Clean → detrend → fold at given period → bin → fit → vetting features.

    Does NOT run BLS; uses the caller-supplied period and epoch (from the
    NASA Exoplanet Archive catalogue).

    Returns
    -------
    dict:
        n_cadences_raw   : int
        n_cadences_clean : int
        n_dropped        : int
        fold_data        : {"phase": [...], "flux": [...|None], "flux_err": [...|None]}
        fit              : trapezoid fit parameters
        vetting          : vetting feature dict
    """
    tc, fc, fec, n_dropped = clean(time, flux, flux_err, quality)
    df, _trend, de = detrend(tc, fc, fec, window_length=window_length)

    phase, fph, fe = phase_fold(tc, df, de, period, epoch)
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=n_bins)

    duration_frac = duration_days / period
    fit = fit_trapezoid(centers, bf, be, duration_frac)

    # Pass a minimal BLS placeholder (SNR from fit depth significance)
    bls_placeholder = {
        "snr": fit.get("fit_depth_sigma", 0.0),
        "power": 0.0,
    }
    vetting = compute_vetting_features(
        tc, df, de, period, epoch, duration_days,
        bls_placeholder, fit,
    )

    # Serialise fold arrays: replace np.nan with None for JSON safety
    def _to_json_list(arr: np.ndarray) -> list:
        return [None if not np.isfinite(v) else float(v) for v in arr]

    return {
        "n_cadences_raw": int(len(time)),
        "n_cadences_clean": int(len(tc)),
        "n_dropped": int(n_dropped),
        "fold_data": {
            "phase": [float(v) for v in centers],
            "flux": _to_json_list(bf),
            "flux_err": _to_json_list(be),
        },
        "fit": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in fit.items()},
        "vetting": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in vetting.items()},
    }


def hunt_pipeline(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    period_min: float = 0.5,
    period_max: float = 50.0,
    n_periods: int = 2000,
    n_bins: int = 100,
    window_length: int = 101,
) -> dict:
    """
    Clean → detrend → BLS → fold at best period → bin → fit → vetting.

    Used by the §6.2 hunt batch.  Unlike fold_at_period, this runs BLS
    to discover the best period from the data itself.

    Returns
    -------
    dict:
        n_cadences_raw   : int
        n_cadences_clean : int
        n_dropped        : int
        bls              : BLS result dict (period, epoch, duration_days, depth_ppm, power, snr, …)
        fold_data        : {"phase": [...], "flux": [...|None], "flux_err": [...|None]}
        fit              : trapezoid fit parameters
        vetting          : vetting feature dict
    """
    tc, fc, fec, n_dropped = clean(time, flux, flux_err, quality)
    df, _trend, de = detrend(tc, fc, fec, window_length=window_length)

    bls = run_bls(tc, df, de, period_min=period_min, period_max=period_max, n_periods=n_periods)

    phase, fph, fe = phase_fold(tc, df, de, bls["period"], bls["epoch"])
    centers, bf, be = bin_fold(phase, fph, fe, n_bins=n_bins)

    duration_frac = bls["duration_days"] / bls["period"]
    fit = fit_trapezoid(centers, bf, be, duration_frac)

    vetting = compute_vetting_features(
        tc, df, de, bls["period"], bls["epoch"], bls["duration_days"],
        bls, fit,
    )

    def _to_json_list(arr: np.ndarray) -> list:
        return [None if not np.isfinite(v) else float(v) for v in arr]

    def _sanitise(d: dict) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}

    return {
        "n_cadences_raw": int(len(time)),
        "n_cadences_clean": int(len(tc)),
        "n_dropped": int(n_dropped),
        "bls": _sanitise(bls),
        "fold_data": {
            "phase": [float(v) for v in centers],
            "flux": _to_json_list(bf),
            "flux_err": _to_json_list(be),
        },
        "fit": _sanitise(fit),
        "vetting": _sanitise(vetting),
    }
