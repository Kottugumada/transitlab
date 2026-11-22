"""
Trapezoid transit model fit.

Pure function — no network, no database, no Lightkurve I/O.

The trapezoid has 4 free parameters:
  depth        : fractional depth (0 → 1)
  duration     : full duration as a fraction of period (0 → 1)
  ingress_frac : ingress/egress time as a fraction of duration
                 0 → flat-bottomed box (ideal planet)
                 1 → pure V-shape (grazing / EB)
  center       : phase offset of transit centre from 0.0
"""

import numpy as np
from scipy.optimize import minimize


def _trapezoid(phase: np.ndarray, depth: float, duration: float, ingress_frac: float, center: float) -> np.ndarray:
    """
    Evaluate the trapezoid model at the given phases.

    Returns flux array (1.0 outside transit, 1 − depth at flat bottom).
    """
    t = phase - center
    half_dur = duration / 2.0
    ingress = ingress_frac * duration / 2.0  # half-ingress width

    flux = np.ones_like(t)

    # In-transit region (before splitting into ingress/flat/egress)
    in_transit = np.abs(t) < half_dur

    if ingress < 1e-9:
        # Pure box
        flux[in_transit] = 1.0 - depth
    else:
        t_abs = np.abs(t[in_transit])
        flat_half = half_dur - ingress

        # Flat bottom: |t| < flat_half
        is_flat = t_abs < flat_half
        # Ingress/egress ramp: flat_half ≤ |t| < half_dur
        is_ramp = ~is_flat

        flux_in = np.empty(in_transit.sum())
        flux_in[is_flat] = 1.0 - depth
        # Linear ramp from 1 − depth to 1 over ingress width
        ramp_t = t_abs[is_ramp]
        ramp_frac = (ramp_t - flat_half) / ingress  # 0 at start of ramp, 1 at end
        flux_in[is_ramp] = 1.0 - depth * (1.0 - ramp_frac)

        flux[in_transit] = flux_in

    return flux


def fit_trapezoid(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    duration_estimate: float,
) -> dict:
    """
    Fit a trapezoid transit model to a phase-folded, binned light curve.

    Parameters
    ----------
    phase             : phases in [−0.5, +0.5]
    flux              : binned flux (NaN bins are ignored)
    flux_err          : binned uncertainties (NaN bins are ignored)
    duration_estimate : initial guess for transit duration (fraction of period)

    Returns
    -------
    dict with keys:
        depth         : best-fit depth (fractional, > 0 means dimming)
        duration      : best-fit duration (fraction of period)
        ingress_frac  : best-fit ingress fraction (0=box, 1=V)
        center        : best-fit centre offset (phase)
        fit_depth_ppm : depth in ppm
        fit_depth_sigma: depth / depth_uncertainty (from covariance)
        shape_ratio   : ingress_frac (alias for the vetting metric)
        success       : bool — optimiser converged
    """
    # Drop NaN bins
    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if good.sum() < 4:
        return _failed_fit()

    p = phase[good]
    f = flux[good]
    e = flux_err[good]

    # Rough depth estimate from median near transit
    near_transit = np.abs(p) < duration_estimate
    if near_transit.sum() >= 2:
        depth0 = float(max(0.0, 1.0 - np.median(f[near_transit])))
    else:
        depth0 = 1e-3

    x0 = [depth0, duration_estimate, 0.3, 0.0]

    bounds = [
        (1e-6, 0.5),   # depth
        (1e-4, 0.5),   # duration (fraction of period)
        (0.0, 1.0),    # ingress_frac
        (-0.1, 0.1),   # center offset
    ]

    def chi2(params):
        depth, dur, ing, cen = params
        model = _trapezoid(p, depth, dur, ing, cen)
        return float(np.sum(((f - model) / e) ** 2))

    result = minimize(chi2, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if not result.success and result.fun > chi2(x0) * 2:
        return _failed_fit()

    depth, duration, ingress_frac, center = result.x

    # Approximate depth uncertainty from residuals
    model_flux = _trapezoid(p, depth, duration, ingress_frac, center)
    residuals = f - model_flux
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    # Depth uncertainty ≈ rms / sqrt(n_in_transit)
    n_in = int(np.sum(np.abs(p - center) < duration / 2.0))
    depth_err = rms / np.sqrt(max(1, n_in))

    return {
        "depth": float(depth),
        "duration": float(duration),
        "ingress_frac": float(ingress_frac),
        "center": float(center),
        "fit_depth_ppm": float(depth * 1e6),
        "fit_depth_sigma": float(depth / max(depth_err, 1e-9)),
        "shape_ratio": float(ingress_frac),
        "success": True,
    }


def _failed_fit() -> dict:
    return {
        "depth": 0.0,
        "duration": 0.0,
        "ingress_frac": 0.0,
        "center": 0.0,
        "fit_depth_ppm": 0.0,
        "fit_depth_sigma": 0.0,
        "shape_ratio": 0.0,
        "success": False,
    }
