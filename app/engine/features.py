"""
Vetting features derived from a phase-folded light curve.

Pure function — no network, no database, no Lightkurve I/O.

These features feed the classifier and populate Detection.vetting JSONB.
They are independent of the BLS/fit results (computed from raw phase data)
so they remain valid if the pipeline is re-run with different parameters.
"""

import numpy as np

from app.engine.fold import phase_fold, bin_fold


def compute_vetting_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
    bls_result: dict,
    fit_result: dict,
) -> dict:
    """
    Compute transit vetting features from a cleaned, detrended light curve.

    Parameters
    ----------
    time          : cleaned, detrended observation times (days)
    flux          : cleaned, detrended flux (median ≈ 1.0)
    flux_err      : per-cadence uncertainties
    period        : best-fit orbital period (days) from BLS
    epoch         : best-fit transit mid-time (days) from BLS
    duration_days : best-fit transit duration (days) from BLS
    bls_result    : dict returned by run_bls()
    fit_result    : dict returned by fit_trapezoid()

    Returns
    -------
    dict with keys (all float):
        bls_snr           : BLS signal-to-noise ratio
        bls_power         : BLS power at best period
        odd_even_sigma    : odd vs even transit depth difference (σ)
        secondary_depth_ppm: depth of strongest out-of-transit dip (ppm)
        secondary_sigma   : secondary / noise_floor
        shape_ratio       : ingress_frac from trapezoid fit (0=planet, 1=EB)
        n_transits        : number of transit events observed
        depth_scatter     : RMS of per-transit depth residuals / mean depth
        fit_depth_ppm     : trapezoid fit depth (ppm)
        fit_depth_sigma   : fit depth / fit depth uncertainty
    """
    duration_frac = duration_days / period  # duration as fraction of period

    phase, f, fe = phase_fold(time, flux, flux_err, period, epoch)

    # ── Odd/even depth difference ─────────────────────────────────────────
    odd_even_sigma = _odd_even_sigma(time, flux, flux_err, period, epoch, duration_days)

    # ── Secondary eclipse check ───────────────────────────────────────────
    secondary_depth_ppm, secondary_sigma = _secondary_eclipse(
        phase, f, fe, duration_frac
    )

    # ── Per-transit depth scatter ─────────────────────────────────────────
    n_transits, depth_scatter = _per_transit_scatter(
        time, flux, flux_err, period, epoch, duration_days
    )

    return {
        "bls_snr": bls_result.get("snr", 0.0),
        "bls_power": bls_result.get("power", 0.0),
        "odd_even_sigma": odd_even_sigma,
        "secondary_depth_ppm": secondary_depth_ppm,
        "secondary_sigma": secondary_sigma,
        "shape_ratio": fit_result.get("shape_ratio", 0.0),
        "n_transits": float(n_transits),
        "depth_scatter": depth_scatter,
        "fit_depth_ppm": fit_result.get("fit_depth_ppm", 0.0),
        "fit_depth_sigma": fit_result.get("fit_depth_sigma", 0.0),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _odd_even_sigma(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
) -> float:
    """
    Compute |depth_odd − depth_even| / sigma, where sigma is the pooled
    uncertainty on the two mean depths.

    An eclipsing binary with unequal primary/secondary (but secondary at
    phase 0.5) will have alternating transit depths, giving sigma >> 3.
    """
    half = duration_days / 2.0
    n_min = int(np.floor((time[0] - epoch) / period))
    n_max = int(np.ceil((time[-1] - epoch) / period))

    odd_depths, odd_vars = [], []
    even_depths, even_vars = [], []

    for n in range(n_min, n_max + 1):
        t_mid = epoch + n * period
        mask = np.abs(time - t_mid) < half
        if mask.sum() < 2:
            continue
        w = 1.0 / flux_err[mask] ** 2
        d = float(1.0 - np.sum(w * flux[mask]) / np.sum(w))
        v = float(1.0 / np.sum(w))
        if n % 2 == 0:
            even_depths.append(d)
            even_vars.append(v)
        else:
            odd_depths.append(d)
            odd_vars.append(v)

    if not odd_depths or not even_depths:
        return 0.0

    od = float(np.mean(odd_depths))
    ed = float(np.mean(even_depths))

    # Variance on the means
    var_od = float(np.sum(odd_vars)) / len(odd_depths) ** 2
    var_ed = float(np.sum(even_vars)) / len(even_depths) ** 2

    sigma_diff = np.sqrt(var_od + var_ed)
    if sigma_diff < 1e-12:
        return 0.0
    return float(abs(od - ed) / sigma_diff)


def _secondary_eclipse(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    duration_frac: float,
) -> tuple[float, float]:
    """
    Look for a secondary eclipse at phase ≈ ±0.5 (opposite the transit).

    For a planet the secondary is undetectable at our precision.
    For an EB the secondary is a large dip at phase ±0.5.

    Returns
    -------
    (secondary_depth_ppm, secondary_sigma)
    """
    half_dur = duration_frac / 2.0

    # Mask out the primary transit region
    primary_mask = np.abs(phase) < half_dur * 2.0

    # Scan for the deepest dip outside the primary
    # Divide out-of-transit into windows of width = transit duration
    n_windows = max(1, int(1.0 / (duration_frac * 2)))
    window_width = duration_frac

    best_depth_ppm = 0.0
    best_sigma = 0.0

    p_out = phase[~primary_mask]
    f_out = flux[~primary_mask]
    fe_out = flux_err[~primary_mask]

    if len(p_out) == 0:
        return 0.0, 0.0

    # Slide a window centred at phase ±0.5 (most likely EB secondary location)
    for centre in [0.5, -0.5]:
        win = np.abs(p_out - centre) < half_dur
        if win.sum() < 2:
            continue
        w = 1.0 / fe_out[win] ** 2
        depth = float(1.0 - np.sum(w * f_out[win]) / np.sum(w))
        depth_err = float(1.0 / np.sqrt(np.sum(w)))
        depth_ppm = depth * 1e6
        sigma = depth / max(depth_err, 1e-9)
        if depth_ppm > best_depth_ppm:
            best_depth_ppm = depth_ppm
            best_sigma = sigma

    return float(best_depth_ppm), float(best_sigma)


def odd_even_fold_data(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    n_bins: int = 50,
) -> tuple[dict | None, dict | None]:
    """
    Return (odd_fold, even_fold) each as {phase, flux, flux_err} dicts, or None
    if fewer than 2 cadences fall in that group.

    Cadences are assigned to odd or even by the transit number n =
    round((t − epoch) / period).  For an eclipsing binary with unequal
    primary and secondary the two groups show different depths; for a
    planet they look identical.
    """
    n_transit = np.round((time - epoch) / period).astype(int)
    odd_mask = (n_transit % 2 != 0)
    even_mask = ~odd_mask

    def _fold_subset(mask: np.ndarray) -> dict | None:
        if mask.sum() < 2:
            return None
        t_s, f_s, fe_s = time[mask], flux[mask], flux_err[mask]
        phase, fph, fe = phase_fold(t_s, f_s, fe_s, period, epoch)
        centers, bf, be = bin_fold(phase, fph, fe, n_bins=n_bins)

        def _list(arr: np.ndarray) -> list:
            return [None if not np.isfinite(v) else float(v) for v in arr]

        return {
            "phase": [float(v) for v in centers],
            "flux": _list(bf),
            "flux_err": _list(be),
        }

    return _fold_subset(odd_mask), _fold_subset(even_mask)


def _per_transit_scatter(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
) -> tuple[int, float]:
    """
    Count transit events and compute the scatter in per-transit depths.

    depth_scatter = std(per_transit_depths) / mean(per_transit_depths)
    (coefficient of variation)

    A real planet has stable depths; a blended EB or spot-crossing shows scatter.
    """
    half = duration_days / 2.0
    n_min = int(np.floor((time[0] - epoch) / period))
    n_max = int(np.ceil((time[-1] - epoch) / period))

    depths = []
    for n in range(n_min, n_max + 1):
        t_mid = epoch + n * period
        mask = np.abs(time - t_mid) < half
        if mask.sum() < 2:
            continue
        d = float(1.0 - np.median(flux[mask]))
        depths.append(d)

    n_transits = len(depths)
    if n_transits < 2:
        return n_transits, 0.0

    mean_d = float(np.mean(depths))
    if abs(mean_d) < 1e-9:
        return n_transits, 0.0

    scatter = float(np.std(depths) / abs(mean_d))
    return n_transits, scatter
