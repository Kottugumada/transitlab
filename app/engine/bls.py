"""
Box Least Squares (BLS) periodogram wrapper.

Pure function — no network, no database, no Lightkurve I/O.
Uses astropy.timeseries.BoxLeastSquares with Quantity arrays so units
are explicit and cannot be accidentally mixed.
"""

import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares


# BLS period grid: 0.5 – 100 days, 10 000 periods (default)
# Transit duration grid: 0.01 – 0.2 (fraction of period)
_N_PERIODS = 10_000
_DURATION_FRACS = np.linspace(0.01, 0.20, 20)


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_min: float = 0.5,
    period_max: float = 100.0,
    n_periods: int = _N_PERIODS,
) -> dict:
    """
    Run BLS on a detrended light curve and return the best-fit parameters.

    Parameters
    ----------
    time        : observation times in days
    flux        : detrended, normalised flux
    flux_err    : per-cadence uncertainties
    period_min  : minimum trial period in days (default 0.5)
    period_max  : maximum trial period in days (default 100)
    n_periods   : number of trial periods (default 10 000)

    Returns
    -------
    dict with keys:
        period       : best-fit period (days)
        epoch        : best-fit transit mid-time (days, same scale as time)
        duration_days: best-fit transit duration (days)
        depth_ppm    : transit depth in ppm
        power        : BLS power at best period
        snr          : (depth / depth_uncertainty) — approximate S/N ratio
        odd_even_sigma: |odd_depth − even_depth| / pooled_err
                        (preliminary; full odd/even lives in features.py)
    """
    # Cap period_max to half the baseline to require at least 2 transits
    baseline = float(time[-1] - time[0])
    period_max = min(period_max, baseline / 2.0)
    if period_max <= period_min:
        raise ValueError(
            f"Period range [{period_min}, {period_max}] is invalid "
            f"(baseline = {baseline:.1f} d).  Need at least 2 transits."
        )

    # Uniform FREQUENCY grid — not uniform period grid.
    # A uniform period grid undersamples short periods: at P=5 d over a 1461-d
    # baseline the required frequency step is P²/T ≈ 0.017 d, but a linear
    # 2000-point grid from 0.5–50 d has step=0.025 d → misses the peak, pushing
    # BLS to find the 4P harmonic where the grid is denser.
    # Reversing to frequency space gives equal resolution at every period.
    freqs = np.linspace(1.0 / period_max, 1.0 / period_min, n_periods)
    periods = (1.0 / freqs[::-1]) * u.day  # ascending period order
    durations = _DURATION_FRACS  # fractions; BLS accepts unitless fractions

    model = BoxLeastSquares(time * u.day, flux, dy=flux_err)
    result = model.power(
        periods,
        durations,
        objective="snr",
    )

    best_idx = int(np.argmax(result.power))
    best_period = float(result.period[best_idx].to(u.day).value)
    best_power = float(result.power[best_idx])

    # Retrieve detailed parameters at the best period
    stats = model.compute_stats(
        result.period[best_idx],
        result.duration[best_idx],
        result.transit_time[best_idx],
    )

    depth = float(stats["depth"][0])
    depth_err = float(stats["depth"][1]) if stats["depth"][1] > 0 else 1e-9
    snr = depth / depth_err

    duration_days = float(result.duration[best_idx].to(u.day).value)
    epoch = float(result.transit_time[best_idx].to(u.day).value)

    # Quick odd/even screen; full vetting lives in features.py
    odd_even_sigma = _quick_odd_even(time, flux, flux_err, best_period, epoch, duration_days)

    return {
        "period": best_period,
        "epoch": epoch,
        "duration_days": duration_days,
        "depth_ppm": depth * 1e6,
        "power": best_power,
        "snr": snr,
        "odd_even_sigma": odd_even_sigma,
    }


def _quick_odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
) -> float:
    """
    Estimate odd/even transit depth difference in units of sigma.
    Used for a quick BLS-level screen; full vetting is in features.py.
    """
    half = duration_days / 2.0
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period

    odd_depths = []
    even_depths = []

    # Find transit numbers
    n_min = int(np.floor((time[0] - epoch) / period))
    n_max = int(np.ceil((time[-1] - epoch) / period))

    for n in range(n_min, n_max + 1):
        t_mid = epoch + n * period
        in_transit = np.abs(time - t_mid) < half
        if in_transit.sum() < 2:
            continue
        d = float(np.mean(1.0 - flux[in_transit]))
        if n % 2 == 0:
            even_depths.append(d)
        else:
            odd_depths.append(d)

    if not odd_depths or not even_depths:
        return 0.0

    od = np.mean(odd_depths)
    ed = np.mean(even_depths)
    sigma = np.std(flux) / np.sqrt(len(flux) / (n_max - n_min + 1 + 1e-9))
    if sigma < 1e-9:
        return 0.0
    return float(abs(od - ed) / sigma)
