"""
Savitzky-Golay detrending for transit light curves.

Pure function — no network, no database, no Lightkurve I/O.
"""

import numpy as np
from scipy.signal import savgol_filter


def detrend(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    window_length: int = 101,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove long-term stellar variability with a Savitzky-Golay filter.

    The window_length should be odd and chosen so the filter passes
    transit durations (~2–8 hours = ~4–16 Kepler long-cadence points)
    but removes variability on longer timescales.  The default of 101
    cadences ≈ 50 hours works for most Kepler targets.

    Parameters
    ----------
    time          : cleaned observation times (days)
    flux          : cleaned, normalised flux (median = 1.0)
    flux_err      : cleaned per-cadence uncertainties
    window_length : S-G window in cadences (must be odd; default 101)
    polyorder     : S-G polynomial order (default 3)

    Returns
    -------
    (detrended_flux, trend, detrended_err)
    detrended_flux = flux / trend (transit depth preserved)
    trend = savgol_filter(flux, ...)
    detrended_err  = flux_err / trend (propagated)
    """
    if window_length % 2 == 0:
        window_length += 1  # enforce odd

    n = len(flux)
    if n < window_length:
        # Window larger than data: use all data as the window (odd)
        window_length = n if n % 2 == 1 else n - 1
        # polyorder must be < window_length
        polyorder = min(polyorder, window_length - 1)

    trend = savgol_filter(flux, window_length=window_length, polyorder=polyorder)

    # Guard against near-zero trend (shouldn't happen on normalised flux)
    trend = np.where(np.abs(trend) < 1e-6, 1.0, trend)

    detrended_flux = flux / trend
    detrended_err = flux_err / trend

    return detrended_flux, trend, detrended_err
