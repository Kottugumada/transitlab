"""
Step 1 of the transit engine: load and clean a light curve.

Pure function — no network, no database, no Lightkurve I/O.
Accepts plain numpy arrays (produced by the MAST client wrapper in Step 5).
"""

import numpy as np


def clean(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    quality_bitmask: int = 0b0101001010111111,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Remove bad cadences and normalise flux to median = 1.0.

    Parameters
    ----------
    time          : observation times (days, BKJD or BTJD)
    flux          : raw flux array (any units)
    flux_err      : per-cadence uncertainties; estimated from successive
                    differences if not supplied
    quality       : integer quality flags (Kepler/TESS bitmask); if None,
                    only NaN filtering is applied
    quality_bitmask: which quality bits to reject (default: Lightkurve's
                    "default" bitmask — bits known to flag bad cadences)

    Returns
    -------
    (time, flux, flux_err, n_dropped)
    flux and flux_err are normalised so median(flux) = 1.0.
    n_dropped is the number of cadences removed.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Build keep-mask: start by requiring finite values
    mask = np.isfinite(flux) & np.isfinite(time)

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        mask &= np.isfinite(flux_err) & (flux_err > 0)
    else:
        # Estimate noise from successive differences (robust, no outlier sensitivity)
        finite_flux = flux[np.isfinite(flux)]
        if len(finite_flux) > 1:
            sigma_est = float(np.median(np.abs(np.diff(finite_flux))) / np.sqrt(2))
        else:
            sigma_est = 1e-4
        flux_err = np.full_like(flux, sigma_est)

    if quality is not None:
        quality = np.asarray(quality, dtype=int)
        mask &= (quality & quality_bitmask) == 0

    n_dropped = int((~mask).sum())

    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    if len(flux) == 0:
        raise ValueError("All cadences were dropped after quality filtering.")

    # Normalise to median = 1.0
    median = float(np.median(flux))
    if median <= 0:
        raise ValueError(f"Flux median is {median:.4g} — cannot normalise to 1.0.")
    flux = flux / median
    flux_err = flux_err / median

    return time, flux, flux_err, n_dropped
