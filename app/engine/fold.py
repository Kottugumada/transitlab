"""
Phase-fold and bin a light curve.

Pure function — no network, no database, no Lightkurve I/O.
"""

import numpy as np


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve so that transit centre sits at phase = 0.

    Parameters
    ----------
    time    : observation times (days)
    flux    : detrended, normalised flux
    flux_err: per-cadence uncertainties
    period  : orbital period (days)
    epoch   : reference transit mid-time (days, same scale as time)

    Returns
    -------
    (phase, flux, flux_err)
    phase is in [−0.5, +0.5]; data are sorted by phase.
    """
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5

    order = np.argsort(phase)
    return phase[order], flux[order], flux_err[order]


def bin_fold(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a phase-folded light curve using inverse-variance weighting.

    Parameters
    ----------
    phase   : phases in [−0.5, +0.5] (sorted)
    flux    : flux values
    flux_err: per-cadence uncertainties
    n_bins  : number of bins (default 100)

    Returns
    -------
    (bin_centers, binned_flux, binned_err)
    Bins with no data are set to NaN.
    """
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    binned_flux = np.full(n_bins, np.nan)
    binned_err = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.sum() == 0:
            continue
        w = 1.0 / flux_err[mask] ** 2
        binned_flux[i] = float(np.sum(w * flux[mask]) / np.sum(w))
        binned_err[i] = float(1.0 / np.sqrt(np.sum(w)))

    return bin_centers, binned_flux, binned_err
