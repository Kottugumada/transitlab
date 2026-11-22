"""
MAST light curve fetcher — thin Lightkurve wrapper with disk cache.

No Lightkurve objects escape this module; callers receive plain numpy arrays.
Network only on first call; subsequent calls read from data/cache/lc/.
"""

import logging
import numpy as np
import lightkurve as lk
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default cache directory (created on first use)
_DEFAULT_CACHE = Path("data/cache/lc")


def fetch_light_curve(
    kepid: int,
    cache_dir: Optional[Path] = None,
    mission: str = "Kepler",
    cadence: str = "long",
    force_refresh: bool = False,
) -> dict:
    """
    Download and stitch all Kepler long-cadence quarters for kepid.

    Parameters
    ----------
    kepid         : Kepler Input Catalogue ID
    cache_dir     : directory for cached .npz files (default data/cache/lc/)
    mission       : 'Kepler' or 'TESS'
    cadence       : 'long' or 'short'
    force_refresh : ignore cache and re-download

    Returns
    -------
    dict with keys: time, flux, flux_err, quality (all np.ndarray, float64 / int32)
    time is in BKJD (Barycentric Kepler Julian Date = BJD − 2454833.0)
    flux is normalised (median ≈ 1.0 from Lightkurve's default normalisation)
    """
    cache = Path(cache_dir or _DEFAULT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"{kepid}_{mission.lower()}_{cadence}.npz"

    if cache_file.exists() and not force_refresh:
        log.debug("LC cache hit: %s", cache_file)
        data = np.load(cache_file)
        return {k: data[k] for k in ("time", "flux", "flux_err", "quality")}

    log.info("Fetching %s %s light curve for KIC %d from MAST …", mission, cadence, kepid)
    search = lk.search_lightcurve(f"KIC {kepid}", mission=mission, cadence=cadence)
    if len(search) == 0:
        raise ValueError(
            f"No {mission} {cadence}-cadence light curves found for KIC {kepid}. "
            "Check the KIC ID and that MAST is reachable."
        )

    lc_coll = search.download_all()
    lc = lc_coll.stitch()

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    if hasattr(lc, "flux_err") and lc.flux_err is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)
        # Replace NaN flux_err with median estimate
        finite_mask = np.isfinite(flux_err) & (flux_err > 0)
        if finite_mask.sum() > 0:
            fill = float(np.median(flux_err[finite_mask]))
        else:
            fill = float(np.nanmedian(np.abs(np.diff(flux[np.isfinite(flux)]))) / np.sqrt(2))
        flux_err = np.where(finite_mask, flux_err, fill)
    else:
        # Estimate from successive differences (robust to outliers)
        finite_flux = flux[np.isfinite(flux)]
        sigma_est = float(np.median(np.abs(np.diff(finite_flux))) / np.sqrt(2)) if len(finite_flux) > 1 else 1e-4
        flux_err = np.full_like(flux, sigma_est)

    if hasattr(lc, "quality") and lc.quality is not None:
        quality = np.asarray(lc.quality.value, dtype=np.int32)
    else:
        quality = np.zeros(len(flux), dtype=np.int32)

    np.savez(cache_file, time=time, flux=flux, flux_err=flux_err, quality=quality)
    log.info("Cached %d cadences for KIC %d → %s", len(time), kepid, cache_file)

    return {"time": time, "flux": flux, "flux_err": flux_err, "quality": quality}
