"""
TAP client for the NASA Exoplanet Archive.

Endpoint: https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<ADQL>&format=csv
No auth required for public tables.

Design rules (from spec §2.1):
- Never SELECT * — column lists are explicit constants defined here.
- TAP is slow; every fetch is cached as a local CSV under data/cache/.
- Cache is keyed by (table, columns hash) so a column-list change busts it.
- Never call from the API request path — batch jobs only.
- Retries with exponential back-off; raises on persistent failure.
"""

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

log = logging.getLogger(__name__)

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ── Explicit column lists (never SELECT *) ────────────────────────────────────
# Verified against NASA Exoplanet Archive TAP table documentation 2026-11-21.
# koi_tce_plnt_num added so we can join back to TCE if needed; not stored in DB.

KOI_COLUMNS = [
    # Identifiers
    "kepoi_name",
    "kepid",
    "kepler_name",
    "koi_tce_plnt_num",
    # Labels (§5 leakage audit: disposition-derived columns are EXCLUDED from
    # the honest-features model but kept here so we can store them and train
    # the all-features model for the leakage comparison)
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    # Orbital / transit geometry — photometry-derivable (honest features)
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_ror",          # Rp/R*
    "koi_impact",
    # Detection statistics — photometry-derivable (honest features)
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_bin_oedp_sig", # odd/even depth significance
    # Stellar parameters — not photometry-derived but not disposition-derived
    "koi_steff",
    "koi_srad",
    "koi_slogg",
    "koi_kepmag",
    # Robovetter flags (enumerated from cumulative table column docs 2026-11-21):
    # These encode human/algorithmic vetting decisions and are DISPOSITION-DERIVED
    # — they are excluded from the honest-features model (§5 leakage audit).
    # Stored in DB (features JSONB) so the all-features model can use them.
    "koi_fpflag_nt",   # Not Transit-Like
    "koi_fpflag_ss",   # Stellar Eclipse
    "koi_fpflag_co",   # Centroid Offset
    "koi_fpflag_ec",   # Ephemeris Match Contamination
]

PS_COLUMNS = [
    "pl_name",
    "hostname",
    "pl_orbper",   # orbital period (days)
    "pl_tranmid",  # transit mid-time (BJD)
    "pl_trandur",  # transit duration (hours)
    "pl_trandep",  # transit depth (ppm)
    "pl_rade",     # planet radius (R_earth)
    "pl_bmasse",   # planet mass (M_earth), best available
    "sy_vmag",     # host V magnitude
    "ra",
    "dec",
]

CACHE_DIR = Path("data/cache")


def _cache_path(table: str, columns: list[str]) -> Path:
    col_hash = hashlib.md5(",".join(sorted(columns)).encode()).hexdigest()[:8]
    return CACHE_DIR / f"{table}_{col_hash}.csv"


def fetch_tap(
    table: str,
    columns: list[str],
    where: str = "",
    *,
    max_retries: int = 3,
    timeout_s: int = 600,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch a TAP table into a DataFrame, caching the result as a local CSV.

    Parameters
    ----------
    table        : TAP table name (e.g. 'cumulative', 'ps')
    columns      : explicit list of columns to SELECT
    where        : optional ADQL WHERE clause (without the 'WHERE' keyword)
    max_retries  : number of attempts before raising
    timeout_s    : per-request socket timeout in seconds
    force_refresh: ignore cached CSV and re-fetch
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(table, columns)

    if cache.exists() and not force_refresh:
        log.info("TAP cache hit: %s", cache)
        return pd.read_csv(cache, low_memory=False)

    col_list = ",".join(columns)
    adql = f"SELECT {col_list} FROM {table}"
    if where:
        adql += f" WHERE {where}"

    params = urlencode({"query": adql, "format": "csv"})
    url = f"{TAP_BASE}?{params}"
    log.info("TAP fetch: %s (timeout=%ds)", adql[:120], timeout_s)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(url, timeout=timeout_s) as resp:
                raw = resp.read()
            # TAP returns an error XML/VOTable if the query is bad; detect it
            if raw.lstrip()[:5] in (b"<?xml", b"<VOTA"):
                raise ValueError(
                    f"TAP returned an error document instead of CSV.\n"
                    f"First 500 bytes: {raw[:500]!r}\n"
                    f"Query was: {adql}"
                )
            cache.write_bytes(raw)
            df = pd.read_csv(cache, low_memory=False)
            log.info("TAP fetched %d rows from %s", len(df), table)
            return df
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning("TAP attempt %d/%d failed (%s); retrying in %ds", attempt, max_retries, exc, wait)
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError(
        f"TAP fetch failed after {max_retries} attempts for table '{table}'"
    ) from last_exc
