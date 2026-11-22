"""
Catalogue sync: pull KOI cumulative table + confirmed planets from NASA TAP,
upsert into Neon.

Idempotency: uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE so re-running
never creates duplicates.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.archive_client import KOI_COLUMNS, PS_COLUMNS, fetch_tap
from app.models import KoiCatalogue, PsPlanet

log = logging.getLogger(__name__)

# Columns that are disposition-derived (excluded from honest-features model).
# Stored in DB for the leakage-comparison model only.  Enumerated here so the
# classifier training step can read this list directly.
DISPOSITION_DERIVED_FEATURES = [
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]

# Columns used as honest features (photometry + stellar params, no vetting flags).
# This is the headline model in §5.
HONEST_FEATURE_COLUMNS = [
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_impact",
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_bin_oedp_sig",
    "koi_steff",
    "koi_srad",
    "koi_slogg",
    "koi_kepmag",
]


def _nan_to_none(val):
    """Convert numpy/pandas NaN/NaT to None for SQLAlchemy."""
    if val is None:
        return None
    try:
        if np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _row_to_koi(row: pd.Series, now: datetime) -> dict:
    """Map a DataFrame row to a koi_catalogue upsert dict."""
    # Build the full features blob for the classifier (all numeric KOI columns)
    features = {}
    for col in KOI_COLUMNS:
        if col in ("kepoi_name", "kepid", "kepler_name", "koi_tce_plnt_num",
                   "koi_disposition", "koi_pdisposition"):
            continue
        v = _nan_to_none(row.get(col))
        if v is not None:
            features[col] = float(v) if isinstance(v, (int, float, np.floating)) else v

    return {
        "kepoi_name": str(row["kepoi_name"]),
        "kepid": _nan_to_none(row.get("kepid")),
        "kepler_name": _nan_to_none(row.get("kepler_name")),
        "disposition": _nan_to_none(row.get("koi_disposition")),
        "pdisposition": _nan_to_none(row.get("koi_pdisposition")),
        "koi_score": _nan_to_none(row.get("koi_score")),
        "period": _nan_to_none(row.get("koi_period")),
        "time0": _nan_to_none(row.get("koi_time0bk")),
        "duration": _nan_to_none(row.get("koi_duration")),
        "depth": _nan_to_none(row.get("koi_depth")),
        "ror": _nan_to_none(row.get("koi_ror")),
        "impact": _nan_to_none(row.get("koi_impact")),
        "model_snr": _nan_to_none(row.get("koi_model_snr")),
        "odd_even_sig": _nan_to_none(row.get("koi_bin_oedp_sig")),
        "steff": _nan_to_none(row.get("koi_steff")),
        "srad": _nan_to_none(row.get("koi_srad")),
        "kepmag": _nan_to_none(row.get("koi_kepmag")),
        "features": features,
        "synced_at": now,
    }


def _row_to_ps(row: pd.Series, now: datetime) -> dict:
    """Map a DataFrame row to a ps_planets upsert dict."""
    return {
        "pl_name": str(row["pl_name"]),
        "hostname": _nan_to_none(row.get("hostname")),
        "period": _nan_to_none(row.get("pl_orbper")),
        "tranmid": _nan_to_none(row.get("pl_tranmid")),
        "trandur": _nan_to_none(row.get("pl_trandur")),
        "trandep": _nan_to_none(row.get("pl_trandep")),
        "rade": _nan_to_none(row.get("pl_rade")),
        "sy_vmag": _nan_to_none(row.get("sy_vmag")),
        "ra": _nan_to_none(row.get("ra")),
        "dec": _nan_to_none(row.get("dec")),
        "synced_at": now,
    }


def _upsert_koi_batch(db: Session, rows: list[dict]) -> None:
    stmt = insert(KoiCatalogue).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["kepoi_name"],
        set_={k: stmt.excluded[k] for k in rows[0] if k != "kepoi_name"},
    )
    db.execute(stmt)


def _upsert_ps_batch(db: Session, rows: list[dict]) -> None:
    stmt = insert(PsPlanet).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["pl_name"],
        set_={k: stmt.excluded[k] for k in rows[0] if k != "pl_name"},
    )
    db.execute(stmt)


UPSERT_BATCH_SIZE = 500


def sync_koi_catalogue(
    db: Session,
    *,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch the KOI cumulative table from NASA TAP and upsert into koi_catalogue.
    Returns a summary dict with row counts and label distribution.
    """
    if cache_dir:
        import app.archive_client as ac
        ac.CACHE_DIR = cache_dir

    log.info("Fetching KOI cumulative table from NASA TAP...")
    df = fetch_tap("cumulative", KOI_COLUMNS, force_refresh=force_refresh)

    # Verify we got the expected columns; warn if any are missing
    missing_cols = [c for c in KOI_COLUMNS if c not in df.columns]
    if missing_cols:
        log.warning("TAP response missing expected columns: %s", missing_cols)

    now = datetime.now(timezone.utc)
    total = len(df)
    log.info("Upserting %d KOI rows...", total)

    batch: list[dict] = []
    for _, row in df.iterrows():
        batch.append(_row_to_koi(row, now))
        if len(batch) >= UPSERT_BATCH_SIZE:
            _upsert_koi_batch(db, batch)
            db.commit()
            batch = []
    if batch:
        _upsert_koi_batch(db, batch)
        db.commit()

    # Label distribution
    dist = df["koi_disposition"].value_counts().to_dict() if "koi_disposition" in df.columns else {}
    log.info("KOI sync complete. Total=%d  Distribution=%s", total, dist)
    return {"table": "koi_catalogue", "rows_fetched": total, "label_distribution": dist}


def sync_ps_planets(
    db: Session,
    *,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch the confirmed-planets table (ps, default_flag=1) from NASA TAP
    and upsert into ps_planets.
    Returns a summary dict with row count.
    """
    if cache_dir:
        import app.archive_client as ac
        ac.CACHE_DIR = cache_dir

    log.info("Fetching confirmed-planets table from NASA TAP...")
    df = fetch_tap("ps", PS_COLUMNS, where="default_flag=1", force_refresh=force_refresh)

    missing_cols = [c for c in PS_COLUMNS if c not in df.columns]
    if missing_cols:
        log.warning("TAP response missing expected columns: %s", missing_cols)

    now = datetime.now(timezone.utc)
    total = len(df)
    log.info("Upserting %d ps_planets rows...", total)

    batch: list[dict] = []
    for _, row in df.iterrows():
        batch.append(_row_to_ps(row, now))
        if len(batch) >= UPSERT_BATCH_SIZE:
            _upsert_ps_batch(db, batch)
            db.commit()
            batch = []
    if batch:
        _upsert_ps_batch(db, batch)
        db.commit()

    log.info("ps_planets sync complete. Total=%d", total)
    return {"table": "ps_planets", "rows_fetched": total}
