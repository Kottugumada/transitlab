from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import check_db_connection, get_db
from app.api.schemas import HealthResponse, ExplorerResponse, FoldData
from app.models import KoiCatalogue
from app.mast_client import fetch_light_curve
from app.engine.pipeline import fold_at_period

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    db_status = check_db_connection()
    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        db=db_status,
        last_sync_at=None,
        last_hunt_at=None,
        model_version=None,
    )


@router.get("/explorer/{kepoi_name}", response_model=ExplorerResponse)
def explorer(kepoi_name: str, db: Session = Depends(get_db)):
    """
    Fetch a confirmed Kepler system's light curve, fold at the archive period,
    and return the binned folded curve for the UI.
    """
    koi = db.get(KoiCatalogue, kepoi_name)
    if koi is None:
        raise HTTPException(status_code=404, detail=f"KOI {kepoi_name!r} not found in catalogue.")

    if koi.kepid is None:
        raise HTTPException(status_code=422, detail=f"KOI {kepoi_name!r} has no kepid — cannot fetch light curve.")
    if koi.period is None or koi.time0 is None:
        raise HTTPException(status_code=422, detail=f"KOI {kepoi_name!r} is missing period or epoch.")

    # duration is stored in hours; engine expects days
    duration_days = (koi.duration or 2.0) / 24.0

    lc = fetch_light_curve(koi.kepid)
    result = fold_at_period(
        lc["time"], lc["flux"], lc["flux_err"], lc["quality"],
        period=koi.period,
        epoch=koi.time0,
        duration_days=duration_days,
    )

    return ExplorerResponse(
        kepoi_name=koi.kepoi_name,
        kepid=koi.kepid,
        kepler_name=koi.kepler_name,
        disposition=koi.disposition,
        period=koi.period,
        epoch=koi.time0,
        duration_days=duration_days,
        depth_ppm=koi.depth,
        n_cadences_raw=result["n_cadences_raw"],
        n_cadences_clean=result["n_cadences_clean"],
        fold_data=FoldData(**result["fold_data"]),
        fit=result["fit"],
        vetting=result["vetting"],
    )
