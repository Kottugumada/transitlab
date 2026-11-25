import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import check_db_connection, get_db
from app.api.schemas import (
    CandidateDetail,
    CandidateItem,
    CandidatesResponse,
    ExplorerResponse,
    FoldData,
    HealthResponse,
    ModelCardResponse,
)
from app.models import (
    Detection,
    HuntRun,
    KoiCatalogue,
    Target,
    TrainingRun,
)
from app.mast_client import fetch_light_curve
from app.engine.pipeline import fold_at_period
from app.engine.clean import clean
from app.engine.detrend import detrend
from app.engine.features import odd_even_fold_data

log = logging.getLogger(__name__)

router = APIRouter()

_MODEL_CARD = Path("models/MODEL_CARD.md")


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    db_status = check_db_connection()

    # last catalogue sync
    last_sync = db.query(func.max(KoiCatalogue.synced_at)).scalar()

    # last completed hunt run
    last_hunt = (
        db.query(HuntRun.finished_at)
        .filter(HuntRun.status == "completed")
        .order_by(HuntRun.finished_at.desc())
        .limit(1)
        .scalar()
    )

    # latest training run model path as version
    tr = (
        db.query(TrainingRun)
        .order_by(TrainingRun.trained_at.desc())
        .first()
    )
    model_version = (Path(tr.model_path).name if tr and tr.model_path else None)

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        db=db_status,
        last_sync_at=last_sync,
        last_hunt_at=last_hunt,
        model_version=model_version,
    )


# ── /candidates ───────────────────────────────────────────────────────────────

@router.get("/candidates", response_model=CandidatesResponse)
def list_candidates(
    min_score: float | None = Query(None, ge=0.0, le=1.0),
    mission: str | None = Query(None),
    sort: str = Query("score", pattern="^(score|snr|period|depth)$"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Return detections ranked by catalogue model score (or BLS SNR).
    One detection per target, joined with KoiCatalogue for the KOI name.
    """
    # Subquery: representative kepoi_name per kepid (alphabetically first)
    rep_koi = (
        db.query(
            KoiCatalogue.kepid.label("kepid"),
            func.min(KoiCatalogue.kepoi_name).label("kepoi_name"),
        )
        .filter(KoiCatalogue.kepoi_name.isnot(None))
        .group_by(KoiCatalogue.kepid)
        .subquery()
    )

    q = (
        db.query(Detection, Target, rep_koi.c.kepoi_name)
        .join(Target, Detection.target_id == Target.target_id)
        .outerjoin(rep_koi, Target.kepid == rep_koi.c.kepid)
    )

    if min_score is not None:
        q = q.filter(Detection.catalogue_model_score >= min_score)
    if mission:
        q = q.filter(Target.mission == mission)

    if sort == "score":
        q = q.order_by(Detection.catalogue_model_score.desc().nullslast())
    elif sort == "snr":
        q = q.order_by(Detection.bls_snr.desc().nullslast())
    elif sort == "period":
        q = q.order_by(Detection.period.asc().nullslast())
    elif sort == "depth":
        q = q.order_by(Detection.depth_ppm.desc().nullslast())

    total = q.count()
    rows = q.offset(offset).limit(limit).all()

    items = [
        CandidateItem(
            id=det.id,
            target_id=det.target_id,
            kepid=tgt.kepid,
            kepoi_name=kepoi_name,
            disposition=tgt.label_disposition,
            mission=tgt.mission,
            period=det.period,
            depth_ppm=det.depth_ppm,
            bls_snr=det.bls_snr,
            n_transits=det.n_transits,
            catalogue_model_score=det.catalogue_model_score,
            created_at=det.created_at,
        )
        for det, tgt, kepoi_name in rows
    ]
    return CandidatesResponse(items=items, total=total)


# ── /candidates/{id} ──────────────────────────────────────────────────────────

@router.get("/candidates/{detection_id}", response_model=CandidateDetail)
def get_candidate(detection_id: int, db: Session = Depends(get_db)):
    """
    Full vetting data for a single detection.
    Attempts to compute odd/even fold data from the cached light curve on
    disk; returns null for odd_fold / even_fold if the cache file is absent.
    """
    det = db.get(Detection, detection_id)
    if det is None:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found.")

    tgt = det.target
    koi = (
        db.query(KoiCatalogue)
        .filter(KoiCatalogue.kepid == tgt.kepid)
        .order_by(KoiCatalogue.kepoi_name)
        .first()
    ) if tgt.kepid else None

    # Reconstruct fold_data / fit from stored JSONB
    fold_data_raw = det.fold_data or {}
    fold_data = (
        FoldData(
            phase=fold_data_raw.get("phase", []),
            flux=fold_data_raw.get("flux", []),
            flux_err=fold_data_raw.get("flux_err", []),
        )
        if fold_data_raw
        else None
    )

    # On-demand odd/even fold from cached LC
    odd_fold, even_fold = None, None
    if tgt.kepid and det.period and det.epoch_bjd:
        odd_raw, even_raw = _try_odd_even_fold(
            tgt.kepid, det.period, det.epoch_bjd
        )
        if odd_raw:
            odd_fold = FoldData(**odd_raw)
        if even_raw:
            even_fold = FoldData(**even_raw)

    # Extract fit from vetting JSONB (stored there by hunt_pipeline)
    vet = det.vetting or {}
    fit = {
        "fit_depth_ppm": vet.get("fit_depth_ppm"),
        "fit_depth_sigma": vet.get("fit_depth_sigma"),
        "shape_ratio": vet.get("shape_ratio"),
    }

    return CandidateDetail(
        id=det.id,
        target_id=det.target_id,
        kepid=tgt.kepid,
        kepoi_name=koi.kepoi_name if koi else None,
        kepler_name=koi.kepler_name if koi else None,
        disposition=tgt.label_disposition,
        mission=tgt.mission,
        period=det.period,
        epoch_bjd=det.epoch_bjd,
        duration_h=det.duration_h,
        depth_ppm=det.depth_ppm,
        bls_snr=det.bls_snr,
        bls_power=det.bls_power,
        fit_depth_ppm=det.fit_depth_ppm,
        fit_depth_sigma=det.fit_depth_sigma,
        odd_even_sigma=det.odd_even_sigma,
        secondary_depth_ppm=det.secondary_depth_ppm,
        secondary_sigma=det.secondary_sigma,
        shape_ratio=det.shape_ratio,
        n_transits=det.n_transits,
        depth_scatter=det.depth_scatter,
        catalogue_model_score=det.catalogue_model_score,
        fold_data=fold_data,
        odd_fold=odd_fold,
        even_fold=even_fold,
        vetting=det.vetting,
        fit=fit,
        created_at=det.created_at,
    )


def _try_odd_even_fold(
    kepid: int,
    period: float,
    epoch: float,
    cache_root: Path = Path("data/cache/lc"),
    n_bins: int = 50,
) -> tuple[dict | None, dict | None]:
    """Load cached LC and compute odd/even phase-folded bins. Returns (None, None) on any error."""
    cache_file = cache_root / f"{kepid}_kepler_long.npz"
    if not cache_file.exists():
        return None, None
    try:
        data = np.load(cache_file)
        lc = {k: data[k] for k in ("time", "flux", "flux_err", "quality")}
        tc, fc, fec, _ = clean(lc["time"], lc["flux"], lc["flux_err"], lc["quality"])
        df, _, de = detrend(tc, fc, fec)
        return odd_even_fold_data(tc, df, de, period, epoch, n_bins=n_bins)
    except Exception as exc:
        log.warning("odd/even fold failed for KIC %d: %s", kepid, exc)
        return None, None


# ── /model ────────────────────────────────────────────────────────────────────

@router.get("/model", response_model=ModelCardResponse)
def model_card(db: Session = Depends(get_db)):
    """Return the latest training run metrics + MODEL_CARD.md text."""
    tr = (
        db.query(TrainingRun)
        .order_by(TrainingRun.trained_at.desc())
        .first()
    )
    mc_text = _MODEL_CARD.read_text() if _MODEL_CARD.exists() else None

    if tr is None:
        return ModelCardResponse(
            trained_at=None,
            data_snapshot=None,
            honest_features=None,
            excluded_features=None,
            metrics=None,
            model_path=None,
            model_card_text=mc_text,
        )

    return ModelCardResponse(
        trained_at=tr.trained_at,
        data_snapshot=tr.data_snapshot,
        honest_features=(
            list(tr.honest_features.keys()) if isinstance(tr.honest_features, dict) else tr.honest_features
        ),
        excluded_features=(
            list(tr.excluded_features.keys()) if isinstance(tr.excluded_features, dict) else tr.excluded_features
        ),
        metrics=tr.metrics,
        model_path=tr.model_path,
        model_card_text=mc_text,
    )


# ── /explorer/{kepoi_name} ────────────────────────────────────────────────────

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
