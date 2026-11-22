"""
Hunt batch — §6.2.

Runs BLS on confirmed Kepler hosts, checks period recovery, scores with the
catalogue-trained honest-features classifier, persists Target + Detection rows.

Resumable: targets already marked "completed" are skipped on re-run unless
force=True.  A new HuntRun row is created each invocation.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.catalogue_sync import HONEST_FEATURE_COLUMNS
from app.config import Settings
from app.mast_client import fetch_light_curve
from app.models import Detection, HuntRun, KoiCatalogue, Target
from app.engine.pipeline import hunt_pipeline

log = logging.getLogger(__name__)

_MODEL_DIR = Path("models")


# ── Target selection ──────────────────────────────────────────────────────────

def _select_targets(db: Session, max_n: int) -> list[KoiCatalogue]:
    """
    Select one KOI per kepid (highest model_snr) from confirmed hosts.
    Returns at most max_n rows, ordered by model_snr descending so the
    strongest/most detectable transits are processed first.
    """
    # Subquery: best model_snr per confirmed kepid
    subq = (
        db.query(
            KoiCatalogue.kepid,
            func.max(KoiCatalogue.model_snr).label("max_snr"),
        )
        .filter(
            KoiCatalogue.disposition == "CONFIRMED",
            KoiCatalogue.kepid.isnot(None),
            KoiCatalogue.period.isnot(None),
        )
        .group_by(KoiCatalogue.kepid)
        .subquery()
    )

    return (
        db.query(KoiCatalogue)
        .join(
            subq,
            (KoiCatalogue.kepid == subq.c.kepid)
            & (KoiCatalogue.model_snr == subq.c.max_snr),
        )
        .filter(KoiCatalogue.disposition == "CONFIRMED")
        .order_by(KoiCatalogue.model_snr.desc().nullslast())
        .limit(max_n)
        .all()
    )


# ── Target upsert ─────────────────────────────────────────────────────────────

def _get_or_create_target(db: Session, koi: KoiCatalogue) -> Target:
    target = (
        db.query(Target)
        .filter_by(mission="Kepler", kepid=koi.kepid)
        .first()
    )
    if target is None:
        target = Target(
            mission="Kepler",
            kepid=koi.kepid,
            label_disposition=koi.disposition,
            hunt_status="pending",
        )
        db.add(target)
        db.flush()
    return target


# ── Period recovery check ─────────────────────────────────────────────────────

def period_recovered(
    bls_period: float,
    archive_period: float,
    tolerance: float = 0.01,
) -> tuple[bool, Optional[str]]:
    """
    Return (recovered, alias_type).

    Checks the period itself and common BLS aliases (half and double).
    alias_type is 'exact', 'half', or 'double', or None if not recovered.
    """
    # Check the period itself plus common BLS harmonics: ×2, ×3, ×4 (aliases
    # where only every Nth transit falls in the BLS window) and their sub-aliases.
    for alias, name in [
        (1.0, "exact"),
        (0.5, "half"),    (2.0, "double"),
        (3.0, "triple"),  (4.0, "quadruple"),
        (0.333, "third"), (0.25, "quarter"),
    ]:
        ref = archive_period * alias
        if abs(bls_period - ref) / ref < tolerance:
            return True, name
    return False, None


# ── Catalogue classifier scoring ──────────────────────────────────────────────

def _score_koi(model, koi: KoiCatalogue) -> Optional[float]:
    """
    Apply the trained honest-features classifier to the KOI catalogue row.
    Returns probability of being planet-like (class 1), or None on failure.
    """
    if koi.features is None:
        return None
    try:
        df = pd.DataFrame([koi.features])
        for col in HONEST_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        X = df[HONEST_FEATURE_COLUMNS]
        return float(model.predict_proba(X)[0, 1])
    except Exception as e:
        log.warning("Classifier scoring failed for %s: %s", koi.kepoi_name, e)
        return None


# ── Main batch function ───────────────────────────────────────────────────────

def run_hunt_batch(
    db: Session,
    settings: Settings,
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Resumable BLS hunt on confirmed Kepler hosts.

    Parameters
    ----------
    db       : SQLAlchemy session
    settings : app.config.Settings
    force    : re-process targets even if already completed
    cache_dir: override LC cache directory (useful in tests)

    Returns
    -------
    dict:
        hunt_run_id  : int
        results      : list of per-target result dicts
        n_error      : int
    """
    kois = _select_targets(db, max_n=settings.hunt_max_targets)
    if not kois:
        log.warning("No confirmed KOIs found in catalogue — has sync run?")
        return {"hunt_run_id": None, "results": [], "n_error": 0}

    log.info("Hunt: %d confirmed target(s) selected", len(kois))

    # Create HuntRun record
    hunt_run = HuntRun(
        started_at=datetime.now(timezone.utc),
        n_targets=len(kois),
        n_completed=0,
        n_detections=0,
        status="running",
        config={
            "hunt_max_targets": settings.hunt_max_targets,
            "hunt_n_periods": settings.hunt_n_periods,
            "hunt_period_max_days": settings.hunt_period_max_days,
            "bls_period_min_days": settings.bls_period_min_days,
            "hunt_recovery_tolerance": settings.hunt_recovery_tolerance,
        },
    )
    db.add(hunt_run)
    db.commit()

    # Load classifier
    model_path = _MODEL_DIR / "honest_model.joblib"
    model = joblib.load(model_path) if model_path.exists() else None
    if model is None:
        log.warning("honest_model.joblib not found — catalogue scores will be None")

    results: list[dict] = []
    n_error = 0
    n_completed = 0

    for koi in kois:
        target = _get_or_create_target(db, koi)

        if target.hunt_status == "completed" and not force:
            log.info("Skip %s (KIC %d) — already completed", koi.kepoi_name, koi.kepid)
            # Include in results for the recovery table
            existing = (
                db.query(Detection)
                .filter_by(target_id=target.target_id)
                .order_by(Detection.id.desc())
                .first()
            )
            if existing:
                rec, alias = period_recovered(
                    existing.period, koi.period, settings.hunt_recovery_tolerance
                )
                results.append({
                    "kepoi_name": koi.kepoi_name,
                    "kepid": koi.kepid,
                    "disposition": koi.disposition,
                    "archive_period": koi.period,
                    "bls_period": existing.period,
                    "bls_snr": existing.bls_snr,
                    "recovered": rec,
                    "alias": alias,
                    "score": existing.catalogue_model_score,
                    "skipped": True,
                })
            continue

        # Mark in-progress
        target.hunt_status = "in_progress"
        db.commit()

        # Fetch LC
        try:
            lc = fetch_light_curve(
                koi.kepid,
                cache_dir=cache_dir,
                force_refresh=force,
            )
        except Exception as exc:
            log.error("LC fetch failed for KIC %d (%s): %s", koi.kepid, koi.kepoi_name, exc)
            target.hunt_status = "error"
            db.commit()
            n_error += 1
            results.append({
                "kepoi_name": koi.kepoi_name,
                "kepid": koi.kepid,
                "disposition": koi.disposition,
                "archive_period": koi.period,
                "bls_period": None,
                "bls_snr": None,
                "recovered": False,
                "alias": None,
                "score": None,
                "error": str(exc),
            })
            continue

        # Run BLS pipeline
        try:
            pipe = hunt_pipeline(
                lc["time"],
                lc["flux"],
                lc["flux_err"],
                lc["quality"],
                period_min=settings.bls_period_min_days,
                period_max=settings.hunt_period_max_days,
                n_periods=settings.hunt_n_periods,
                window_length=settings.detrend_window_factor,
            )
        except Exception as exc:
            log.error("Pipeline failed for KIC %d (%s): %s", koi.kepid, koi.kepoi_name, exc)
            target.hunt_status = "error"
            db.commit()
            n_error += 1
            results.append({
                "kepoi_name": koi.kepoi_name,
                "kepid": koi.kepid,
                "disposition": koi.disposition,
                "archive_period": koi.period,
                "bls_period": None,
                "bls_snr": None,
                "recovered": False,
                "alias": None,
                "score": None,
                "error": str(exc),
            })
            continue

        # Score with catalogue classifier
        score = _score_koi(model, koi) if model is not None else None

        # Persist Detection
        bls = pipe["bls"]
        vetting = pipe["vetting"]
        fit = pipe["fit"]
        n_tr = vetting.get("n_transits")

        detection = Detection(
            target_id=target.target_id,
            run_id=hunt_run.id,
            period=bls.get("period"),
            epoch_bjd=bls.get("epoch"),
            duration_h=(bls["duration_days"] * 24) if bls.get("duration_days") is not None else None,
            depth_ppm=bls.get("depth_ppm"),
            bls_power=bls.get("power"),
            bls_snr=bls.get("snr"),
            fit_depth_ppm=fit.get("fit_depth_ppm"),
            fit_depth_sigma=fit.get("fit_depth_sigma"),
            odd_even_sigma=vetting.get("odd_even_sigma"),
            secondary_depth_ppm=vetting.get("secondary_depth_ppm"),
            secondary_sigma=vetting.get("secondary_sigma"),
            shape_ratio=vetting.get("shape_ratio"),
            n_transits=int(n_tr) if n_tr is not None else None,
            depth_scatter=vetting.get("depth_scatter"),
            vetting=vetting,
            fold_data=pipe["fold_data"],
            catalogue_model_score=score,
        )
        db.add(detection)

        target.hunt_status = "completed"
        n_completed += 1
        db.commit()

        # Period recovery check
        rec, alias = period_recovered(
            bls["period"], koi.period, settings.hunt_recovery_tolerance
        )
        log.info(
            "%s KIC %d  arch=%.4fd  bls=%.4fd  snr=%.1f  recovered=%s(%s)  score=%s",
            koi.kepoi_name, koi.kepid,
            koi.period, bls["period"], bls.get("snr", 0.0),
            rec, alias,
            f"{score:.3f}" if score is not None else "—",
        )

        results.append({
            "kepoi_name": koi.kepoi_name,
            "kepid": koi.kepid,
            "disposition": koi.disposition,
            "archive_period": koi.period,
            "bls_period": bls["period"],
            "bls_snr": bls.get("snr"),
            "recovered": rec,
            "alias": alias,
            "score": score,
        })

    # Finalise HuntRun
    n_detections = len([r for r in results if r.get("bls_period") is not None])
    hunt_run.finished_at = datetime.now(timezone.utc)
    hunt_run.n_completed = n_completed
    hunt_run.n_detections = n_detections
    hunt_run.status = "completed"
    db.commit()

    log.info(
        "Hunt complete: %d processed, %d detections, %d errors",
        len(results), n_detections, n_error,
    )
    return {"hunt_run_id": hunt_run.id, "results": results, "n_error": n_error}
