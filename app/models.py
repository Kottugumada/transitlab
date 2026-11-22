"""SQLAlchemy ORM models — mirrors the §7 data model exactly."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class KoiCatalogue(Base):
    __tablename__ = "koi_catalogue"

    kepoi_name: Mapped[str] = mapped_column(String(32), primary_key=True)
    kepid: Mapped[int | None] = mapped_column(BigInteger)
    kepler_name: Mapped[str | None] = mapped_column(String(64))
    disposition: Mapped[str | None] = mapped_column(String(32))   # CONFIRMED / CANDIDATE / FALSE POSITIVE
    pdisposition: Mapped[str | None] = mapped_column(String(32))
    koi_score: Mapped[float | None] = mapped_column(Double)
    period: Mapped[float | None] = mapped_column(Double)          # days
    time0: Mapped[float | None] = mapped_column(Double)           # BKJD
    duration: Mapped[float | None] = mapped_column(Double)        # hours
    depth: Mapped[float | None] = mapped_column(Double)           # ppm
    ror: Mapped[float | None] = mapped_column(Double)             # Rp/R*
    impact: Mapped[float | None] = mapped_column(Double)
    model_snr: Mapped[float | None] = mapped_column(Double)
    odd_even_sig: Mapped[float | None] = mapped_column(Double)
    steff: Mapped[float | None] = mapped_column(Double)           # K
    srad: Mapped[float | None] = mapped_column(Double)            # R_sun
    kepmag: Mapped[float | None] = mapped_column(Double)
    features: Mapped[dict | None] = mapped_column(JSONB)          # full feature set for classifier
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    targets: Mapped[list["Target"]] = relationship(
        "Target", foreign_keys="Target.kepid", primaryjoin="KoiCatalogue.kepid == Target.kepid",
        back_populates="koi_entry", viewonly=True,
    )


class PsPlanet(Base):
    __tablename__ = "ps_planets"

    pl_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    hostname: Mapped[str | None] = mapped_column(String(128))
    period: Mapped[float | None] = mapped_column(Double)          # days
    tranmid: Mapped[float | None] = mapped_column(Double)         # BJD
    trandur: Mapped[float | None] = mapped_column(Double)         # hours
    trandep: Mapped[float | None] = mapped_column(Double)         # ppm
    rade: Mapped[float | None] = mapped_column(Double)            # R_earth
    sy_vmag: Mapped[float | None] = mapped_column(Double)
    ra: Mapped[float | None] = mapped_column(Double)              # deg
    dec: Mapped[float | None] = mapped_column(Double)             # deg
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    backyard_predictions: Mapped[list["BackyardPrediction"]] = relationship(
        "BackyardPrediction", back_populates="planet"
    )


class Target(Base):
    __tablename__ = "targets"

    target_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mission: Mapped[str] = mapped_column(String(16))              # 'Kepler' or 'TESS'
    kepid: Mapped[int | None] = mapped_column(BigInteger)
    tic_id: Mapped[int | None] = mapped_column(BigInteger)
    label_disposition: Mapped[str | None] = mapped_column(String(32))  # known label if any
    hunt_status: Mapped[str] = mapped_column(String(32), default="pending")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    koi_entry: Mapped["KoiCatalogue | None"] = relationship(
        "KoiCatalogue", foreign_keys=[kepid],
        primaryjoin="Target.kepid == KoiCatalogue.kepid",
        back_populates="targets", viewonly=True,
    )
    detections: Mapped[list["Detection"]] = relationship("Detection", back_populates="target")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(Integer, ForeignKey("targets.target_id"))
    run_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("hunt_runs.id"))
    period: Mapped[float | None] = mapped_column(Double)          # days
    epoch_bjd: Mapped[float | None] = mapped_column(Double)
    duration_h: Mapped[float | None] = mapped_column(Double)      # hours
    depth_ppm: Mapped[float | None] = mapped_column(Double)
    bls_power: Mapped[float | None] = mapped_column(Double)
    bls_snr: Mapped[float | None] = mapped_column(Double)
    fit_depth_ppm: Mapped[float | None] = mapped_column(Double)
    fit_depth_sigma: Mapped[float | None] = mapped_column(Double)
    odd_even_sigma: Mapped[float | None] = mapped_column(Double)
    secondary_depth_ppm: Mapped[float | None] = mapped_column(Double)
    secondary_sigma: Mapped[float | None] = mapped_column(Double)
    shape_ratio: Mapped[float | None] = mapped_column(Double)     # ingress+egress / total duration
    n_transits: Mapped[int | None] = mapped_column(Integer)
    depth_scatter: Mapped[float | None] = mapped_column(Double)
    vetting: Mapped[dict | None] = mapped_column(JSONB)           # full vetting feature dict
    fold_data: Mapped[dict | None] = mapped_column(JSONB)         # binned folded curve for UI
    catalogue_model_score: Mapped[float | None] = mapped_column(Double)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    target: Mapped["Target"] = relationship("Target", back_populates="detections")
    hunt_run: Mapped["HuntRun | None"] = relationship("HuntRun", back_populates="detections")


class HuntRun(Base):
    __tablename__ = "hunt_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    n_targets: Mapped[int | None] = mapped_column(Integer)
    n_completed: Mapped[int | None] = mapped_column(Integer)
    n_detections: Mapped[int | None] = mapped_column(Integer)
    config: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32), default="pending")

    detections: Mapped[list["Detection"]] = relationship("Detection", back_populates="hunt_run")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    data_snapshot: Mapped[str | None] = mapped_column(Text)       # e.g. '2026-11-21'
    split_seed: Mapped[int | None] = mapped_column(Integer)
    honest_features: Mapped[dict | None] = mapped_column(JSONB)
    excluded_features: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    model_path: Mapped[str | None] = mapped_column(Text)


class BackyardPrediction(Base):
    __tablename__ = "backyard_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date_local: Mapped[str] = mapped_column(String(10))           # YYYY-MM-DD
    pl_name: Mapped[str] = mapped_column(String(128), ForeignKey("ps_planets.pl_name"))
    hostname: Mapped[str | None] = mapped_column(String(128))
    ingress: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    mid_transit: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    egress: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    depth_pct: Mapped[float | None] = mapped_column(Double)
    sy_vmag: Mapped[float | None] = mapped_column(Double)
    observable_rank: Mapped[int | None] = mapped_column(Integer)
    reasons: Mapped[dict | None] = mapped_column(JSONB)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    planet: Mapped["PsPlanet"] = relationship("PsPlanet", back_populates="backyard_predictions")
