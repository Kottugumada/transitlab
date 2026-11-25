from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str
    last_sync_at: datetime | None
    last_hunt_at: datetime | None
    model_version: str | None


class FoldData(BaseModel):
    phase: list[float]
    flux: list[float | None]
    flux_err: list[float | None]


class ExplorerResponse(BaseModel):
    kepoi_name: str
    kepid: int
    kepler_name: str | None
    disposition: str | None
    period: float
    epoch: float
    duration_days: float
    depth_ppm: float | None
    n_cadences_raw: int
    n_cadences_clean: int
    fold_data: FoldData
    fit: dict
    vetting: dict


class CandidateItem(BaseModel):
    id: int
    target_id: int
    kepid: int | None
    kepoi_name: str | None
    disposition: str | None
    mission: str
    period: float | None
    depth_ppm: float | None
    bls_snr: float | None
    n_transits: int | None
    catalogue_model_score: float | None
    created_at: datetime


class CandidatesResponse(BaseModel):
    items: list[CandidateItem]
    total: int


class CandidateDetail(BaseModel):
    id: int
    target_id: int
    kepid: int | None
    kepoi_name: str | None
    kepler_name: str | None
    disposition: str | None
    mission: str
    # BLS / fit columns
    period: float | None
    epoch_bjd: float | None
    duration_h: float | None
    depth_ppm: float | None
    bls_snr: float | None
    bls_power: float | None
    fit_depth_ppm: float | None
    fit_depth_sigma: float | None
    odd_even_sigma: float | None
    secondary_depth_ppm: float | None
    secondary_sigma: float | None
    shape_ratio: float | None
    n_transits: int | None
    depth_scatter: float | None
    catalogue_model_score: float | None
    # JSON blobs
    fold_data: FoldData | None
    odd_fold: FoldData | None      # None when cached LC unavailable
    even_fold: FoldData | None
    vetting: dict | None
    fit: dict | None
    created_at: datetime


class ModelCardResponse(BaseModel):
    trained_at: datetime | None
    data_snapshot: str | None
    honest_features: list[str] | None
    excluded_features: list[str] | None
    metrics: dict | None
    model_path: str | None
    model_card_text: str | None
