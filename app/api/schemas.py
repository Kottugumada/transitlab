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
