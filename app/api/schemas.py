from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str
    last_sync_at: datetime | None
    last_hunt_at: datetime | None
    model_version: str | None
