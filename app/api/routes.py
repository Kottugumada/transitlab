from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import check_db_connection, get_db
from app.api.schemas import HealthResponse

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
