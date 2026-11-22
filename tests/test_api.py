"""
Step 1 done-check tests.

These tests require a live DATABASE_URL in the environment (Neon or local Postgres).
They must pass before Step 1 is considered complete.
They never mock the database — the point is to verify the real connection and schema.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db import check_db_connection

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_db_connected():
    response = client.get("/api/v1/health")
    data = response.json()
    assert data["db"] == "connected", (
        f"Database not reachable. db field: {data['db']!r}\n"
        "Make sure DATABASE_URL is set in .env and the Neon database is accessible."
    )


def test_health_status_ok_when_db_connected():
    response = client.get("/api/v1/health")
    data = response.json()
    if data["db"] == "connected":
        assert data["status"] == "ok"


def test_health_response_shape():
    response = client.get("/api/v1/health")
    data = response.json()
    assert set(data.keys()) == {"status", "db", "last_sync_at", "last_hunt_at", "model_version"}


def test_check_db_connection_function():
    result = check_db_connection()
    assert result == "connected", f"check_db_connection() returned: {result!r}"


def test_all_tables_exist():
    """Verify Alembic migration created all §7 tables."""
    from sqlalchemy import inspect
    from app.db import engine

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    required = {
        "koi_catalogue",
        "ps_planets",
        "targets",
        "detections",
        "hunt_runs",
        "training_runs",
        "backyard_predictions",
    }
    missing = required - existing
    assert not missing, f"Missing tables after migration: {missing}"
