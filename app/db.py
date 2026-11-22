from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,          # re-verify connection before use (handles Neon idle timeouts)
    connect_args={"connect_timeout": 10},
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> str:
    """Return 'connected' or an error message string."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "connected"
    except Exception as exc:
        return f"error: {exc}"
