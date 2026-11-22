from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str

    # Backyard prediction location (default: Morgan Hill, CA)
    location_lat: float = 37.1305
    location_lon: float = -121.6544
    location_name: str = "Morgan Hill, CA"

    # Hunt batch budget — max targets per run; keep low in dev, raise for production
    hunt_max_targets: int = 50

    # BLS period search grid (days)
    # Lower bound: 0.5 d catches ultra-short-period planets
    # Upper bound: 1/3 of baseline avoids spurious long-period fits with few transits
    bls_period_min_days: float = 0.5
    # bls_period_max_days is computed at runtime as baseline / 3

    # Detrending window (multiples of transit duration)
    # Must be >> transit duration to avoid eating the signal;
    # too small → transit suppressed, too large → long-period systematics leak through.
    # 101-cadence (≈2 d for Kepler long cadence) is the Lightkurve default and
    # a reasonable starting point; override per-target when transit duration is known.
    detrend_window_factor: int = 101  # Savitzky-Golay window (must be odd)

    # Synthetic injection recovery thresholds (§6.1)
    injection_period_tolerance: float = 0.01   # 1% period recovery
    injection_depth_tolerance: float = 0.25    # 25% depth recovery
    injection_min_snr: float = 3.0             # inject at ≥3× noise floor


settings = Settings()
