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
    # Lower bound: 0.75 d avoids the P/6 sub-harmonic floor trap seen when
    # period_min=0.5 divides evenly into common short periods (1.5, 3.0, 4.5 d).
    # Upper bound: 1/3 of baseline avoids spurious long-period fits with few transits
    bls_period_min_days: float = 0.75
    # bls_period_max_days is computed at runtime as baseline / 3

    # Detrending window (multiples of transit duration)
    # Must be >> transit duration to avoid eating the signal;
    # too small → transit suppressed, too large → long-period systematics leak through.
    # 101-cadence (≈2 d for Kepler long cadence) is the Lightkurve default and
    # a reasonable starting point; override per-target when transit duration is known.
    detrend_window_factor: int = 101  # Savitzky-Golay window (must be odd)

    # Hunt BLS grid parameters (§6.2)
    # 2000 frequencies spans 0.5–50 d at Kepler-baseline resolution (1/1461 d⁻¹);
    # too few → peaks aliased, too many → 10+ min per target.
    hunt_n_periods: int = 2000
    # Cap at half the Kepler baseline; periods above this have < 3 transits.
    hunt_period_max_days: float = 50.0
    # Period recovery tolerance: BLS period is "recovered" if within this fraction
    # of the archive period (or its half / double alias).
    hunt_recovery_tolerance: float = 0.01     # 1% — same as injection_period_tolerance

    # Synthetic injection recovery thresholds (§6.1)
    injection_period_tolerance: float = 0.01   # 1% period recovery
    injection_depth_tolerance: float = 0.25    # 25% depth recovery
    injection_min_snr: float = 3.0             # inject at ≥3× noise floor


settings = Settings()
