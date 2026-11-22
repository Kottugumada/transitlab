"""
Generate synthetic light curve fixtures for offline engine tests.

Run once:
    python tests/fixtures/generate_synthetic.py

Produces:
    tests/fixtures/kic_quiet_1.csv   — 1500 cadences, no transit, 200 ppm noise
    tests/fixtures/kic_quiet_2.csv   — 1500 cadences, no transit, 200 ppm noise (different seed)
    tests/fixtures/kic_planet_5d.csv — 1500 cadences, planet injection P=5d depth=1500ppm dur=2h
    tests/fixtures/kic_eb_10d.csv    — 1500 cadences, EB injection P=10d primary=10000ppm secondary=5000ppm

Columns: time_bkjd, flux, flux_err, quality
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent

# Kepler long-cadence cadence = 29.4 minutes
CADENCE_DAYS = 29.4 / 60 / 24
N_CADENCES = 1500
T0_BKJD = 120.0        # arbitrary BKJD start
NOISE_PPM = 200.0      # per-cadence noise level in ppm


def _make_time() -> np.ndarray:
    return T0_BKJD + np.arange(N_CADENCES) * CADENCE_DAYS


def _add_noise(rng: np.random.Generator, n: int = N_CADENCES) -> tuple[np.ndarray, np.ndarray]:
    flux = 1.0 + rng.normal(0, NOISE_PPM * 1e-6, size=n)
    flux_err = np.full(n, NOISE_PPM * 1e-6)
    return flux, flux_err


def _trapezoid_model(time, period, epoch, depth, duration_days, ingress_frac=0.1):
    """Box-with-ramp trapezoid transit model."""
    flux = np.ones(len(time))
    half_dur = duration_days / 2.0
    ingress = ingress_frac * duration_days / 2.0

    n_min = int(np.floor((time[0] - epoch) / period))
    n_max = int(np.ceil((time[-1] - epoch) / period))

    for n in range(n_min, n_max + 1):
        t_mid = epoch + n * period
        t = np.abs(time - t_mid)
        in_flat = t < (half_dur - ingress)
        in_ramp = (t >= (half_dur - ingress)) & (t < half_dur)

        flux[in_flat] -= depth
        if ingress > 0:
            ramp_frac = (t[in_ramp] - (half_dur - ingress)) / ingress
            flux[in_ramp] -= depth * (1.0 - ramp_frac)

    return flux


def make_quiet_1():
    rng = np.random.default_rng(42)
    time = _make_time()
    flux, flux_err = _add_noise(rng)
    quality = np.zeros(N_CADENCES, dtype=int)
    # Inject a few bad cadences
    bad = rng.integers(0, N_CADENCES, size=10)
    quality[bad] = 16  # bit 4 set
    flux[bad] = np.nan
    df = pd.DataFrame({"time_bkjd": time, "flux": flux, "flux_err": flux_err, "quality": quality})
    df.to_csv(OUT / "kic_quiet_1.csv", index=False)
    print(f"Wrote kic_quiet_1.csv  ({len(df)} cadences, 10 bad)")


def make_quiet_2():
    rng = np.random.default_rng(99)
    time = _make_time()
    flux, flux_err = _add_noise(rng)
    quality = np.zeros(N_CADENCES, dtype=int)
    df = pd.DataFrame({"time_bkjd": time, "flux": flux, "flux_err": flux_err, "quality": quality})
    df.to_csv(OUT / "kic_quiet_2.csv", index=False)
    print(f"Wrote kic_quiet_2.csv  ({len(df)} cadences)")


def make_planet_5d():
    """P=5d, depth=1500ppm, duration=2h → SNR ≈ 37 above 200ppm noise."""
    rng = np.random.default_rng(7)
    time = _make_time()
    flux_noise, flux_err = _add_noise(rng)

    period = 5.0           # days
    epoch = T0_BKJD + 1.3  # arbitrary epoch within the first period
    depth = 1500e-6        # 1500 ppm
    duration_days = 2.0 / 24.0  # 2 hours in days
    ingress_frac = 0.1

    transit = _trapezoid_model(time, period, epoch, depth, duration_days, ingress_frac)
    flux = flux_noise * transit  # multiply so normalisation is preserved

    quality = np.zeros(N_CADENCES, dtype=int)
    df = pd.DataFrame({"time_bkjd": time, "flux": flux, "flux_err": flux_err, "quality": quality})
    df.to_csv(OUT / "kic_planet_5d.csv", index=False)

    n_transits = int((time[-1] - time[0]) / period)
    pts_per_transit = int(duration_days / CADENCE_DAYS)
    snr_approx = depth / (200e-6 / np.sqrt(pts_per_transit * n_transits))
    print(f"Wrote kic_planet_5d.csv ({len(df)} cadences, P=5d, depth=1500ppm, "
          f"SNR≈{snr_approx:.0f}, n_transits≈{n_transits})")


def make_eb_10d():
    """
    EB: P=10d, primary=10000ppm, secondary=5000ppm at phase 0.5.
    Large odd/even depth difference should trigger vetting flag.
    """
    rng = np.random.default_rng(13)
    time = _make_time()
    flux_noise, flux_err = _add_noise(rng)

    period = 10.0
    epoch = T0_BKJD + 2.5
    primary_depth = 10000e-6
    secondary_depth = 5000e-6
    duration_days = 3.0 / 24.0  # 3 hours
    ingress_frac = 0.8  # V-shape EB

    # Primary transits (even-numbered)
    flux = _trapezoid_model(time, period, epoch, primary_depth, duration_days, ingress_frac)
    # Secondary transits at epoch + period/2 (odd-numbered relative to primary)
    flux_secondary = _trapezoid_model(time, period, epoch + period / 2.0, secondary_depth, duration_days, ingress_frac)
    # Combine: both dim the star
    flux = flux * flux_secondary
    flux = flux * flux_noise

    quality = np.zeros(N_CADENCES, dtype=int)
    df = pd.DataFrame({"time_bkjd": time, "flux": flux, "flux_err": flux_err, "quality": quality})
    df.to_csv(OUT / "kic_eb_10d.csv", index=False)
    print(f"Wrote kic_eb_10d.csv   ({len(df)} cadences, P=10d, pri=10000ppm, sec=5000ppm)")


if __name__ == "__main__":
    make_quiet_1()
    make_quiet_2()
    make_planet_5d()
    make_eb_10d()
    print("Done.")
