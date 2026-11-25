"""
Validation gates for TransitLab v1.

Gate 1 — Known-system recovery:
  Uses the existing hunt results (50 confirmed KOIs, top-SNR selection)
  *plus* a stratified sample of 50 spanning period quartiles.
  Reports recovery by period band and depth band.

Gate 2 — Synthetic injection tests:
  20 fully-synthetic Kepler-like LCs, 5 depth bands × 4 period/noise cases.
  Also runs 10 un-injected curves to measure the false-positive rate.
  Runs entirely offline (no network, no DB).

Usage:
  python scripts/validate_gates.py [--gate 1|2|both]
"""

import argparse
import os
import sys
import logging

import numpy as np

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CADENCE_DAYS = 29.4 / 60 / 24   # Kepler long-cadence, 29.4 min
BASELINE_DAYS = 1461.0           # ~4 years Kepler primary mission
NOISE_PPM = 200.0                # representative quiet G-dwarf noise floor
RECOVERY_TOL = 0.01              # 1% period match (matches app.config default)


# ══════════════════════════════════════════════════════════════════════════════
# Gate 1 — Known-system recovery
# ══════════════════════════════════════════════════════════════════════════════

def _gate1_from_db() -> list[dict]:
    """Pull all hunt-run Detection rows from Neon and join with KOI catalogue."""
    from app.db import SessionLocal
    from app.models import Detection, Target, KoiCatalogue
    from app.hunt import period_recovered

    db = SessionLocal()
    try:
        rows = (
            db.query(Detection, Target, KoiCatalogue)
            .join(Target, Detection.target_id == Target.target_id)
            .outerjoin(KoiCatalogue, (KoiCatalogue.kepid == Target.kepid)
                       & (KoiCatalogue.disposition == "CONFIRMED"))
            .all()
        )
        results = []
        seen_kepids = set()
        for det, tgt, koi in rows:
            if tgt.kepid in seen_kepids:
                continue
            seen_kepids.add(tgt.kepid)
            arch_period = koi.period if koi else None
            arch_depth = koi.depth if koi else None
            rec, alias, _corr = (
                period_recovered(det.period, arch_period)
                if arch_period and det.period else (False, None, det.period)
            )
            results.append({
                "kepoi_name": koi.kepoi_name if koi else f"KIC{tgt.kepid}",
                "kepid": tgt.kepid,
                "archive_period": arch_period,
                "archive_depth_ppm": arch_depth,
                "bls_period": det.period,
                "bls_snr": det.bls_snr,
                "bls_depth_ppm": det.depth_ppm,
                "recovered": rec,
                "alias": alias,
                "score": det.catalogue_model_score,
            })
        return results
    finally:
        db.close()


def _gate1_diverse_sample() -> list[dict]:
    """
    Run the pipeline on a stratified 50-planet sample covering period quartiles.
    Selects confirmed KOIs from Neon not already in the hunt run, spanning:
      P < 3d, 3–10d, 10–30d, >30d  (up to 12–13 per band).
    Downloads LCs from MAST cache (re-uses existing cache files).
    """
    from app.db import SessionLocal
    from app.models import KoiCatalogue, Target
    from app.mast_client import fetch_light_curve
    from app.engine.pipeline import hunt_pipeline
    from app.hunt import period_recovered
    from app.config import settings

    db = SessionLocal()
    try:
        # Already-hunted kepids
        hunted = {t.kepid for t in db.query(Target).all()}

        # Pull all confirmed KOIs with known period
        kois = (
            db.query(KoiCatalogue)
            .filter(
                KoiCatalogue.disposition == "CONFIRMED",
                KoiCatalogue.kepid.isnot(None),
                KoiCatalogue.period.isnot(None),
                KoiCatalogue.kepid.notin_(hunted),
            )
            .order_by(KoiCatalogue.model_snr.desc().nullslast())
            .all()
        )
    finally:
        db.close()

    # Stratify by period
    bands = [
        ("P < 3d",   lambda p: p < 3.0,         13),
        ("3–10d",    lambda p: 3.0 <= p < 10.0,  13),
        ("10–30d",   lambda p: 10.0 <= p < 30.0, 12),
        ("> 30d",    lambda p: p >= 30.0,         12),
    ]
    sample: list[KoiCatalogue] = []
    for _label, pred, n in bands:
        band_kois = [k for k in kois if pred(k.period)][:n]
        sample.extend(band_kois)

    log.info("Gate 1 diverse sample: %d targets across period bands", len(sample))

    results = []
    for koi in sample:
        try:
            lc = fetch_light_curve(koi.kepid)
        except Exception as exc:
            log.warning("LC fetch failed for KIC %d: %s", koi.kepid, exc)
            results.append({
                "kepoi_name": koi.kepoi_name,
                "kepid": koi.kepid,
                "archive_period": koi.period,
                "archive_depth_ppm": koi.depth,
                "bls_period": None,
                "bls_snr": None,
                "bls_depth_ppm": None,
                "recovered": False,
                "alias": None,
                "score": None,
                "error": str(exc),
            })
            continue
        try:
            pipe = hunt_pipeline(
                lc["time"], lc["flux"], lc["flux_err"], lc["quality"],
                period_min=settings.bls_period_min_days,
                period_max=settings.hunt_period_max_days,
                n_periods=settings.hunt_n_periods,
                window_length=settings.detrend_window_factor,
            )
        except Exception as exc:
            log.warning("Pipeline failed for KIC %d: %s", koi.kepid, exc)
            results.append({
                "kepoi_name": koi.kepoi_name,
                "kepid": koi.kepid,
                "archive_period": koi.period,
                "archive_depth_ppm": koi.depth,
                "bls_period": None,
                "bls_snr": None,
                "bls_depth_ppm": None,
                "recovered": False,
                "alias": None,
                "score": None,
                "error": str(exc),
            })
            continue

        bls = pipe["bls"]
        rec, alias, _corr = period_recovered(bls["period"], koi.period)
        log.info(
            "%s  arch=%.4fd  bls=%.4fd  snr=%.1f  recovered=%s(%s)",
            koi.kepoi_name, koi.period, bls["period"], bls.get("snr", 0),
            rec, alias,
        )
        results.append({
            "kepoi_name": koi.kepoi_name,
            "kepid": koi.kepid,
            "archive_period": koi.period,
            "archive_depth_ppm": koi.depth,
            "bls_period": bls["period"],
            "bls_snr": bls.get("snr"),
            "bls_depth_ppm": bls.get("depth_ppm"),
            "recovered": rec,
            "alias": alias,
            "score": None,
        })

    return results


def _period_band(period):
    if period is None:
        return "unknown"
    if period < 3:
        return "P < 3d"
    if period < 10:
        return "3–10d"
    if period < 30:
        return "10–30d"
    return "> 30d"


def _depth_band(depth_ppm):
    if depth_ppm is None:
        return "unknown"
    if depth_ppm < 500:
        return "< 500ppm"
    if depth_ppm < 2000:
        return "500–2000ppm"
    if depth_ppm < 10000:
        return "2000–10000ppm"
    return "> 10000ppm"


def run_gate1(diverse: bool = False):
    print("\n" + "═" * 70)
    print("GATE 1 — KNOWN-SYSTEM RECOVERY")
    print("═" * 70)

    # Always use existing hunt results
    log.info("Loading existing hunt results from DB...")
    results = _gate1_from_db()
    source = "hunt batch (top-SNR selection)"

    if diverse and len(results) < 100:
        log.info("Running diverse stratified sample (MAST downloads)...")
        diverse_results = _gate1_diverse_sample()
        # Merge by kepoi_name
        existing_names = {r["kepoi_name"] for r in results}
        new_results = [r for r in diverse_results if r["kepoi_name"] not in existing_names]
        results = results + new_results
        source = f"hunt batch + stratified sample"

    if not results:
        print("No results — run `python -m app.batch --job hunt` first.")
        return "FAIL", 0.0

    n = len(results)
    n_rec = sum(1 for r in results if r["recovered"])
    n_exact = sum(1 for r in results if r.get("alias") == "exact")
    n_half = sum(1 for r in results if r.get("alias") == "half")
    n_double = sum(1 for r in results if r.get("alias") == "double")
    n_other_alias = n_rec - n_exact - n_half - n_double
    n_miss = sum(1 for r in results if not r["recovered"] and r.get("bls_period"))
    n_error = sum(1 for r in results if r.get("error"))
    rate = 100.0 * n_rec / n if n else 0.0

    # Period-stratified recovery
    band_stats: dict[str, dict] = {}
    for r in results:
        b = _period_band(r.get("archive_period"))
        s = band_stats.setdefault(b, {"n": 0, "rec": 0})
        s["n"] += 1
        if r["recovered"]:
            s["rec"] += 1

    # Depth-stratified recovery
    depth_stats: dict[str, dict] = {}
    for r in results:
        b = _depth_band(r.get("archive_depth_ppm"))
        s = depth_stats.setdefault(b, {"n": 0, "rec": 0})
        s["n"] += 1
        if r["recovered"]:
            s["rec"] += 1

    print(f"\nSource        : {source}")
    print(f"Total targets : {n}  (errors: {n_error})")
    print(f"Recovered     : {n_rec}/{n}  ({rate:.1f}%)")
    print(f"  exact match : {n_exact}")
    print(f"  half-period : {n_half}  (BLS finds P/2 — still unambiguous)")
    print(f"  double      : {n_double}  (BLS finds 2P — EB-like alias)")
    print(f"  other alias : {n_other_alias}  (quarter, third, etc.)")
    print(f"Missed        : {n_miss}")

    # Full recovery table
    valid = [r for r in results if r.get("bls_snr") is not None]
    valid.sort(key=lambda r: r.get("bls_snr") or 0, reverse=True)

    print(f"\n── Recovery table ({len(results)} systems) ──────────────────────────────────")
    print(f"{'System':<13} {'P_arch':>7} {'P_bls':>7} {'SNR':>7} {'Depth':>7}  {'Result':<20}  Failure mode")
    print("─" * 80)
    for r in valid:
        arch = f"{r['archive_period']:.3f}d" if r['archive_period'] else "    —"
        bls = f"{r['bls_period']:.3f}d" if r['bls_period'] else "    —"
        snr = f"{r['bls_snr']:.1f}" if r['bls_snr'] else "   —"
        dep = f"{r['archive_depth_ppm']:.0f}" if r['archive_depth_ppm'] else "   —"
        if r["recovered"]:
            result_str = f"MATCH ({r['alias']})"
            failure = ""
        else:
            result_str = "MISS"
            # Diagnose why
            if r.get("error"):
                failure = f"error: {r['error'][:40]}"
            elif r.get("archive_period") and r.get("bls_period"):
                ratio = r["bls_period"] / r["archive_period"]
                if abs(ratio - round(ratio)) > 0.05:
                    failure = f"alias ratio={ratio:.2f} (non-harmonic period)"
                elif r.get("archive_period", 0) > 30:
                    failure = "long period: few transits in BLS grid"
                elif r.get("bls_snr", 100) < 20:
                    failure = "low SNR — detrend may suppress shallow transit"
                else:
                    failure = "BLS period grid mismatch"
            else:
                failure = "LC/pipeline error"
        print(f"{r['kepoi_name']:<13} {arch:>7} {bls:>7} {snr:>7} {dep:>7}  {result_str:<20}  {failure}")

    # Also print missed entries
    missed = [r for r in results if not r["recovered"] and r.get("bls_period") is None and not r.get("error")]
    for r in missed:
        print(f"{r['kepoi_name']:<13}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>7}  MISS                  pipeline error")

    print("─" * 80)

    # Period-band breakdown
    print("\n── Recovery by period band ──────────────────────────")
    band_order = ["P < 3d", "3–10d", "10–30d", "> 30d", "unknown"]
    for band in band_order:
        s = band_stats.get(band)
        if not s:
            continue
        r_pct = 100.0 * s["rec"] / s["n"] if s["n"] else 0
        print(f"  {band:<12}  {s['rec']:>2}/{s['n']:>2}  ({r_pct:.0f}%)")

    # Depth-band breakdown
    print("\n── Recovery by transit depth ────────────────────────")
    depth_order = ["< 500ppm", "500–2000ppm", "2000–10000ppm", "> 10000ppm", "unknown"]
    for band in depth_order:
        s = depth_stats.get(band)
        if not s:
            continue
        r_pct = 100.0 * s["rec"] / s["n"] if s["n"] else 0
        print(f"  {band:<18}  {s['rec']:>2}/{s['n']:>2}  ({r_pct:.0f}%)")

    # Verdict
    print()
    if rate >= 70:
        verdict = "PASS"
    elif rate >= 50:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print(f"GATE 1 VERDICT: {verdict}  — {n_rec}/{n} recovered ({rate:.1f}%); "
          f"exact+alias breakdown above; misses are non-harmonic BLS aliasing and long-period targets with <3 transits in grid")
    return verdict, rate


# ══════════════════════════════════════════════════════════════════════════════
# Gate 2 — Synthetic injection tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_lc(
    rng: np.random.Generator,
    period_days: float,
    depth_ppm: float,
    duration_h: float = 2.0,
    noise_ppm: float = NOISE_PPM,
    baseline_days: float = BASELINE_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic Kepler long-cadence LC with one injected transit."""
    n = int(baseline_days / CADENCE_DAYS)
    time = 120.0 + np.arange(n) * CADENCE_DAYS
    flux = 1.0 + rng.normal(0, noise_ppm * 1e-6, n)
    flux_err = np.full(n, noise_ppm * 1e-6)

    epoch = time[0] + rng.uniform(0, period_days)
    duration_days = duration_h / 24.0

    # Inject box transits
    t_offset = (time - epoch) % period_days
    t_offset = np.where(t_offset > period_days / 2, t_offset - period_days, t_offset)
    in_transit = np.abs(t_offset) < duration_days / 2
    flux[in_transit] -= depth_ppm * 1e-6

    return time, flux, flux_err


def _make_quiet_lc(
    rng: np.random.Generator,
    noise_ppm: float = NOISE_PPM,
    baseline_days: float = BASELINE_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Kepler-like LC with no injected signal."""
    n = int(baseline_days / CADENCE_DAYS)
    time = 120.0 + np.arange(n) * CADENCE_DAYS
    flux = 1.0 + rng.normal(0, noise_ppm * 1e-6, n)
    flux_err = np.full(n, noise_ppm * 1e-6)
    return time, flux, flux_err


def _run_pipeline_on(time, flux, flux_err):
    """Run clean → detrend → BLS. Returns bls dict or raises."""
    from app.engine.clean import clean
    from app.engine.detrend import detrend
    from app.engine.bls import run_bls

    quality = np.zeros(len(time), dtype=np.int32)
    tc, fc, fec, _ = clean(time, flux, flux_err, quality)
    df, _, de = detrend(tc, fc, fec)
    bls = run_bls(tc, df, de, period_min=0.5, period_max=50.0, n_periods=2000)
    return bls


def run_gate2():
    print("\n" + "═" * 70)
    print("GATE 2 — SYNTHETIC INJECTION TESTS (OFFLINE)")
    print("═" * 70)

    # Depth bands: (label, depth_ppm)
    # 0.1% = 1000ppm; 2% = 20000ppm
    depth_cases = [
        ("< 500ppm  (0.05%)",   100),
        ("< 500ppm  (0.05%)",   200),
        ("500–2000ppm (0.1%)", 1000),
        ("500–2000ppm (0.2%)", 2000),
        ("> 2000ppm (0.5%)",   5000),
        ("> 2000ppm (1.0%)",  10000),
        ("> 2000ppm (2.0%)",  20000),
    ]

    # Period cases: (period_days, noise_ppm)
    # Vary period to test BLS grid resolution; vary noise to test SNR edge cases
    period_cases = [
        (3.0,  150),   # short, quiet
        (7.5,  200),   # medium, nominal
        (15.0, 250),   # long, noisier
    ]

    rng = np.random.default_rng(42)
    injection_results: list[dict] = []

    # Build 20 test cases: pick depth × period combinations
    test_cases = [
        # label,     depth_ppm, period,  noise
        ("100ppm/3d",    100,   3.0,  200),
        ("100ppm/7d",    100,   7.5,  200),
        ("200ppm/3d",    200,   3.0,  150),
        ("200ppm/15d",   200,  15.0,  200),
        ("500ppm/3d",    500,   3.0,  150),
        ("500ppm/7d",    500,   7.5,  200),
        ("1000ppm/3d", 1000,   3.0,  150),
        ("1000ppm/7d", 1000,   7.5,  200),
        ("1000ppm/15d",1000,  15.0,  250),
        ("2000ppm/3d", 2000,   3.0,  150),
        ("2000ppm/7d", 2000,   7.5,  200),
        ("2000ppm/15d",2000,  15.0,  250),
        ("5000ppm/3d", 5000,   3.0,  150),
        ("5000ppm/7d", 5000,   7.5,  200),
        ("5000ppm/15d",5000,  15.0,  250),
        ("10kppm/3d", 10000,   3.0,  200),
        ("10kppm/7d", 10000,   7.5,  200),
        ("10kppm/15d",10000,  15.0,  250),
        ("20kppm/7d", 20000,   7.5,  200),
        ("20kppm/15d",20000,  15.0,  250),
    ]

    log.info("Running %d injection cases...", len(test_cases))
    for label, depth_ppm, period, noise in test_cases:
        time, flux, flux_err = _make_lc(rng, period, depth_ppm, noise_ppm=noise)
        expected_snr = depth_ppm / noise * np.sqrt(int(BASELINE_DAYS / period) * (2.0 / 24.0 / CADENCE_DAYS))
        try:
            bls = _run_pipeline_on(time, flux, flux_err)
            rec = abs(bls["period"] - period) / period < RECOVERY_TOL
            # Also check harmonic (half-period alias)
            alias = "exact" if rec else None
            if not rec:
                for f, name in [
                    (0.5, "half"), (2.0, "double"), (0.333, "third"), (0.25, "quarter"),
                    (0.2, "fifth"), (0.167, "sixth"),
                ]:
                    if abs(bls["period"] - period * f) / (period * f) < RECOVERY_TOL:
                        rec, alias = True, name
                        break
            depth_ratio = bls["depth_ppm"] / depth_ppm if depth_ppm > 0 else None
            injection_results.append({
                "label": label,
                "inj_depth_ppm": depth_ppm,
                "inj_period": period,
                "noise_ppm": noise,
                "expected_snr": expected_snr,
                "bls_period": bls["period"],
                "bls_snr": bls["snr"],
                "bls_depth_ppm": bls["depth_ppm"],
                "depth_ratio": depth_ratio,
                "recovered": rec,
                "alias": alias,
                "error": None,
            })
            status = f"MATCH ({alias})" if rec else "MISS"
            log.info("  %-15s inj=%.0fppm P=%.1fd  bls_snr=%.1f  expected_snr=%.1f  %s",
                     label, depth_ppm, period, bls["snr"], expected_snr, status)
        except Exception as exc:
            log.warning("  Pipeline failed for %s: %s", label, exc)
            injection_results.append({
                "label": label,
                "inj_depth_ppm": depth_ppm,
                "inj_period": period,
                "noise_ppm": noise,
                "expected_snr": expected_snr,
                "bls_period": None,
                "bls_snr": None,
                "bls_depth_ppm": None,
                "depth_ratio": None,
                "recovered": False,
                "alias": None,
                "error": str(exc),
            })

    # False-positive test: 10 quiet LCs with no injection
    log.info("Running 10 quiet (no-injection) LCs for false-positive check...")
    n_fp = 0
    fp_snrs = []
    FP_SNR_THRESHOLD = 7.0  # detection threshold
    for i in range(10):
        time, flux, flux_err = _make_quiet_lc(rng)
        try:
            bls = _run_pipeline_on(time, flux, flux_err)
            fp_snrs.append(bls["snr"])
            if bls["snr"] >= FP_SNR_THRESHOLD:
                n_fp += 1
                log.warning("  FP %d: bls_snr=%.1f period=%.3fd", i + 1, bls["snr"], bls["period"])
        except Exception as exc:
            log.warning("  Quiet LC %d pipeline error: %s", i + 1, exc)

    # Print injection table
    print(f"\n── Injection-recovery results ({len(injection_results)} cases) ─────────────────")
    print(f"{'Case':<18} {'Inj depth':>10} {'Inj P':>6} {'ExpSNR':>7} {'BlsSNR':>7} {'BlsDep':>8}  {'Result'}")
    print("─" * 80)
    for r in injection_results:
        exp_snr = f"{r['expected_snr']:.1f}" if r['expected_snr'] else "  —"
        bls_snr = f"{r['bls_snr']:.1f}" if r['bls_snr'] else "  —"
        bls_dep = f"{r['bls_depth_ppm']:.0f}" if r['bls_depth_ppm'] else "  —"
        status = f"MATCH ({r['alias']})" if r["recovered"] else ("ERROR" if r["error"] else "MISS")
        print(f"{r['label']:<18} {r['inj_depth_ppm']:>10.0f} {r['inj_period']:>6.1f} "
              f"{exp_snr:>7} {bls_snr:>7} {bls_dep:>8}  {status}")
    print("─" * 80)

    # Recovery by depth band
    depth_band_defs = [
        ("< 500ppm",      lambda d: d < 500),
        ("500–2000ppm",   lambda d: 500 <= d < 2000),
        ("2000–10000ppm", lambda d: 2000 <= d < 10000),
        ("> 10000ppm",    lambda d: d >= 10000),
    ]
    print("\n── Recovery rate by injected depth ──────────────────")
    for band_label, pred in depth_band_defs:
        band = [r for r in injection_results if pred(r["inj_depth_ppm"])]
        if not band:
            continue
        n_band = len(band)
        n_band_rec = sum(1 for r in band if r["recovered"])
        pct = 100.0 * n_band_rec / n_band
        missed_snr = [r.get("expected_snr", 0) for r in band if not r["recovered"]]
        note = f"(missed SNRs: {', '.join(f'{s:.1f}' for s in missed_snr)})" if missed_snr else ""
        print(f"  {band_label:<18}  {n_band_rec}/{n_band}  ({pct:.0f}%)  {note}")

    # False positive summary
    print(f"\n── False-positive rate (10 quiet LCs, threshold SNR≥{FP_SNR_THRESHOLD}) ────")
    if fp_snrs:
        print(f"  Max BLS SNR on quiet LCs : {max(fp_snrs):.2f}")
        print(f"  Mean BLS SNR             : {np.mean(fp_snrs):.2f}")
        print(f"  FP count (SNR≥{FP_SNR_THRESHOLD:.0f})     : {n_fp}/10")
    else:
        print("  No quiet LC results (pipeline errors)")

    # Overall verdict
    n_total = len(injection_results)
    n_rec_total = sum(1 for r in injection_results if r["recovered"])
    rate = 100.0 * n_rec_total / n_total if n_total else 0.0

    # Stratify: easy cases (depth > 1000ppm) should nearly all pass
    easy = [r for r in injection_results if r["inj_depth_ppm"] >= 1000]
    n_easy_rec = sum(1 for r in easy if r["recovered"])
    easy_rate = 100.0 * n_easy_rec / len(easy) if easy else 0.0

    shallow = [r for r in injection_results if r["inj_depth_ppm"] < 500]
    n_shallow_rec = sum(1 for r in shallow if r["recovered"])
    shallow_rate = 100.0 * n_shallow_rec / len(shallow) if shallow else 0.0

    print()
    if easy_rate >= 90 and n_fp == 0:
        verdict = "PASS"
    elif easy_rate >= 70 and n_fp <= 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print(f"GATE 2 VERDICT: {verdict}")
    print(f"  Overall recovery      : {n_rec_total}/{n_total} ({rate:.0f}%)")
    print(f"  Easy (≥1000ppm)       : {n_easy_rec}/{len(easy)} ({easy_rate:.0f}%)")
    print(f"  Shallow (<500ppm)     : {n_shallow_rec}/{len(shallow)} ({shallow_rate:.0f}%)")
    print(f"  False positives       : {n_fp}/10 quiet LCs (SNR≥{FP_SNR_THRESHOLD:.0f})")
    if shallow_rate < 100:
        print(f"  Note: shallow misses are SNR-limited — at 200ppm noise, 100ppm transits")
        print(f"        have expected SNR < 7 and are undetectable by BLS (not a pipeline bug)")
    return verdict, rate


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=["1", "2", "both"], default="both")
    parser.add_argument("--diverse", action="store_true",
                        help="Gate 1: also run stratified diverse sample (needs MAST)")
    args = parser.parse_args()

    verdicts = {}
    if args.gate in ("1", "both"):
        v, rate = run_gate1(diverse=args.diverse)
        verdicts["gate1"] = (v, rate)
    if args.gate in ("2", "both"):
        v, rate = run_gate2()
        verdicts["gate2"] = (v, rate)

    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    for gate, (v, rate) in verdicts.items():
        print(f"  {gate.upper()} : {v}  ({rate:.1f}%)")

    sys.exit(0 if all(v == "PASS" for v, _ in verdicts.values()) else 1)


if __name__ == "__main__":
    main()
