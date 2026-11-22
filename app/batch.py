"""
Batch job entrypoint.

Usage:
    python -m app.batch --job sync    # catalogue sync
    python -m app.batch --job train   # classifier training (Step 3)
    python -m app.batch --job hunt    # hunt mode (Step 6)
    python -m app.batch --job predict # backyard predictions (Step 8)
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_sync(args) -> None:
    from app.catalogue_sync import sync_koi_catalogue, sync_ps_planets
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        koi_result = sync_koi_catalogue(db, force_refresh=args.force_refresh)
        ps_result = sync_ps_planets(db, force_refresh=args.force_refresh)
    finally:
        db.close()

    print("\n── Catalogue Sync Results ──────────────────────────────")
    print(f"KOI rows fetched : {koi_result['rows_fetched']:,}")
    dist = koi_result.get("label_distribution", {})
    for label, count in sorted(dist.items()):
        print(f"  {label:<20} {count:>6,}")
    print(f"ps_planets rows  : {ps_result['rows_fetched']:,}")
    print("────────────────────────────────────────────────────────\n")


def run_train(args) -> None:
    from app.train import run_training
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        result = run_training(db)
    finally:
        db.close()

    hm = result["honest_metrics"]
    am = result["all_metrics"]
    bs = result["baseline_snr"]
    bm = result["baseline_majority"]

    print("\n── Classifier Training Results ─────────────────────────────")
    print(f"Train / test split : {result['n_train']} / {result['n_test']}")
    print(f"{'Model':<35} {'ROC-AUC':>8} {'Rec@FPR0.1':>11} {'Prec-top100':>12}")
    print(f"{'─'*35} {'─'*8} {'─'*11} {'─'*12}")
    for m in [bm, bs, hm, am]:
        p = f"{m['precision_top100']:.4f}" if m.get("precision_top100") else "    —   "
        print(f"{m['label']:<35} {m['roc_auc']:>8.4f} {m['recall_at_fpr01']:>11.4f} {p:>12}")
    print(f"\nLeakage gap (all − honest):  AUC {am['roc_auc']-hm['roc_auc']:+.4f}  "
          f"recall@FPR0.1 {am['recall_at_fpr01']-hm['recall_at_fpr01']:+.4f}")
    print(f"Model saved: {result['honest_model_path']}")
    print("────────────────────────────────────────────────────────────\n")


def run_hunt(args) -> None:
    from app.config import settings
    from app.db import SessionLocal
    from app.hunt import run_hunt_batch, period_recovered

    db = SessionLocal()
    try:
        summary = run_hunt_batch(db, settings, force=args.force_refresh)
    finally:
        db.close()

    results = summary["results"]
    n_error = summary["n_error"]

    if not results:
        print("\nNo targets processed.")
        return

    n_recovered = sum(1 for r in results if r.get("recovered"))
    n_exact = sum(1 for r in results if r.get("alias") == "exact")
    n_half = sum(1 for r in results if r.get("alias") == "half")
    n_double = sum(1 for r in results if r.get("alias") == "double")
    n_processed = len(results)
    rate = 100.0 * n_recovered / n_processed if n_processed else 0.0

    print(f"\n── Hunt Batch Results (run {summary['hunt_run_id']}) ─────────────────────────")
    print(f"Targets processed  : {n_processed}")
    print(f"Recovered (±{settings.hunt_recovery_tolerance*100:.0f}%)   "
          f": {n_recovered} / {n_processed}  ({rate:.1f}%)")
    print(f"  exact            : {n_exact}")
    print(f"  half-period (*)  : {n_half}")
    print(f"  double-period (*): {n_double}")
    print(f"Errors (LC/BLS)    : {n_error}")
    print(f"Not recovered      : {n_processed - n_recovered - n_error}")

    # Recovery table: all results sorted by BLS SNR descending
    valid = [r for r in results if r.get("bls_period") is not None]
    valid.sort(key=lambda r: r.get("bls_snr") or 0.0, reverse=True)

    if valid:
        print(f"\n── Period Recovery Table (top {min(len(valid), 30)} by BLS SNR) ───────────────")
        print(f"{'kepoi_name':<12}  {'archive_P':>9}  {'bls_P':>9}  {'SNR':>6}  "
              f"{'model_p':>7}  {'recovered':<11}  alias")
        print("─" * 72)
        for r in valid[:30]:
            arch = f"{r['archive_period']:.4f} d" if r["archive_period"] else "  —"
            bls = f"{r['bls_period']:.4f} d" if r["bls_period"] else "  —"
            snr = f"{r['bls_snr']:.1f}" if r["bls_snr"] is not None else "  —"
            sc = f"{r['score']:.3f}" if r.get("score") is not None else "   —"
            rec = "YES" if r["recovered"] else "no"
            al = r.get("alias") or "—"
            al_sfx = " (*)" if al in ("half", "double") else "    "
            print(f"{r['kepoi_name']:<12}  {arch:>9}  {bls:>9}  {snr:>6}  "
                  f"{sc:>7}  {rec:<11}  {al}{al_sfx}")
        print("─" * 72)

    print()


def run_predict(args) -> None:
    # Implemented in Step 8
    log.error("'predict' job not yet implemented (Step 8)")
    sys.exit(1)


JOBS = {
    "sync": run_sync,
    "train": run_train,
    "hunt": run_hunt,
    "predict": run_predict,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="TransitLab batch runner")
    parser.add_argument(
        "--job",
        required=True,
        choices=list(JOBS),
        help="Which batch job to run",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Ignore cached TAP CSVs and re-fetch from NASA",
    )
    args = parser.parse_args()
    JOBS[args.job](args)


if __name__ == "__main__":
    main()
