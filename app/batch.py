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
    # Implemented in Step 6
    log.error("'hunt' job not yet implemented (Step 6)")
    sys.exit(1)


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
