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
    # Implemented in Step 3
    log.error("'train' job not yet implemented (Step 3)")
    sys.exit(1)


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
