"""Admin CLI for INFERNIS - key management and grid initialization.

Usage:
    python -m infernis.admin create_key --name "My App" --tier free
    python -m infernis.admin list_keys
    python -m infernis.admin init_grid
    python -m infernis.admin run_pipeline [--date 2025-07-15]
"""

import argparse
import hashlib
import logging
import secrets
import sys
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("infernis.admin")


def create_key(args):
    """Generate a new API key."""
    from infernis.db.engine import SessionLocal
    from infernis.db.tables import APIKeyDB

    raw_key = secrets.token_hex(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    tier_limits = {"free": 50, "pro": 10_000, "enterprise": 100_000}

    db = SessionLocal()
    try:
        record = APIKeyDB(
            key_hash=key_hash,
            name=args.name,
            tier=args.tier,
            daily_limit=tier_limits.get(args.tier, 50),
            is_active=True,
        )
        db.add(record)
        db.commit()
        print("API Key created:")
        print(f"  Name: {args.name}")
        print(f"  Tier: {args.tier}")
        print(f"  Key:  {raw_key}")
        print("  (Store this key securely - it cannot be retrieved later)")
    finally:
        db.close()


def list_keys(args):
    """List all API keys."""
    from infernis.db.engine import SessionLocal
    from infernis.db.tables import APIKeyDB

    db = SessionLocal()
    try:
        keys = db.query(APIKeyDB).all()
        if not keys:
            print("No API keys found")
            return
        print(f"{'ID':<5} {'Name':<30} {'Tier':<12} {'Limit':<8} {'Today':<8} {'Active':<8}")
        print("-" * 75)
        for k in keys:
            print(
                f"{k.id:<5} {k.name:<30} {k.tier:<12} {k.daily_limit:<8} {k.requests_today:<8} {k.is_active}"
            )
    finally:
        db.close()


def init_grid(args):
    """Initialize the BC grid with static features."""
    from infernis.grid.initializer import grid_to_db, initialize_grid

    grid = initialize_grid()
    print(f"Generated grid with {len(grid)} cells")

    if args.save_db:
        count = grid_to_db(grid)
        print(f"Saved {count} cells to database")

    if args.save_csv:
        out = args.save_csv
        grid.drop(columns=["geometry"], errors="ignore").to_csv(out, index=False)
        print(f"Grid saved to {out}")


def run_pipeline(args):
    """Run the daily pipeline manually."""
    from infernis.pipelines.runner import run_daily_pipeline

    target = date.fromisoformat(args.date) if args.date else None
    predictions = run_daily_pipeline(target_date=target)
    print(f"Pipeline complete: {len(predictions)} cells processed")


def cleanup(args):
    """Delete old predictions and pipeline runs based on retention settings."""
    from infernis.pipelines.runner import cleanup_old_data

    pred_days = args.days or None
    run_days = args.run_days or None
    cleanup_old_data(prediction_days=pred_days, pipeline_run_days=run_days)
    print("Cleanup complete. Check logs for details.")


def main():
    parser = argparse.ArgumentParser(description="INFERNIS Admin CLI")
    sub = parser.add_subparsers(dest="command")

    # create_key
    p_key = sub.add_parser("create_key", help="Create a new API key")
    p_key.add_argument("--name", required=True, help="Key name/description")
    p_key.add_argument("--tier", choices=["free", "pro", "enterprise"], default="free")

    # list_keys
    sub.add_parser("list_keys", help="List all API keys")

    # init_grid
    p_grid = sub.add_parser("init_grid", help="Initialize BC grid")
    p_grid.add_argument("--save-db", action="store_true", help="Save to database")
    p_grid.add_argument("--save-csv", type=str, help="Save to CSV file")

    # run_pipeline
    p_run = sub.add_parser("run_pipeline", help="Run daily pipeline manually")
    p_run.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")

    # cleanup
    p_clean = sub.add_parser("cleanup", help="Delete old predictions and pipeline runs")
    p_clean.add_argument(
        "--days", type=int, default=0, help="Prediction retention days (default: from config)"
    )
    p_clean.add_argument(
        "--run-days", type=int, default=0, help="Pipeline run retention days (default: from config)"
    )

    args = parser.parse_args()

    if args.command == "create_key":
        create_key(args)
    elif args.command == "list_keys":
        list_keys(args)
    elif args.command == "init_grid":
        init_grid(args)
    elif args.command == "run_pipeline":
        run_pipeline(args)
    elif args.command == "cleanup":
        cleanup(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
