from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import sys
from pathlib import Path

from .config import DATA_DIR, METRICS_PATH, read_settings, write_settings
from .config import score_to_level
from .diagnostics import (
    format_overfit_report,
    load_diagnostic_examples,
    plot_loss_curve,
    run_loss_curve,
    run_overfit_check,
)
from .model import MODEL_VERSION, UrgencyScorer, train_model
from .review import ReviewServer
from .slack_client import SlackPriorityService
from .storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(prog="slack_priority")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="Save Slack tokens to local data/settings.json.")

    backfill = subparsers.add_parser("backfill", help="Import recent Slack messages.")
    backfill.add_argument("--days", type=int, default=14)
    backfill.add_argument("--max-conversations", type=int, default=None)
    backfill.add_argument(
        "--channel",
        action="append",
        default=None,
        help="Specific Slack channel/conversation ID to backfill. Can be repeated.",
    )
    backfill.add_argument(
        "--types",
        default="public_channel,private_channel,im,mpim",
        help="Slack conversation types to scan, comma-separated.",
    )
    backfill.add_argument("--verbose", action="store_true", help="Print per-conversation backfill details.")

    train = subparsers.add_parser("train", help="Train the local MLX classifier.")
    train.add_argument("--public-limit", type=int, default=8000)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--local-weight", type=float, default=5.0)
    train.add_argument("--learning-rate", type=float, default=1e-3)

    overfit = subparsers.add_parser("overfit-check", help="Run train/validation diagnostics for overfitting.")
    overfit.add_argument("--public-limit", type=int, default=1000)
    overfit.add_argument("--local-only", action="store_true", help="Use only your locally labeled Slack examples.")
    overfit.add_argument("--epochs", type=int, default=5)
    overfit.add_argument("--batch-size", type=int, default=128)
    overfit.add_argument("--local-weight", type=float, default=5.0)
    overfit.add_argument("--learning-rate", type=float, default=1e-3)
    overfit.add_argument("--seed", type=int, default=42)
    overfit.add_argument("--validation-fraction", type=float, default=0.2)
    overfit.add_argument("--max-vocab", type=int, default=4096)
    overfit.add_argument("--min-freq", type=int, default=2)
    overfit.add_argument("--no-shuffle-baseline", action="store_true")
    overfit.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    loss_curve = subparsers.add_parser("learning-curve", help="Plot training iteration vs loss.")
    loss_curve.add_argument("--output", default=str(DATA_DIR / "loss_curve.png"))
    loss_curve.add_argument("--public-limit", type=int, default=1000)
    loss_curve.add_argument("--local-only", action="store_true", help="Use only your locally labeled Slack examples.")
    loss_curve.add_argument("--epochs", type=int, default=5)
    loss_curve.add_argument("--batch-size", type=int, default=128)
    loss_curve.add_argument("--local-weight", type=float, default=5.0)
    loss_curve.add_argument("--learning-rate", type=float, default=1e-3)
    loss_curve.add_argument("--seed", type=int, default=42)
    loss_curve.add_argument("--validation-fraction", type=float, default=0.2)
    loss_curve.add_argument("--max-vocab", type=int, default=4096)
    loss_curve.add_argument("--min-freq", type=int, default=2)
    loss_curve.add_argument("--log-every", type=int, default=1)
    loss_curve.add_argument("--json", action="store_true", help="Print curve data as JSON instead of plotting.")

    score = subparsers.add_parser("score", help="Score one text string.")
    score.add_argument("text")

    label_text = subparsers.add_parser("label-text", help="Add a manually labeled training example.")
    label_text.add_argument("text")
    label_text.add_argument("--label", required=True, choices=["ignore", "low", "high", "critical"])

    review = subparsers.add_parser("review", help="Open the local review queue.")
    review.add_argument("--no-open", action="store_true")

    subparsers.add_parser("listen", help="Run live Slack monitoring in this terminal.")
    rescore = subparsers.add_parser("rescore", help="Refresh stored message scores with current model.")
    rescore.add_argument("--limit", type=int, default=5000)
    subparsers.add_parser("metrics", help="Print the latest training metrics.")
    conversations = subparsers.add_parser("conversations", help="List visible Slack conversations.")
    conversations.add_argument("--types", default="public_channel,private_channel,im,mpim")
    conversations.add_argument("--limit", type=int, default=50)
    debug_channel = subparsers.add_parser("debug-channel", help="Debug Slack history access for one channel.")
    debug_channel.add_argument("channel_id")
    debug_channel.add_argument("--days", type=int, default=14)
    debug_channel.add_argument("--latest-ts", default=None)

    args = parser.parse_args()

    if args.command == "setup":
        app_token = getpass.getpass("Slack app token (xapp-): ").strip()
        user_token = getpass.getpass("Slack user token (xoxp-): ").strip()
        write_settings(slack_app_token=app_token, slack_user_token=user_token)
        print("Saved local settings to data/settings.json")
        return

    if args.command == "backfill":
        service = SlackPriorityService(read_settings())
        try:
            count = service.backfill(
                days=args.days,
                max_conversations=args.max_conversations,
                conversation_types=args.types,
                channel_ids=args.channel,
                on_status=print,
                verbose=args.verbose,
            )
            print(f"Imported {count} messages")
        finally:
            service.close()
        return

    if args.command == "train":
        result = train_model(
            public_limit=args.public_limit,
            epochs=args.epochs,
            local_weight=args.local_weight,
            learning_rate=args.learning_rate,
        )
        print(json.dumps(result.metrics, indent=2))
        return

    if args.command == "overfit-check":
        store = Store()
        try:
            examples = load_diagnostic_examples(
                store=store,
                public_limit=args.public_limit,
                local_weight=args.local_weight,
                local_only=args.local_only,
            )
        finally:
            store.close()
        try:
            report = run_overfit_check(
                examples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                max_vocab=args.max_vocab,
                min_freq=args.min_freq,
                shuffle_baseline=not args.no_shuffle_baseline,
            )
        except ValueError as error:
            print(str(error))
            sys.exit(1)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(format_overfit_report(report))
        return

    if args.command == "learning-curve":
        store = Store()
        try:
            examples = load_diagnostic_examples(
                store=store,
                public_limit=args.public_limit,
                local_weight=args.local_weight,
                local_only=args.local_only,
            )
        finally:
            store.close()
        try:
            report = run_loss_curve(
                examples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                max_vocab=args.max_vocab,
                min_freq=args.min_freq,
                log_every=args.log_every,
            )
        except ValueError as error:
            print(str(error))
            sys.exit(1)

        if args.json:
            print(json.dumps(report, indent=2))
            return

        try:
            output_path = plot_loss_curve(report, Path(args.output))
        except ImportError:
            print(
                "Missing matplotlib. Run this command with: "
                "uv run --with matplotlib python -m slack_priority learning-curve"
            )
            sys.exit(1)
        print(f"Wrote {output_path}")
        return

    if args.command == "score":
        scorer = UrgencyScorer(thresholds=read_settings().thresholds)
        if not scorer.ready:
            print("Model is not trained yet. Run: uv run python -m slack_priority train")
            sys.exit(1)
        score_value, level = scorer.score_level(args.text)
        print(json.dumps({"score": score_value, "level": level}, indent=2))
        return

    if args.command == "label-text":
        message_id = "manual:" + hashlib.sha256(args.text.encode("utf-8")).hexdigest()[:24]
        store = Store()
        try:
            store.upsert_message(
                message_id=message_id,
                channel_id="manual",
                channel_name="manual",
                user_id="you",
                text=args.text,
            )
            store.set_label(message_id, args.label)
        finally:
            store.close()
        print(f"Saved {args.label} example: {message_id}")
        return

    if args.command == "review":
        server = ReviewServer()
        url = server.start()
        print(url)
        if not args.no_open:
            server.open()
        try:
            input("Press Enter to stop the review server...")
        finally:
            server.stop()
        return

    if args.command == "listen":
        service = SlackPriorityService(read_settings())
        try:
            service.listen_forever(on_status=print)
        finally:
            service.close()
        return

    if args.command == "rescore":
        settings = read_settings()
        scorer = UrgencyScorer(thresholds=settings.thresholds)
        if not scorer.ready:
            print("Model is not trained yet. Run: uv run python -m slack_priority train")
            sys.exit(1)
        store = Store()
        try:
            messages = store.recent_messages(limit=args.limit)
            for message in messages:
                score_value = scorer.score(message.text)
                level = score_to_level(score_value, settings.thresholds)
                store.upsert_prediction(
                    message_id=message.message_id,
                    score=score_value,
                    level=level,
                    model_version=MODEL_VERSION,
                )
        finally:
            store.close()
        print(f"Rescored {len(messages)} messages")
        return

    if args.command == "conversations":
        service = SlackPriorityService(read_settings())
        try:
            rows = service.list_conversations(types=args.types, limit=args.limit)
        finally:
            service.close()
        for row in rows:
            name = row.get("name") or row.get("user") or row.get("id")
            print(f"{row.get('id')}\t{name}\t{row.get('is_im', False)}")
        return

    if args.command == "debug-channel":
        service = SlackPriorityService(read_settings())
        try:
            result = service.debug_channel_history(
                channel_id=args.channel_id,
                days=args.days,
                latest_ts=args.latest_ts,
            )
        finally:
            service.close()
        print(json.dumps(result, indent=2))
        return

    if args.command == "metrics":
        if not METRICS_PATH.exists():
            print("No metrics yet. Train the model first.")
            sys.exit(1)
        print(METRICS_PATH.read_text())
        return


if __name__ == "__main__":
    main()
