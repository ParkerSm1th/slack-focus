from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SETTINGS_PATH = DATA_DIR / "settings.json"
DB_PATH = DATA_DIR / "slack_priority.sqlite3"
PUBLIC_DATA_PATH = DATA_DIR / "public_priority.jsonl"
MODEL_PATH = MODEL_DIR / "urgent_classifier.safetensors"
VOCAB_PATH = MODEL_DIR / "vocab.json"
METRICS_PATH = MODEL_DIR / "metrics.json"


LABEL_SCORES = {
    "ignore": 0.0,
    "low": 0.35,
    "high": 0.80,
    "critical": 1.0,
}

PUBLIC_PRIORITY_SCORES = {
    "low": 0.15,
    "medium": 0.50,
    "high": 0.85,
    "critical": 0.98,
}

DEFAULT_THRESHOLDS = {
    "low": 0.35,
    "medium": 0.55,
    "high": 0.75,
    "critical": 0.90,
}


@dataclass(frozen=True)
class Settings:
    slack_app_token: str | None
    slack_user_token: str | None
    notify_min_level: str
    thresholds: dict[str, float]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def read_settings() -> Settings:
    ensure_dirs()
    file_settings = {}
    if SETTINGS_PATH.exists():
        file_settings = json.loads(SETTINGS_PATH.read_text())

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(file_settings.get("thresholds", {}))

    return Settings(
        slack_app_token=os.getenv("SLACK_APP_TOKEN") or file_settings.get("slack_app_token"),
        slack_user_token=os.getenv("SLACK_USER_TOKEN") or file_settings.get("slack_user_token"),
        notify_min_level=os.getenv("SLACK_PRIORITY_NOTIFY_MIN_LEVEL")
        or file_settings.get("notify_min_level", "high"),
        thresholds=thresholds,
    )


def write_settings(
    *,
    slack_app_token: str | None = None,
    slack_user_token: str | None = None,
    notify_min_level: str = "high",
    thresholds: dict[str, float] | None = None,
) -> None:
    ensure_dirs()
    payload = {
        "slack_app_token": slack_app_token,
        "slack_user_token": slack_user_token,
        "notify_min_level": notify_min_level,
        "thresholds": thresholds or DEFAULT_THRESHOLDS,
    }
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def score_to_level(score: float, thresholds: dict[str, float] | None = None) -> str:
    levels = thresholds or DEFAULT_THRESHOLDS
    if score >= levels["critical"]:
        return "critical"
    if score >= levels["high"]:
        return "high"
    if score >= levels["medium"]:
        return "medium"
    if score >= levels["low"]:
        return "low"
    return "ignore"


def should_notify(level: str, min_level: str = "high") -> bool:
    order = {"ignore": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    return order[level] >= order[min_level]
