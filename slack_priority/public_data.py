from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path

from .config import PUBLIC_DATA_PATH, PUBLIC_PRIORITY_SCORES, ensure_dirs


DATASET_NAME = "Prady06/customer-support-tickets"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_public_examples(
    *,
    path: Path = PUBLIC_DATA_PATH,
    max_rows: int = 8000,
    page_size: int = 100,
    refresh: bool = False,
) -> list[tuple[str, float]]:
    ensure_dirs()
    existing = load_public_examples(path) if path.exists() else []
    if path.exists() and not refresh:
        if len(existing) >= max_rows:
            return existing[:max_rows]

    written = 0
    offset = 0
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("w") as file:
            while written < max_rows:
                params = urllib.parse.urlencode(
                    {
                        "dataset": DATASET_NAME,
                        "config": "default",
                        "split": "train",
                        "offset": offset,
                        "length": min(page_size, max_rows - written),
                    }
                )
                payload = _request_json(f"{ROWS_ENDPOINT}?{params}")

                rows = payload.get("rows", [])
                if not rows:
                    break

                for item in rows:
                    row = item.get("row", {})
                    if str(row.get("language", "")).lower() != "en":
                        continue

                    priority = str(row.get("priority", "")).lower()
                    score = PUBLIC_PRIORITY_SCORES.get(priority)
                    if score is None:
                        continue

                    text = " ".join(
                        part.strip()
                        for part in [str(row.get("subject") or ""), str(row.get("body") or "")]
                        if part and part.strip()
                    )
                    if not text:
                        continue

                    file.write(json.dumps({"text": text, "score": score}) + "\n")
                    written += 1
                    if written >= max_rows:
                        break

                offset += len(rows)
                time.sleep(0.5)

        temp_path.replace(path)
    except (HTTPError, URLError, TimeoutError):
        if temp_path.exists():
            temp_path.unlink()
        if existing:
            return existing[:max_rows]
        raise

    return load_public_examples(path)[:max_rows]


def _request_json(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code != 429 or attempt == retries - 1:
                raise
            retry_after = int(error.headers.get("Retry-After", str(2 ** (attempt + 1))))
            time.sleep(retry_after)
        except (URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Unreachable dataset request state.")


def load_public_examples(path: Path = PUBLIC_DATA_PATH) -> list[tuple[str, float]]:
    if not path.exists():
        return []

    examples = []
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append((row["text"], float(row["score"])))
    return examples
