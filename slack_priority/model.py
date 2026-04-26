from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import DEFAULT_THRESHOLDS, METRICS_PATH, MODEL_PATH, VOCAB_PATH, ensure_dirs, score_to_level
from .public_data import fetch_public_examples
from .storage import Store
from .text import build_vocab, vectorize


MODEL_VERSION = "bow-mlp-v1"


class UrgencyMLP(nn.Module):
    def __init__(self, input_dims: int):
        super().__init__()
        self.layers = [
            nn.Linear(input_dims, 64),
            nn.Linear(64, 1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.maximum(self.layers[0](x), 0)
        return self.layers[1](x).squeeze(-1)


@dataclass(frozen=True)
class TrainingResult:
    public_examples: int
    local_examples: int
    train_examples: int
    validation_examples: int
    metrics: dict[str, float | int | str]


class UrgencyScorer:
    def __init__(
        self,
        *,
        model_path: Path = MODEL_PATH,
        vocab_path: Path = VOCAB_PATH,
        thresholds: dict[str, float] | None = None,
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.vocab = self._load_vocab()
        self.model = UrgencyMLP(len(self.vocab))
        if self.model_path.exists() and self.vocab:
            self.model.load_weights(str(self.model_path))
            mx.eval(self.model.parameters())

    @property
    def ready(self) -> bool:
        return bool(self.vocab) and self.model_path.exists()

    def score(self, text: str) -> float:
        if not self.ready:
            return 0.0

        features = vectorize([text], self.vocab)
        logits = self.model(mx.array(features))
        score = mx.sigmoid(logits)[0]
        return float(score.item())

    def score_level(self, text: str) -> tuple[float, str]:
        score = self.score(text)
        return score, score_to_level(score, self.thresholds)

    def _load_vocab(self) -> dict[str, int]:
        if not self.vocab_path.exists():
            return {}
        return json.loads(self.vocab_path.read_text())


def train_model(
    *,
    store: Store | None = None,
    public_limit: int = 8000,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    local_weight: float = 5.0,
    max_vocab: int = 4096,
    min_freq: int = 2,
) -> TrainingResult:
    ensure_dirs()
    owns_store = store is None
    store = store or Store()
    try:
        public_examples = fetch_public_examples(max_rows=public_limit)
        local_examples = store.list_labeled_examples()
    finally:
        if owns_store:
            store.close()

    texts = [text for text, _ in public_examples] + [text for text, _ in local_examples]
    targets = [score for _, score in public_examples] + [score for _, score in local_examples]
    weights = [1.0] * len(public_examples) + [local_weight] * len(local_examples)

    if len(texts) < 20:
        raise RuntimeError("Need at least 20 examples to train.")

    vocab = build_vocab(texts, max_tokens=max_vocab, min_freq=min_freq)
    if not vocab:
        raise RuntimeError("No vocabulary terms were produced from training data.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(texts))
    validation_size = max(10, int(len(texts) * 0.2))
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    model = UrgencyMLP(len(vocab))
    optimizer = optim.Adam(learning_rate=learning_rate)
    mx.eval(model.parameters())

    def loss_fn(model: UrgencyMLP, x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        logits = model(x)
        pred = mx.sigmoid(logits)
        return (((pred - y) ** 2) * w).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    target_array = np.array(targets, dtype=np.float32)
    weight_array = np.array(weights, dtype=np.float32)

    for _ in range(epochs):
        epoch_indices = rng.permutation(train_indices)
        for start in range(0, len(epoch_indices), batch_size):
            batch_indices = epoch_indices[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            x = mx.array(vectorize(batch_texts, vocab))
            y = mx.array(target_array[batch_indices])
            w = mx.array(weight_array[batch_indices])
            loss, grads = loss_and_grad(model, x, y, w)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    metrics = evaluate_model(model, vocab, texts, target_array, val_indices)
    metrics.update(
        {
            "model_version": MODEL_VERSION,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "public_examples": len(public_examples),
            "local_examples": len(local_examples),
            "vocab_size": len(vocab),
            "epochs": epochs,
        }
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(MODEL_PATH))
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2) + "\n")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    return TrainingResult(
        public_examples=len(public_examples),
        local_examples=len(local_examples),
        train_examples=len(train_indices),
        validation_examples=len(val_indices),
        metrics=metrics,
    )


def evaluate_model(
    model: UrgencyMLP,
    vocab: dict[str, int],
    texts: list[str],
    targets: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    predictions = []
    for start in range(0, len(indices), 256):
        batch_indices = indices[start : start + 256]
        x = mx.array(vectorize([texts[i] for i in batch_indices], vocab))
        scores = mx.sigmoid(model(x))
        predictions.extend(np.array(scores, dtype=np.float32).tolist())

    y_true = targets[indices]
    y_pred = np.array(predictions, dtype=np.float32)
    true_high = y_true >= 0.80
    pred_high = y_pred >= DEFAULT_THRESHOLDS["high"]

    tp = int(np.logical_and(true_high, pred_high).sum())
    fp = int(np.logical_and(~true_high, pred_high).sum())
    fn = int(np.logical_and(true_high, ~pred_high).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "mae": float(np.abs(y_true - y_pred).mean()),
        "precision_high_plus": precision,
        "recall_high_plus": recall,
        "f1_high_plus": f1,
        "true_positive_high_plus": tp,
        "false_positive_high_plus": fp,
        "false_negative_high_plus": fn,
    }
