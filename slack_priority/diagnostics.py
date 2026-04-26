from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import UrgencyMLP, evaluate_model
from .public_data import fetch_public_examples
from .storage import Store
from .text import build_vocab, vectorize


@dataclass(frozen=True)
class DiagnosticExample:
    text: str
    score: float
    source: str
    weight: float


def load_diagnostic_examples(
    *,
    store: Store,
    public_limit: int = 1000,
    local_weight: float = 5.0,
    local_only: bool = False,
) -> list[DiagnosticExample]:
    examples: list[DiagnosticExample] = []
    if not local_only and public_limit > 0:
        examples.extend(
            DiagnosticExample(text=text, score=score, source="public", weight=1.0)
            for text, score in fetch_public_examples(max_rows=public_limit)
        )

    examples.extend(
        DiagnosticExample(text=text, score=score, source="local", weight=local_weight)
        for text, score in store.list_labeled_examples()
    )
    return examples


def run_overfit_check(
    examples: list[DiagnosticExample],
    *,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    validation_fraction: float = 0.2,
    max_vocab: int = 4096,
    min_freq: int = 2,
    shuffle_baseline: bool = True,
) -> dict:
    if len(examples) < 20:
        raise ValueError(
            "Need at least 20 labeled examples for an overfit check. "
            "Label more Slack messages or include public examples with --public-limit."
        )
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")

    texts = [example.text for example in examples]
    targets = np.array([example.score for example in examples], dtype=np.float32)
    weights = np.array([example.weight for example in examples], dtype=np.float32)
    sources = [example.source for example in examples]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(examples))
    validation_size = max(10, int(len(examples) * validation_fraction))
    validation_size = min(validation_size, len(examples) - 1)
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    model, vocab = _train_temporary_model(
        texts=texts,
        targets=targets,
        weights=weights,
        train_indices=train_indices,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_vocab=max_vocab,
        min_freq=min_freq,
    )

    train_metrics = evaluate_model(model, vocab, texts, targets, train_indices)
    validation_metrics = evaluate_model(model, vocab, texts, targets, val_indices)
    constant_score = float(targets[train_indices].mean())
    constant_validation_mae = float(np.abs(targets[val_indices] - constant_score).mean())

    shuffled_metrics = None
    if shuffle_baseline:
        shuffled_targets = targets.copy()
        shuffled_targets[train_indices] = rng.permutation(shuffled_targets[train_indices])
        shuffled_model, shuffled_vocab = _train_temporary_model(
            texts=texts,
            targets=shuffled_targets,
            weights=weights,
            train_indices=train_indices,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_vocab=max_vocab,
            min_freq=min_freq,
        )
        shuffled_metrics = evaluate_model(shuffled_model, shuffled_vocab, texts, targets, val_indices)

    source_counts = {
        "public": sum(1 for source in sources if source == "public"),
        "local": sum(1 for source in sources if source == "local"),
    }
    train_high_plus = _high_plus_count(targets, train_indices)
    validation_high_plus = _high_plus_count(targets, val_indices)
    warnings = _warnings(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        constant_validation_mae=constant_validation_mae,
        shuffled_metrics=shuffled_metrics,
        source_counts=source_counts,
        train_high_plus=train_high_plus,
        validation_high_plus=validation_high_plus,
    )

    return {
        "examples": {
            "total": len(examples),
            "public": source_counts["public"],
            "local": source_counts["local"],
            "train": len(train_indices),
            "validation": len(val_indices),
            "train_high_plus": train_high_plus,
            "validation_high_plus": validation_high_plus,
        },
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "vocab_size": len(vocab),
            "max_vocab": max_vocab,
            "min_freq": min_freq,
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "gaps": {
            "mae": validation_metrics["mae"] - train_metrics["mae"],
            "f1_high_plus": train_metrics["f1_high_plus"] - validation_metrics["f1_high_plus"],
        },
        "baselines": {
            "constant_score": constant_score,
            "constant_validation_mae": constant_validation_mae,
            "shuffled_validation_mae": None if shuffled_metrics is None else shuffled_metrics["mae"],
            "shuffled_validation_f1_high_plus": None
            if shuffled_metrics is None
            else shuffled_metrics["f1_high_plus"],
        },
        "warnings": warnings,
    }


def run_loss_curve(
    examples: list[DiagnosticExample],
    *,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    validation_fraction: float = 0.2,
    max_vocab: int = 4096,
    min_freq: int = 2,
    log_every: int = 1,
) -> dict:
    if len(examples) < 20:
        raise ValueError(
            "Need at least 20 labeled examples for a loss curve. "
            "Label more Slack messages or include public examples with --public-limit."
        )
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    if log_every < 1:
        raise ValueError("log_every must be at least 1")

    texts = [example.text for example in examples]
    targets = np.array([example.score for example in examples], dtype=np.float32)
    weights = np.array([example.weight for example in examples], dtype=np.float32)
    sources = [example.source for example in examples]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(examples))
    validation_size = max(10, int(len(examples) * validation_fraction))
    validation_size = min(validation_size, len(examples) - 1)
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_texts = [texts[i] for i in train_indices]
    vocab = build_vocab(train_texts, max_tokens=max_vocab, min_freq=min_freq)
    if not vocab:
        raise ValueError("No vocabulary terms were produced from the training split.")

    model = UrgencyMLP(len(vocab))
    optimizer = optim.Adam(learning_rate=learning_rate)
    mx.eval(model.state, optimizer.state)

    def loss_fn(model: UrgencyMLP, x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        pred = mx.sigmoid(model(x))
        return (((pred - y) ** 2) * w).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, x, y, w)
        optimizer.update(model, grads)
        return loss

    train_step = mx.compile(
        step,
        inputs=[model.state, optimizer.state],
        outputs=[model.state, optimizer.state],
    )

    points = []
    iteration = 0
    for epoch in range(1, epochs + 1):
        epoch_indices = rng.permutation(train_indices)
        for start in range(0, len(epoch_indices), batch_size):
            batch_indices = epoch_indices[start : start + batch_size]
            x = mx.array(vectorize([texts[i] for i in batch_indices], vocab))
            y = mx.array(targets[batch_indices])
            w = mx.array(weights[batch_indices])
            loss = train_step(x, y, w)
            mx.eval(model.state, optimizer.state, loss)
            iteration += 1

            if iteration == 1 or iteration % log_every == 0:
                points.append(
                    {
                        "iteration": iteration,
                        "epoch": epoch,
                        "train_loss": float(loss.item()),
                        "validation_loss": _loss_for_indices(
                            model=model,
                            vocab=vocab,
                            texts=texts,
                            targets=targets,
                            weights=weights,
                            indices=val_indices,
                            batch_size=batch_size,
                        ),
                    }
                )

    source_counts = {
        "public": sum(1 for source in sources if source == "public"),
        "local": sum(1 for source in sources if source == "local"),
    }
    return {
        "examples": {
            "total": len(examples),
            "public": source_counts["public"],
            "local": source_counts["local"],
            "train": len(train_indices),
            "validation": len(val_indices),
        },
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "vocab_size": len(vocab),
            "max_vocab": max_vocab,
            "min_freq": min_freq,
            "log_every": log_every,
        },
        "points": points,
    }


def plot_loss_curve(report: dict, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = report["points"]
    iterations = [point["iteration"] for point in points]
    train_losses = [point["train_loss"] for point in points]
    validation_losses = [point["validation_loss"] for point in points]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, train_losses, color="#2563eb", linewidth=2, label="train batch loss")
    ax.plot(
        iterations,
        validation_losses,
        color="#dc2626",
        linewidth=2,
        marker="o",
        markersize=3,
        label="validation loss",
    )
    ax.set_title("Slack Focus Loss Curve")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Weighted MSE loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    examples = report["examples"]
    settings = report["settings"]
    caption = (
        f"{examples['train']} train / {examples['validation']} validation examples, "
        f"epochs={settings['epochs']}, batch={settings['batch_size']}, vocab={settings['vocab_size']}"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def format_overfit_report(report: dict) -> str:
    examples = report["examples"]
    settings = report["settings"]
    train = report["train"]
    validation = report["validation"]
    gaps = report["gaps"]
    baselines = report["baselines"]
    warnings = report["warnings"]

    lines = [
        "Overfit check",
        (
            f"Examples: {examples['total']} total "
            f"({examples['public']} public, {examples['local']} local); "
            f"{examples['train']} train / {examples['validation']} validation"
        ),
        (
            f"High+ labels: {examples['train_high_plus']} train / "
            f"{examples['validation_high_plus']} validation"
        ),
        (
            f"Settings: epochs={settings['epochs']}, batch_size={settings['batch_size']}, "
            f"vocab={settings['vocab_size']}/{settings['max_vocab']}, seed={settings['seed']}"
        ),
        "",
        "Metrics",
        (
            f"  train:      mae={train['mae']:.4f}, "
            f"precision_high+={train['precision_high_plus']:.3f}, "
            f"recall_high+={train['recall_high_plus']:.3f}, "
            f"f1_high+={train['f1_high_plus']:.3f}"
        ),
        (
            f"  validation: mae={validation['mae']:.4f}, "
            f"precision_high+={validation['precision_high_plus']:.3f}, "
            f"recall_high+={validation['recall_high_plus']:.3f}, "
            f"f1_high+={validation['f1_high_plus']:.3f}"
        ),
        (
            f"  gap:        mae={gaps['mae']:.4f}, "
            f"f1_high+={gaps['f1_high_plus']:.3f}"
        ),
        "",
        "Baselines",
        (
            f"  constant validation mae: {baselines['constant_validation_mae']:.4f} "
            f"(constant score={baselines['constant_score']:.3f})"
        ),
    ]

    if baselines["shuffled_validation_mae"] is not None:
        lines.append(
            f"  shuffled-label validation mae: {baselines['shuffled_validation_mae']:.4f}"
        )
        lines.append(
            f"  shuffled-label validation f1_high+: "
            f"{baselines['shuffled_validation_f1_high_plus']:.3f}"
        )

    lines.append("")
    if warnings:
        lines.append("Warnings")
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none")

    return "\n".join(lines)


def _loss_for_indices(
    *,
    model: UrgencyMLP,
    vocab: dict[str, int],
    texts: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    weighted_loss = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        x = mx.array(vectorize([texts[i] for i in batch_indices], vocab))
        y = mx.array(targets[batch_indices])
        w = mx.array(weights[batch_indices])
        pred = mx.sigmoid(model(x))
        loss = (((pred - y) ** 2) * w).mean()
        mx.eval(loss)
        weighted_loss += float(loss.item()) * len(batch_indices)
        count += len(batch_indices)
    return weighted_loss / max(count, 1)


def _train_temporary_model(
    *,
    texts: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    train_indices: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_vocab: int,
    min_freq: int,
) -> tuple[UrgencyMLP, dict[str, int]]:
    train_texts = [texts[i] for i in train_indices]
    vocab = build_vocab(train_texts, max_tokens=max_vocab, min_freq=min_freq)
    if not vocab:
        raise ValueError("No vocabulary terms were produced from the training split.")

    model = UrgencyMLP(len(vocab))
    optimizer = optim.Adam(learning_rate=learning_rate)
    mx.eval(model.state, optimizer.state)

    def loss_fn(model: UrgencyMLP, x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        pred = mx.sigmoid(model(x))
        return (((pred - y) ** 2) * w).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, x, y, w)
        optimizer.update(model, grads)
        return loss

    train_step = mx.compile(
        step,
        inputs=[model.state, optimizer.state],
        outputs=[model.state, optimizer.state],
    )

    rng = np.random.default_rng(12345)
    for _ in range(epochs):
        epoch_indices = rng.permutation(train_indices)
        for start in range(0, len(epoch_indices), batch_size):
            batch_indices = epoch_indices[start : start + batch_size]
            x = mx.array(vectorize([texts[i] for i in batch_indices], vocab))
            y = mx.array(targets[batch_indices])
            w = mx.array(weights[batch_indices])
            loss = train_step(x, y, w)
            mx.eval(model.state, optimizer.state, loss)

    return model, vocab


def _high_plus_count(targets: np.ndarray, indices: np.ndarray) -> int:
    return int((targets[indices] >= 0.80).sum())


def _warnings(
    *,
    train_metrics: dict[str, float | int],
    validation_metrics: dict[str, float | int],
    constant_validation_mae: float,
    shuffled_metrics: dict[str, float | int] | None,
    source_counts: dict[str, int],
    train_high_plus: int,
    validation_high_plus: int,
) -> list[str]:
    warnings: list[str] = []
    mae_gap = validation_metrics["mae"] - train_metrics["mae"]
    f1_gap = train_metrics["f1_high_plus"] - validation_metrics["f1_high_plus"]

    if source_counts["local"] < 30:
        warnings.append(
            "You have fewer than 30 local labels, so personal overfit checks are noisy. "
            "Keep labeling mistakes."
        )
    if train_high_plus == 0 or validation_high_plus == 0:
        warnings.append(
            "One split has no high/critical labels, so high+ precision/recall is not very informative."
        )
    if train_high_plus > 0 and train_metrics["recall_high_plus"] == 0:
        warnings.append(
            "Train high+ recall is 0. The model is not learning the urgent examples yet."
        )
    if validation_high_plus > 0 and validation_metrics["recall_high_plus"] == 0:
        warnings.append(
            "Validation high+ recall is 0. The model is missing urgent examples in the holdout split."
        )
    if mae_gap > 0.15:
        warnings.append(
            "Validation MAE is much worse than train MAE. That is a classic overfitting signal."
        )
    if f1_gap > 0.30 and train_high_plus > 0 and validation_high_plus > 0:
        warnings.append(
            "Train high+ F1 is much higher than validation high+ F1. The model may be memorizing."
        )
    if validation_metrics["mae"] >= constant_validation_mae:
        warnings.append(
            "Validation MAE is not better than predicting the train-set average score."
        )
    if shuffled_metrics and validation_metrics["mae"] >= shuffled_metrics["mae"] - 0.02:
        warnings.append(
            "The real-label model is not clearly better than the shuffled-label baseline."
        )

    return warnings
