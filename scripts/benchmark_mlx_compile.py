from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from slack_priority.model import UrgencyMLP


@dataclass(frozen=True)
class Timing:
    eager_ms: float
    compiled_first_ms: float
    compiled_ms: float

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.compiled_ms if self.compiled_ms else 0.0


def timed(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1000.0


def build_predict(model: UrgencyMLP):
    def predict(x: mx.array) -> mx.array:
        return mx.sigmoid(model(x))

    return predict


def benchmark_inference(*, vocab_size: int, batch_size: int, iterations: int) -> Timing:
    model = UrgencyMLP(vocab_size)
    x = mx.random.uniform(shape=(batch_size, vocab_size)).astype(mx.float32)
    mx.eval(model.state, x)

    eager_predict = build_predict(model)
    compiled_predict = mx.compile(eager_predict, inputs=model.state)

    mx.eval(eager_predict(x))
    eager_ms = timed(lambda: mx.eval(eager_predict(x)), iterations)

    start = time.perf_counter()
    mx.eval(compiled_predict(x))
    compiled_first_ms = (time.perf_counter() - start) * 1000.0

    compiled_ms = timed(lambda: mx.eval(compiled_predict(x)), iterations)
    return Timing(eager_ms=eager_ms, compiled_first_ms=compiled_first_ms, compiled_ms=compiled_ms)


def build_train_step(model: UrgencyMLP, optimizer: optim.Optimizer):
    def loss_fn(model: UrgencyMLP, x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        pred = mx.sigmoid(model(x))
        return (((pred - y) ** 2) * w).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x: mx.array, y: mx.array, w: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, x, y, w)
        optimizer.update(model, grads)
        return loss

    return step


def benchmark_training(*, vocab_size: int, batch_size: int, iterations: int) -> Timing:
    x = mx.random.uniform(shape=(batch_size, vocab_size)).astype(mx.float32)
    y = mx.random.uniform(shape=(batch_size,)).astype(mx.float32)
    w = mx.ones((batch_size,), dtype=mx.float32)
    mx.eval(x, y, w)

    eager_model = UrgencyMLP(vocab_size)
    eager_optimizer = optim.Adam(learning_rate=1e-3)
    eager_step = build_train_step(eager_model, eager_optimizer)
    mx.eval(eager_model.state, eager_optimizer.state)
    mx.eval(eager_step(x, y, w))
    eager_ms = timed(lambda: mx.eval(eager_model.state, eager_optimizer.state, eager_step(x, y, w)), iterations)

    compiled_model = UrgencyMLP(vocab_size)
    compiled_optimizer = optim.Adam(learning_rate=1e-3)
    compiled_step = build_train_step(compiled_model, compiled_optimizer)
    compiled_step = mx.compile(
        compiled_step,
        inputs=[compiled_model.state, compiled_optimizer.state],
        outputs=[compiled_model.state, compiled_optimizer.state],
    )
    mx.eval(compiled_model.state, compiled_optimizer.state)

    start = time.perf_counter()
    first_loss = compiled_step(x, y, w)
    mx.eval(compiled_model.state, compiled_optimizer.state, first_loss)
    compiled_first_ms = (time.perf_counter() - start) * 1000.0

    compiled_ms = timed(
        lambda: mx.eval(compiled_model.state, compiled_optimizer.state, compiled_step(x, y, w)),
        iterations,
    )
    return Timing(eager_ms=eager_ms, compiled_first_ms=compiled_first_ms, compiled_ms=compiled_ms)


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return batch_sizes


def print_result(name: str, timing: Timing, iterations: int) -> None:
    print(f"\n{name}")
    print(f"  eager steady-state:     {timing.eager_ms:9.2f} ms total  {timing.eager_ms / iterations:8.4f} ms/iter")
    print(f"  compiled first call:    {timing.compiled_first_ms:9.2f} ms  (compile + run)")
    print(f"  compiled steady-state:  {timing.compiled_ms:9.2f} ms total  {timing.compiled_ms / iterations:8.4f} ms/iter")
    print(f"  steady-state speedup:   {timing.speedup:9.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eager vs mx.compile performance for the Slack Focus MLX MLP.",
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,128"))
    parser.add_argument("--inference-iterations", type=int, default=1000)
    parser.add_argument("--training-iterations", type=int, default=200)
    args = parser.parse_args()

    print("Slack Focus MLX compile benchmark")
    print(f"  vocab_size:            {args.vocab_size}")
    print(f"  inference_iterations:  {args.inference_iterations}")
    print(f"  training_iterations:   {args.training_iterations}")
    print("\nNotes:")
    print("  - Timings sync with mx.eval every iteration to measure completed work.")
    print("  - This isolates the MLX graph. It does not include Slack I/O or Python text vectorization.")
    print("  - The first compiled call includes compilation. Steady-state is what repeated scoring/training uses.")

    for batch_size in args.batch_sizes:
        timing = benchmark_inference(
            vocab_size=args.vocab_size,
            batch_size=batch_size,
            iterations=args.inference_iterations,
        )
        print_result(
            f"Inference graph, batch={batch_size}",
            timing,
            args.inference_iterations,
        )

    training_timing = benchmark_training(
        vocab_size=args.vocab_size,
        batch_size=max(args.batch_sizes),
        iterations=args.training_iterations,
    )
    print_result(
        f"Training step, batch={max(args.batch_sizes)}",
        training_timing,
        args.training_iterations,
    )


if __name__ == "__main__":
    main()
