import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(784, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 10),
        ]

    def __call__(self, x):
        x = mx.maximum(self.layers[0](x), 0)
        x = mx.maximum(self.layers[1](x), 0)
        return self.layers[2](x)


def download_mnist(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    for name, url in MNIST_URLS.items():
        path = data_dir / Path(url).name
        files[name] = path

        if path.exists():
            continue

        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)

    return files


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")

        data = np.frombuffer(file.read(), dtype=np.uint8)

    images = data.reshape(count, rows * cols).astype(np.float32)
    return images / 255.0


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    return labels.astype(np.int32)


def batch_iter(images: np.ndarray, labels: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(len(images))

    for start in range(0, len(images), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield mx.array(images[batch_indices]), mx.array(labels[batch_indices])


def loss_fn(model: MLP, images: mx.array, labels: mx.array) -> mx.array:
    logits = model(images)
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def accuracy(model: MLP, images: np.ndarray, labels: np.ndarray, batch_size: int) -> float:
    correct = 0
    total = 0

    for start in range(0, len(images), batch_size):
        end = start + batch_size
        x = mx.array(images[start:end])
        y = mx.array(labels[start:end])
        predictions = mx.argmax(model(x), axis=1)
        correct += int(mx.sum(predictions == y).item())
        total += len(labels[start:end])

    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--model-path", type=Path, default=Path("models/mnist_mlp.safetensors"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit training rows for a faster test run.")
    args = parser.parse_args()

    files = download_mnist(args.data_dir)
    train_images = load_images(files["train_images"])
    train_labels = load_labels(files["train_labels"])
    test_images = load_images(files["test_images"])
    test_labels = load_labels(files["test_labels"])

    if args.limit is not None:
        train_images = train_images[: args.limit]
        train_labels = train_labels[: args.limit]

    rng = np.random.default_rng(args.seed)
    model = MLP()
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    mx.eval(model.parameters())

    print(f"Training rows: {len(train_images):,}")
    print(f"Test rows: {len(test_images):,}")

    for epoch in range(1, args.epochs + 1):
        losses = []

        for images, labels in batch_iter(train_images, train_labels, args.batch_size, rng):
            loss, grads = loss_and_grad(model, images, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss.item()))

        test_accuracy = accuracy(model, test_images, test_labels, args.batch_size)
        print(
            f"epoch={epoch} "
            f"loss={np.mean(losses):.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(args.model_path))
    print(f"Saved model weights to {args.model_path}")


if __name__ == "__main__":
    main()
