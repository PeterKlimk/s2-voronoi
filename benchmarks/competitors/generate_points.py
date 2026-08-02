#!/usr/bin/env python3
"""Generate shared packed-f32 sphere inputs outside timed benchmark regions."""

import argparse
from pathlib import Path

import numpy as np


def parse_count(text: str) -> int:
    suffix = text[-1:].lower()
    scale = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
    number = text[:-1] if scale != 1 else text
    return int(float(number) * scale)


def fibonacci(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    y = 1.0 - (2.0 * i + 1.0) / n
    radius = np.sqrt(1.0 - y * y)
    theta = np.float64(2.0 * np.pi) * i / np.float64(1.618033988749895)
    return np.column_stack((radius * np.cos(theta), y, radius * np.sin(theta)))


def uniform(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(theta), z, radius * np.sin(theta)))


def repair_duplicate_f32_points(points: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    """Deterministically resample sites that collide after f32 conversion."""
    points = np.ascontiguousarray(points, dtype="<f4")
    repaired = 0
    rng = np.random.default_rng(seed ^ 0x5_3256_4F52_4F4E_4F49)

    while True:
        packed = points.view(np.dtype((np.void, 12))).ravel()
        _, first = np.unique(packed, return_index=True)
        duplicate = np.ones(len(points), dtype=bool)
        duplicate[first] = False
        indices = np.flatnonzero(duplicate)
        if len(indices) == 0:
            return points, repaired

        z = rng.uniform(-1.0, 1.0, len(indices))
        theta = rng.uniform(0.0, 2.0 * np.pi, len(indices))
        radius = np.sqrt(1.0 - z * z)
        points[indices] = np.column_stack(
            (radius * np.cos(theta), z, radius * np.sin(theta))
        ).astype("<f4")
        repaired += len(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("count", type=parse_count)
    parser.add_argument("--dist", choices=("fib", "uniform"), default="fib")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    points = fibonacci(args.count) if args.dist == "fib" else uniform(args.count, args.seed)
    points, repaired = repair_duplicate_f32_points(points, args.seed)
    lengths = np.linalg.norm(points.astype(np.float64), axis=1)
    if not np.all(np.isfinite(points)) or np.max(np.abs(lengths - 1.0)) > 1e-6:
        raise RuntimeError("generated points are not finite unit vectors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    points.tofile(args.output)
    print(
        f"wrote {args.count} {args.dist} points to {args.output} "
        f"({points.nbytes} bytes, repaired_duplicates={repaired})"
    )


if __name__ == "__main__":
    main()
