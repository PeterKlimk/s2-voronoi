#!/usr/bin/env python3
"""Summarize raw competitor CSV without discarding individual measurements."""

import argparse
import csv
from collections import defaultdict

import numpy as np


def interval(values: np.ndarray, statistic, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(20_000, len(values)), replace=True)
    estimates = statistic(draws, axis=1)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--metric", choices=("construct_ms", "total_ms"), default="construct_ms")
    parser.add_argument("--baseline", default="s2-voronoi")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    groups = defaultdict(list)
    by_round = {}
    for row in rows:
        value = float(row[args.metric])
        key = (row["dist"], row["size"], row["threads"], row["backend"])
        groups[key].append(value)
        by_round[(row["dist"], row["size"], row["round"], row["iteration"], row["backend"])] = value

    rng = np.random.default_rng(0x5EED)
    print(f"metric={args.metric}")
    print("dist,size,threads,backend,n,median_ms,ci95_low_ms,ci95_high_ms")
    for key, samples in sorted(groups.items()):
        values = np.asarray(samples)
        low, high = interval(values, np.median, rng)
        print(f"{','.join(key)},{len(values)},{np.median(values):.6f},{low:.6f},{high:.6f}")

    ratios = defaultdict(list)
    for (dist, size, round_number, iteration, backend), value in by_round.items():
        if backend == args.baseline:
            continue
        baseline = by_round.get((dist, size, round_number, iteration, args.baseline))
        if baseline is not None:
            ratios[(dist, size, backend)].append(value / baseline)

    print(f"\npaired_ratio=backend/{args.baseline}")
    print("dist,size,backend,n,median_ratio,geomean_ratio,geomean_ci95_low,geomean_ci95_high")
    for key, samples in sorted(ratios.items()):
        values = np.asarray(samples)
        logs = np.log(values)
        low, high = interval(logs, np.mean, rng)
        print(
            f"{','.join(key)},{len(values)},{np.median(values):.6f},"
            f"{np.exp(np.mean(logs)):.6f},{np.exp(low):.6f},{np.exp(high):.6f}"
        )


if __name__ == "__main__":
    main()
