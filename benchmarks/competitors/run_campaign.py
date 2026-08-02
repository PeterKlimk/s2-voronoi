#!/usr/bin/env python3
"""Run rotated, affinity-controlled spherical Voronoi comparison rounds."""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_RE = re.compile(r"(?:^|\s)([a-z_]+)=([^\s]+)")
PERF_EVENTS = "cycles,instructions,cache-references,cache-misses,page-faults,context-switches,cpu-migrations"


def parse_count(text: str) -> int:
    suffix = text[-1:].lower()
    scale = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
    return int(float(text[:-1] if scale != 1 else text) * scale)


def commands(
    data: Path,
    repeat: int,
    threads: int,
    cpus: str,
    qhull_bin: Path,
    stripack_bin: Path,
) -> dict[str, list[str]]:
    prefix = ["taskset", "-c", cpus]
    return {
        "s2-voronoi": prefix
        + [
            "env",
            f"RAYON_NUM_THREADS={threads}",
            str(ROOT / "target/competitors/rust/release/bench_compare"),
            str(data),
            "--repeat",
            str(repeat),
        ],
        "cgal": prefix
        + [
            str(ROOT / "target/competitors/build/bench_cgal_sphere"),
            str(data),
            "--repeat",
            str(repeat),
        ],
        "qhull": prefix
        + [
            str(qhull_bin),
            str(data),
            "--repeat",
            str(repeat),
        ],
        "stripack": prefix
        + [
            str(stripack_bin),
            str(data),
            "--repeat",
            str(repeat),
        ],
        "stripack-construct": prefix
        + [
            str(stripack_bin),
            str(data),
            "--construct-only",
            "--repeat",
            str(repeat),
        ],
    }


def generate(data: Path, size: str, dist: str, seed: int) -> None:
    expected_size = parse_count(size) * 12
    if data.exists() and data.stat().st_size == expected_size:
        points = np.memmap(data, dtype="<f4", mode="r").reshape(-1, 3)
        packed = np.ascontiguousarray(points).view(np.dtype((np.void, 12))).ravel()
        if len(np.unique(packed)) == len(points):
            return
        print(f"regenerating {data}: duplicate packed-f32 sites", flush=True)
    subprocess.run(
        [
            "python3",
            str(ROOT / "benchmarks/competitors/generate_points.py"),
            str(data),
            size,
            "--dist",
            dist,
            "--seed",
            str(seed),
        ],
        check=True,
    )


def parse_perf(path: Path) -> dict[str, str]:
    counters = {}
    with path.open() as stream:
        for row in csv.reader(stream):
            if len(row) >= 3 and row[0] not in ("<not counted>", "<not supported>"):
                counters[row[2]] = row[0]
    return counters


def run_one(command: list[str]) -> tuple[list[dict[str, str]], dict[str, str], str]:
    with tempfile.TemporaryDirectory(prefix="s2-competitor-", dir="/tmp") as temporary:
        temporary = Path(temporary)
        perf_path = temporary / "perf.csv"
        time_path = temporary / "time.txt"
        wrapped = [
            "perf",
            "stat",
            "--no-big-num",
            "-x,",
            "-o",
            str(perf_path),
            "-e",
            PERF_EVENTS,
            "--",
            "/usr/bin/time",
            "-f",
            "%M",
            "-o",
            str(time_path),
        ] + command
        completed = subprocess.run(wrapped, check=True, text=True, capture_output=True)
        results = []
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT "):
                results.append(dict(RESULT_RE.findall(line)))
        if not results:
            raise RuntimeError(f"benchmark emitted no RESULT line: {' '.join(command)}")
        counters = parse_perf(perf_path)
        counters["max_rss_kib"] = time_path.read_text().strip()
        return results, counters, completed.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["10k", "100k"])
    parser.add_argument("--dist", choices=("fib", "uniform"), default="fib")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--inner-repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", default="0")
    parser.add_argument("--backends", nargs="+",
                        choices=("s2-voronoi", "cgal", "qhull", "stripack",
                                 "stripack-construct"),
                        default=["s2-voronoi", "cgal", "qhull", "stripack-construct"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--qhull-bin",
        type=Path,
        default=ROOT / "target/competitors/build/bench_qhull_sphere",
    )
    parser.add_argument(
        "--stripack-bin",
        type=Path,
        default=ROOT / "target/competitors/stripack/release/bench-stripack-sphere",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.inner_repeat < 1 or args.threads < 1:
        parser.error("rounds, inner-repeat, and threads must be positive")

    output = args.output or ROOT / "target/competitors/results" / (
        f"{args.dist}-t{args.threads}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dist", "size", "round", "order", "threads", "cpus", "backend", "n",
        "iteration", "construct_ms", "materialize_ms", "total_ms", "vertices", "cells",
        "incidences", "checksum", "cycles", "instructions", "cache-references",
        "cache-misses", "page-faults", "context-switches", "cpu-migrations", "max_rss_kib",
    ]

    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for size in args.sizes:
            data = ROOT / "target/competitors/data" / f"{args.dist}-{size}-seed{args.seed}.f32"
            generate(data, size, args.dist, args.seed)
            available = commands(
                data,
                args.inner_repeat,
                args.threads,
                args.cpus,
                args.qhull_bin.resolve(),
                args.stripack_bin.resolve(),
            )

            for backend in args.backends:
                subprocess.run(available[backend][:-2] + ["--repeat", "1"], check=True,
                               stdout=subprocess.DEVNULL)

            for round_number in range(1, args.rounds + 1):
                start = (round_number - 1) % len(args.backends)
                order = args.backends[start:] + args.backends[:start]
                for order_number, backend in enumerate(order, 1):
                    print(f"{size} round {round_number}/{args.rounds} order {order_number}: {backend}",
                          flush=True)
                    results, counters, stderr = run_one(available[backend])
                    if stderr:
                        print(stderr, file=os.sys.stderr)
                    for result in results:
                        row = {
                            "dist": args.dist,
                            "size": size,
                            "round": round_number,
                            "order": order_number,
                            "threads": args.threads if backend == "s2-voronoi" else 1,
                            "cpus": args.cpus,
                            **result,
                            **counters,
                        }
                        writer.writerow({field: row.get(field, "") for field in fields})
                        stream.flush()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
