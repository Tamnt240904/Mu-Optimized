#!/usr/bin/env python3
"""Summarize and plot generated sweep JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_COLUMNS = [
    "source_file",
    "sweep_type",
    "N",
    "tau",
    "iters",
    "lr",
    "method",
    "Q",
    "CV2",
    "Var_nu",
    "mu_l2_sq",
    "mu_entropy",
    "mu_max",
    "mu_min",
    "mu_active_count",
    "mu_sum",
    "weighted_residual_l1",
    "weighted_residual_l2",
    "elapsed_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize sweep JSON outputs and create diagnostic plots."
    )
    parser.add_argument("json_files", nargs="+", help="Sweep JSON files to load")
    parser.add_argument("--output-csv", default="results/summary.csv")
    parser.add_argument("--plot-dir", default="results/plots")
    parser.add_argument("--representative-n", type=int, default=16)
    parser.add_argument("--representative-tau", type=float, default=0.01)
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError(f"{path}: expected top-level 'runs' list")
    return data


def infer_sweep_type(path: Path, runs: list[dict[str, Any]]) -> str:
    name = path.name.lower()
    if "tau_sweep" in name:
        return "tau_sweep"
    if "steps_sweep" in name or "step_sweep" in name:
        return "steps_sweep"

    ns = {
        to_float(run.get("config", {}).get("N"))
        for run in runs
        if isinstance(run, dict)
    }
    taus = {
        to_float(run.get("config", {}).get("tau"))
        for run in runs
        if isinstance(run, dict)
    }
    ns.discard(None)
    taus.discard(None)

    if len(taus) > 1 and len(ns) <= 1:
        return "tau_sweep"
    if len(ns) > 1 and len(taus) <= 1:
        return "steps_sweep"
    if len(ns) > 1 and len(taus) > 1:
        return "grid_sweep"
    return "single_run"


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def check_row(row: dict[str, Any], source: Path, run_idx: int, method: str) -> int:
    failed = 0

    def fail(check_name: str, value: Any) -> None:
        nonlocal failed
        failed += 1
        warn(
            f"{source} run={run_idx} method={method}: "
            f"{check_name} failed (value={value!r})"
        )

    mu_sum = to_float(row["mu_sum"])
    if mu_sum is None or abs(mu_sum - 1.0) >= 1e-5:
        fail("abs(mu_sum - 1.0) < 1e-5", row["mu_sum"])

    mu_min = to_float(row["mu_min"])
    if mu_min is None or mu_min < -1e-8:
        fail("mu_min >= -1e-8", row["mu_min"])

    q = to_float(row["Q"])
    if q is None or not (0.0 <= q <= 1.0):
        fail("0 <= Q <= 1", row["Q"])

    cv2 = to_float(row["CV2"])
    if cv2 is None or cv2 < 0.0:
        fail("CV2 >= 0", row["CV2"])

    n = to_float(row["N"])
    mu_l2_sq = to_float(row["mu_l2_sq"])
    if n is None or n <= 0 or mu_l2_sq is None:
        fail("1/N - 1e-6 <= mu_l2_sq <= 1 + 1e-6", row["mu_l2_sq"])
    else:
        lower = 1.0 / n - 1e-6
        upper = 1.0 + 1e-6
        if not (lower <= mu_l2_sq <= upper):
            fail("1/N - 1e-6 <= mu_l2_sq <= 1 + 1e-6", row["mu_l2_sq"])

    return failed


def extract_rows(
    path: Path, data: dict[str, Any], sweep_type: str
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0

    for run_idx, run in enumerate(data["runs"]):
        if not isinstance(run, dict):
            warn(f"{path} run={run_idx}: expected run object")
            continue

        config = run.get("config", {})
        if not isinstance(config, dict):
            config = {}
        model_info = run.get("model_info", {})
        if not isinstance(model_info, dict):
            model_info = {}
        methods = run.get("methods", {})
        if not isinstance(methods, dict):
            warn(f"{path} run={run_idx}: expected methods object")
            continue

        for method_key, method_data in methods.items():
            if not isinstance(method_data, dict):
                warn(f"{path} run={run_idx} method={method_key}: expected object")
                continue

            method_name = str(method_data.get("name", method_key))
            row = {
                "source_file": str(path),
                "sweep_type": sweep_type,
                "N": first_present(config.get("N"), model_info.get("N")),
                "tau": first_present(config.get("tau"), method_data.get("tau")),
                "iters": first_present(config.get("iters"), method_data.get("n_iters")),
                "lr": first_present(config.get("lr"), method_data.get("learning_rate")),
                "method": method_name,
                "Q": method_data.get("Q"),
                "CV2": method_data.get("CV2"),
                "Var_nu": method_data.get("Var_nu"),
                "mu_l2_sq": method_data.get("mu_l2_sq"),
                "mu_entropy": method_data.get("mu_entropy"),
                "mu_max": method_data.get("mu_max"),
                "mu_min": method_data.get("mu_min"),
                "mu_active_count": method_data.get("mu_active_count"),
                "mu_sum": method_data.get("mu_sum"),
                "weighted_residual_l1": method_data.get("weighted_residual_l1"),
                "weighted_residual_l2": method_data.get("weighted_residual_l2"),
                "elapsed_s": method_data.get("elapsed_s"),
            }
            failures += check_row(row, path, run_idx, method_name)
            rows.append(row)

    return rows, failures


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in CSV_COLUMNS})


def row_float(row: dict[str, Any], key: str) -> float | None:
    return to_float(row.get(key))


def plot_metric(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    x_log: bool = False,
) -> bool:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = row_float(row, x_key)
        y = row_float(row, y_key)
        if x is None or y is None:
            continue
        series[str(row["method"])].append((x, y))

    if not series:
        warn(f"no plottable values for {y_key} vs {x_key} at {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method, points in sorted(series.items()):
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=method)

    if x_log and all(x > 0 for points in series.values() for x, _ in points):
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def make_sweep_plots(file_rows: list[dict[str, Any]], source: Path, plot_dir: Path) -> list[Path]:
    made: list[Path] = []
    if not file_rows:
        return made

    sweep_type = str(file_rows[0]["sweep_type"])
    stem = safe_stem(source)

    if sweep_type == "tau_sweep":
        specs = [
            ("Q", "Q", "Q vs tau", f"{stem}_Q_vs_tau.png"),
            ("CV2", "CV2", "CV2 vs tau", f"{stem}_CV2_vs_tau.png"),
            ("mu_l2_sq", "mu_l2_sq", "mu_l2_sq vs tau", f"{stem}_mu_l2_sq_vs_tau.png"),
        ]
        for metric, ylabel, title, filename in specs:
            output = plot_dir / filename
            if plot_metric(
                file_rows,
                "tau",
                metric,
                f"{title} ({source.name})",
                "tau",
                ylabel,
                output,
                x_log=True,
            ):
                made.append(output)

    if sweep_type == "steps_sweep":
        specs = [
            ("Q", "Q", "Q vs N", f"{stem}_Q_vs_N.png"),
            (
                "weighted_residual_l1",
                "weighted_residual_l1",
                "weighted_residual_l1 vs N",
                f"{stem}_weighted_residual_l1_vs_N.png",
            ),
        ]
        for metric, ylabel, title, filename in specs:
            output = plot_dir / filename
            if plot_metric(
                file_rows,
                "N",
                metric,
                f"{title} ({source.name})",
                "N",
                ylabel,
                output,
            ):
                made.append(output)

    return made


def floats_close(left: Any, right: float) -> bool:
    left_float = to_float(left)
    return left_float is not None and math.isclose(
        left_float, right, rel_tol=1e-9, abs_tol=1e-12
    )


def find_representative_run(
    files: list[tuple[Path, dict[str, Any], str]],
    representative_n: int,
    representative_tau: float,
) -> tuple[Path, dict[str, Any]] | None:
    for source, data, _sweep_type in files:
        for run in data["runs"]:
            if not isinstance(run, dict):
                continue
            config = run.get("config", {})
            if not isinstance(config, dict):
                continue
            if to_int(config.get("N")) == representative_n and floats_close(
                config.get("tau"), representative_tau
            ):
                return source, run
    return None


def make_mu_distribution_plot(
    files: list[tuple[Path, dict[str, Any], str]],
    plot_dir: Path,
    representative_n: int,
    representative_tau: float,
) -> Path | None:
    match = find_representative_run(files, representative_n, representative_tau)
    if match is None:
        warn(
            "no representative run found for "
            f"N={representative_n}, tau={representative_tau}"
        )
        return None

    source, run = match
    methods = run.get("methods", {})
    if not isinstance(methods, dict):
        warn(f"{source}: representative run has no methods object")
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = False
    for method_key, method_data in sorted(methods.items()):
        if not isinstance(method_data, dict):
            continue
        steps = method_data.get("steps", [])
        if not isinstance(steps, list):
            continue
        xs: list[float] = []
        ys: list[float] = []
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            mu = to_float(step.get("mu_k"))
            if mu is None:
                continue
            t = to_float(step.get("t"))
            xs.append(t if t is not None else float(idx))
            ys.append(mu)
        if not xs:
            continue
        label = str(method_data.get("name", method_key))
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        warn(f"{source}: representative run has no plottable mu_k values")
        return None

    output = plot_dir / (
        f"representative_mu_distribution_N{representative_n}"
        f"_tau{representative_tau:g}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ax.set_title(
        f"mu distribution by step ({source.name}, N={representative_n}, "
        f"tau={representative_tau:g})"
    )
    ax.set_xlabel("step t")
    ax.set_ylabel("mu_k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    output_csv = Path(args.output_csv)
    plot_dir = Path(args.plot_dir)

    loaded_files: list[tuple[Path, dict[str, Any], str]] = []
    all_rows: list[dict[str, Any]] = []
    total_failures = 0

    for filename in args.json_files:
        path = Path(filename)
        data = load_json(path)
        sweep_type = infer_sweep_type(path, data["runs"])
        loaded_files.append((path, data, sweep_type))
        rows, failures = extract_rows(path, data, sweep_type)
        all_rows.extend(rows)
        total_failures += failures

    write_csv(all_rows, output_csv)

    plot_paths: list[Path] = []
    for source, _data, _sweep_type in loaded_files:
        file_rows = [row for row in all_rows if row["source_file"] == str(source)]
        plot_paths.extend(make_sweep_plots(file_rows, source, plot_dir))

    mu_plot = make_mu_distribution_plot(
        loaded_files, plot_dir, args.representative_n, args.representative_tau
    )
    if mu_plot is not None:
        plot_paths.append(mu_plot)

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote {len(plot_paths)} plot(s) to: {plot_dir}")
    if total_failures == 0:
        print("Sanity checks passed.")
    else:
        print(f"Sanity check warnings: {total_failures}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
