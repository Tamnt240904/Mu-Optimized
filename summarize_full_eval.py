#!/usr/bin/env python3
"""Summarize full evaluation JSON files across seeds and models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ["IG", "IDG-PDF", "μ-Optimized"]
SUMMARY_METRICS = [
    "Q",
    "CV2",
    "insertion_auc",
    "deletion_auc",
    "insdel",
    "mu_entropy",
    "mu_max",
    "weighted_residual_l1",
]
FIELDNAMES = [
    "seed",
    "model_name",
    "method",
    "Q_mean",
    "Q_std",
    "CV2_mean",
    "CV2_std",
    "insertion_auc_mean",
    "insertion_auc_std",
    "deletion_auc_mean",
    "deletion_auc_std",
    "insdel_mean",
    "insdel_std",
    "mu_entropy_mean",
    "mu_entropy_std",
    "mu_max_mean",
    "mu_max_std",
    "weighted_residual_l1_mean",
    "weighted_residual_l1_std",
    "num_images_success",
    "num_images_failed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full_eval JSON files.")
    parser.add_argument("json_files", nargs="*", default=None)
    parser.add_argument("--input-glob", default="results/full_eval_seed*_*.json")
    parser.add_argument("--output-csv", default="results/full_eval_summary.csv")
    parser.add_argument("--output-md", default="results/full_eval_summary.md")
    return parser.parse_args()


def finite_values(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            output.append(float(value))
    return output


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    vals = finite_values(values)
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / len(vals)
    return mean, math.sqrt(variance)


def collect_values(payload: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    values: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for image in payload.get("images", []):
        if not image.get("success"):
            continue
        for method, metrics in image.get("methods", {}).items():
            for key in ["Q", "CV2", "mu_entropy", "mu_max",
                        "weighted_residual_l1"]:
                values[method][key].append(metrics.get(key))
        for method, metrics in image.get("insertion_deletion", {}).get("methods", {}).items():
            for key in ["insertion_auc", "deletion_auc", "insdel"]:
                values[method][key].append(metrics.get(key))
    return values


def row_from_values(
    seed: int | str,
    model_name: str,
    method: str,
    values: dict[str, list[Any]],
    success: int,
    failed: int,
) -> dict[str, Any]:
    row = {
        "seed": seed,
        "model_name": model_name,
        "method": method,
        "num_images_success": success,
        "num_images_failed": failed,
    }
    for metric in SUMMARY_METRICS:
        mean, std = mean_std(values.get(metric, []))
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    return row


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = FIELDNAMES
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in args.json_files] if args.json_files else sorted(Path().glob(args.input_glob))
    rows: list[dict[str, Any]] = []
    overall: dict[tuple[str, str], dict[str, Any]] = {}

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        seed = cfg.get("seed")
        model_name = cfg.get("model_name")
        success = sum(1 for image in payload.get("images", []) if image.get("success"))
        failed = sum(1 for image in payload.get("images", []) if not image.get("success"))
        values = collect_values(payload)

        for method in METHODS:
            row = row_from_values(seed, model_name, method, values[method], success, failed)
            rows.append(row)
            key = (model_name, method)
            state = overall.setdefault(key, {
                "values": defaultdict(list),
                "success": 0,
                "failed": 0,
            })
            state["success"] += success
            state["failed"] += failed
            for metric, vals in values[method].items():
                state["values"][metric].extend(vals)

    for (model_name, method), state in sorted(overall.items()):
        rows.append(row_from_values(
            "overall", model_name, method, state["values"],
            state["success"], state["failed"]))

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_md)
    print(f"Read {len(paths)} JSON files")
    print(f"CSV -> {output_csv}")
    print(f"MD  -> {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
