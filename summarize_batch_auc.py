#!/usr/bin/env python3
"""Summarize batch insertion/deletion JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_NAMES = ["IG", "IDG-PDF", "μ-Optimized"]
SUMMARY_KEYS = [
    "Q",
    "CV2",
    "insertion_auc",
    "deletion_auc",
    "insdel",
    "mu_entropy",
    "mu_max",
    "weighted_residual_l1",
]
CSV_COLUMNS = ["method"] + [
    f"{key}_{suffix}" for key in SUMMARY_KEYS for suffix in ("mean", "std")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV and Markdown summaries from batch AUC JSON."
    )
    parser.add_argument("batch_json", help="Batch JSON produced by batch_eval.py")
    parser.add_argument("--output-csv", default="results/batch_auc_summary.csv")
    parser.add_argument("--output-md", default="results/batch_auc_summary.md")
    return parser.parse_args()


def load_batch(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    if not isinstance(data.get("images"), list):
        raise ValueError(f"{path}: expected top-level images list")
    return data


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    vals = [float(value) for value in values if finite(value)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / len(vals)
    return mean, math.sqrt(variance)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def sanity_check_image(image: dict[str, Any], insdel_steps: int | None) -> None:
    if not image.get("success", True):
        return
    image_name = image.get("image_name", "<unknown>")
    methods = image.get("methods", {})
    if set(methods) != set(METHOD_NAMES):
        warn(f"{image_name}: expected all three methods, got {sorted(methods)}")

    for method_name, metrics in methods.items():
        q = metrics.get("Q")
        if not finite(q) or not (-1e-7 <= q <= 1.0 + 1e-7):
            warn(f"{image_name} {method_name}: Q outside [0, 1]")
        mu_sum = metrics.get("mu_sum")
        if not finite(mu_sum) or abs(float(mu_sum) - 1.0) >= 1e-5:
            warn(f"{image_name} {method_name}: mu_sum not close to 1")

    insertion_deletion = image.get("insertion_deletion")
    if insertion_deletion is None:
        return
    expected_len = None if insdel_steps is None else insdel_steps + 1
    for method_name, metrics in insertion_deletion.get("methods", {}).items():
        insertion_curve = metrics.get("insertion_curve")
        deletion_curve = metrics.get("deletion_curve")
        if expected_len is not None:
            if not isinstance(insertion_curve, list) or len(insertion_curve) != expected_len:
                warn(f"{image_name} {method_name}: invalid insertion_curve length")
            if not isinstance(deletion_curve, list) or len(deletion_curve) != expected_len:
                warn(f"{image_name} {method_name}: invalid deletion_curve length")
        if not finite(metrics.get("insertion_auc")):
            warn(f"{image_name} {method_name}: insertion_auc is not finite")
        if not finite(metrics.get("deletion_auc")):
            warn(f"{image_name} {method_name}: deletion_auc is not finite")


def collect_values(data: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    config = data.get("config", {})
    insdel_steps = config.get("insdel_steps") if isinstance(config, dict) else None
    if not isinstance(insdel_steps, int):
        insdel_steps = None

    values: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for image in data["images"]:
        if not isinstance(image, dict):
            continue
        sanity_check_image(image, insdel_steps)
        if not image.get("success", True):
            continue

        for method_name, metrics in image.get("methods", {}).items():
            for key in ["Q", "CV2", "mu_entropy", "mu_max", "weighted_residual_l1"]:
                values[method_name][key].append(metrics.get(key))

        insertion_deletion = image.get("insertion_deletion", {})
        for method_name, metrics in insertion_deletion.get("methods", {}).items():
            for key in ["insertion_auc", "deletion_auc", "insdel"]:
                values[method_name][key].append(metrics.get(key))
    return values


def build_rows(values: dict[str, dict[str, list[Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in METHOD_NAMES:
        row: dict[str, Any] = {"method": method_name}
        for key in SUMMARY_KEYS:
            mean, std = mean_std(values[method_name][key])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: "" if row.get(column) is None else row.get(column)
                for column in CSV_COLUMNS
            })


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def write_markdown(rows: list[dict[str, Any]], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch AUC Summary",
        "",
        "| " + " | ".join(CSV_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(CSV_COLUMNS)) + " |",
    ]
    for row in rows:
        values = [str(row["method"])] + [fmt(row.get(column)) for column in CSV_COLUMNS[1:]]
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    data = load_batch(Path(args.batch_json))
    rows = build_rows(collect_values(data))
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
