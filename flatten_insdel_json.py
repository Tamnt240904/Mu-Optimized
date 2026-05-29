#!/usr/bin/env python3
"""Flatten batch_eval insertion/deletion JSON into one CSV row per image-method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "correct_index",
    "rank",
    "image_name",
    "image_path",
    "target_class",
    "target_label_name",
    "method",
    "insertion_auc",
    "deletion_auc",
    "insdel",
    "auc_score",
    "perturb_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten batch_eval.py insertion/deletion JSON to CSV."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_image_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported JSON payload in {path}: expected object or list")
    for key in ("images", "results", "image_results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(
        f"Could not find image results in {path}; expected one of images, results, image_results"
    )


def flatten(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in images:
        if image.get("success") is False:
            continue
        insertion_deletion = image.get("insertion_deletion")
        if not isinstance(insertion_deletion, dict):
            continue
        methods = insertion_deletion.get("methods")
        if not isinstance(methods, dict):
            continue

        for method_name, metrics in methods.items():
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "correct_index": image.get("correct_index"),
                "rank": image.get("rank"),
                "image_name": image.get("image_name"),
                "image_path": image.get("image_path"),
                "target_class": image.get("target_class"),
                "target_label_name": image.get("target_label_name"),
                "method": method_name,
                "insertion_auc": metrics.get("insertion_auc"),
                "deletion_auc": metrics.get("deletion_auc"),
                "insdel": metrics.get("insdel"),
                "auc_score": metrics.get("auc_score", insertion_deletion.get("auc_score")),
                "perturb_steps": metrics.get(
                    "perturb_steps", insertion_deletion.get("perturb_steps")
                ),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = flatten(load_image_results(args.input_json))
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
