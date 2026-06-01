#!/usr/bin/env python3
"""Build progressive selected CSVs from batch_eval insertion/deletion JSON."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROGRESSIVE_FILTER_RULES = ("mu_gt_idg_insdel", "mu_gt_ig_idg_insdel")


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


def normalize_method_name(name: Any) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("μ", "mu")
    compact = normalized.replace("_", "").replace(" ", "").replace("-", "")
    if compact == "ig":
        return "IG"
    if "idg" in compact:
        return "IDG-PDF"
    if compact == "mu" or ("mu" in compact and "optimized" in compact):
        return "Mu"
    return None


def canonical_key(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
    return str(value)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def method_metrics(image: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    insertion_deletion = image.get("insertion_deletion")
    if not isinstance(insertion_deletion, dict):
        return {}
    methods = insertion_deletion.get("methods")
    if not isinstance(methods, dict):
        return {}

    normalized: dict[str, dict[str, float | None]] = {}
    for raw_name, raw_metrics in methods.items():
        method = normalize_method_name(raw_name)
        if method is None or not isinstance(raw_metrics, dict):
            continue
        normalized[method] = {
            "insertion_auc": parse_float(raw_metrics.get("insertion_auc")),
            "deletion_auc": parse_float(raw_metrics.get("deletion_auc")),
            "insdel": parse_float(raw_metrics.get("insdel")),
        }
    return normalized


def passes_rule(metrics: dict[str, dict[str, float | None]], rule: str) -> bool:
    if rule not in PROGRESSIVE_FILTER_RULES:
        raise ValueError(
            f"Unknown progressive filter rule: {rule}. "
            f"Expected one of: {', '.join(PROGRESSIVE_FILTER_RULES)}"
        )

    mu = metrics.get("Mu", {}).get("insdel")
    idg = metrics.get("IDG-PDF", {}).get("insdel")
    ig = metrics.get("IG", {}).get("insdel")
    if mu is None or idg is None:
        return False
    if rule == "mu_gt_idg_insdel":
        return mu > idg
    if ig is None:
        return False
    return mu > ig and mu > idg


def build_progressive_filtered_csv(
    original_selected_csv: Path,
    batch_json: Path,
    output_csv: Path,
    rule: str,
) -> int:
    try:
        return _build_progressive_filtered_csv_with_pandas(
            original_selected_csv, batch_json, output_csv, rule
        )
    except ModuleNotFoundError as exc:
        if exc.name != "pandas":
            raise
        return _build_progressive_filtered_csv_with_csv(
            original_selected_csv, batch_json, output_csv, rule
        )


def passing_identities(batch_json: Path, rule: str) -> tuple[set[str], set[str]]:
    passing_correct_indices: set[str] = set()
    passing_image_names: set[str] = set()

    for image in load_image_results(batch_json):
        if not isinstance(image, dict) or image.get("success") is False:
            continue
        if not passes_rule(method_metrics(image), rule):
            continue

        correct_index = canonical_key(image.get("correct_index"))
        if correct_index is not None:
            passing_correct_indices.add(correct_index)
            continue

        image_name = canonical_key(image.get("image_name"))
        if image_name is not None:
            passing_image_names.add(image_name)

    return passing_correct_indices, passing_image_names


def _build_progressive_filtered_csv_with_pandas(
    original_selected_csv: Path,
    batch_json: Path,
    output_csv: Path,
    rule: str,
) -> int:
    import pandas as pd

    selected = pd.read_csv(original_selected_csv, dtype=str, keep_default_na=False)
    passing_correct_indices, passing_image_names = passing_identities(batch_json, rule)

    keep = pd.Series(False, index=selected.index)
    if passing_correct_indices and "correct_index" in selected.columns:
        keep |= selected["correct_index"].map(canonical_key).isin(passing_correct_indices)
    if passing_image_names and "image_name" in selected.columns:
        keep |= selected["image_name"].map(canonical_key).isin(passing_image_names)

    filtered = selected.loc[keep]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    return len(filtered)


def _build_progressive_filtered_csv_with_csv(
    original_selected_csv: Path,
    batch_json: Path,
    output_csv: Path,
    rule: str,
) -> int:
    passing_correct_indices, passing_image_names = passing_identities(batch_json, rule)
    with original_selected_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"selected CSV has no header: {original_selected_csv}")
        rows = []
        for row in reader:
            keep = False
            if "correct_index" in row:
                correct_index = canonical_key(row.get("correct_index"))
                keep = correct_index in passing_correct_indices
            if not keep and "image_name" in row:
                image_name = canonical_key(row.get("image_name"))
                keep = image_name in passing_image_names
            if keep:
                rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
