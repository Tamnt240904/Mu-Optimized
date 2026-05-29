#!/usr/bin/env python3
"""Select images where Mu-Optimized beats IG and IDG-PDF on two models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = ("resnet50", "vgg16")
METHODS = ("IG", "IDG-PDF", "Mu")
METRICS = ("insertion_auc", "deletion_auc", "insdel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge flattened ResNet50/VGG16 insertion-deletion CSVs and select "
            "images where Mu-Optimized outperforms IG and IDG-PDF."
        )
    )
    parser.add_argument("--resnet-csv", type=Path, required=True)
    parser.add_argument("--vgg-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_method(name: str) -> str | None:
    normalized = name.strip().lower().replace("μ", "mu")
    compact = normalized.replace("_", "").replace(" ", "").replace("-", "")
    if "idg" in compact:
        return "IDG-PDF"
    if "mu" in compact and "optimized" in compact:
        return "Mu"
    if compact == "ig":
        return "IG"
    return None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def sort_key(row: dict[str, Any]) -> tuple[int, str]:
    try:
        return int(row["correct_index"]), row["image_name"]
    except (TypeError, ValueError):
        return 10**12, row["image_name"]


def read_model_csv(path: Path, model_prefix: str) -> dict[tuple[str, str], dict[str, Any]]:
    by_image: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            correct_index = row.get("correct_index")
            image_name = row.get("image_name")
            method = normalize_method(row.get("method", ""))
            if correct_index in (None, "") or not image_name or method is None:
                continue

            key = (str(correct_index), image_name)
            output = by_image.setdefault(key, {
                "correct_index": correct_index,
                "image_name": image_name,
            })
            for metric in METRICS:
                output[f"{model_prefix}_{method}_{metric}"] = parse_float(row.get(metric))

    return by_image


def metric_columns() -> list[str]:
    return [
        f"{model}_{method}_{metric}"
        for model in MODELS
        for method in METHODS
        for metric in METRICS
    ]


def merge_tables(
    resnet: dict[tuple[str, str], dict[str, Any]],
    vgg: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(resnet) & set(vgg)):
        merged = {
            "correct_index": resnet[key]["correct_index"],
            "image_name": resnet[key]["image_name"],
        }
        for column in metric_columns():
            merged[column] = None
        merged.update({k: v for k, v in resnet[key].items() if k not in {"correct_index", "image_name"}})
        merged.update({k: v for k, v in vgg[key].items() if k not in {"correct_index", "image_name"}})
        rows.append(merged)
    rows.sort(key=sort_key)
    return rows


def value(row: dict[str, Any], model: str, method: str, metric: str) -> float | None:
    item = row.get(f"{model}_{method}_{metric}")
    if item is None:
        return None
    return float(item)


def list1_model_margin(row: dict[str, Any], model: str) -> float | None:
    mu = value(row, model, "Mu", "insdel")
    ig = value(row, model, "IG", "insdel")
    idg = value(row, model, "IDG-PDF", "insdel")
    if mu is None or ig is None or idg is None:
        return None
    return mu - max(ig, idg)


def list2_model_margins(row: dict[str, Any], model: str) -> dict[str, float] | None:
    insertion_mu = value(row, model, "Mu", "insertion_auc")
    insertion_ig = value(row, model, "IG", "insertion_auc")
    insertion_idg = value(row, model, "IDG-PDF", "insertion_auc")
    deletion_mu = value(row, model, "Mu", "deletion_auc")
    deletion_ig = value(row, model, "IG", "deletion_auc")
    deletion_idg = value(row, model, "IDG-PDF", "deletion_auc")
    insdel_margin = list1_model_margin(row, model)
    values = (
        insertion_mu, insertion_ig, insertion_idg,
        deletion_mu, deletion_ig, deletion_idg, insdel_margin,
    )
    if any(item is None for item in values):
        return None
    return {
        "insertion_margin": insertion_mu - max(insertion_ig, insertion_idg),
        "deletion_margin": min(deletion_ig, deletion_idg) - deletion_mu,
        "insdel_margin": insdel_margin,
    }


def select_list1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        margins = {model: list1_model_margin(row, model) for model in MODELS}
        if any(margin is None or margin <= 0 for margin in margins.values()):
            continue
        output = dict(row)
        output["list1_resnet50_margin"] = margins["resnet50"]
        output["list1_vgg16_margin"] = margins["vgg16"]
        output["list1_avg_margin"] = sum(margins.values()) / len(MODELS)
        selected.append(output)
    selected.sort(key=lambda row: row["list1_avg_margin"], reverse=True)
    return selected


def select_list2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        margins = {model: list2_model_margins(row, model) for model in MODELS}
        if any(model_margins is None for model_margins in margins.values()):
            continue
        if any(
            margin <= 0
            for model_margins in margins.values()
            for margin in model_margins.values()
        ):
            continue

        output = dict(row)
        all_margins = []
        for model, model_margins in margins.items():
            for metric_name, margin in model_margins.items():
                output[f"list2_{model}_{metric_name}"] = margin
                all_margins.append(margin)
        output["list2_avg_margin_score"] = sum(all_margins) / len(all_margins)
        selected.append(output)

    selected.sort(key=lambda row: row["list2_avg_margin_score"], reverse=True)
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], extra_columns: list[str] | None = None) -> None:
    fieldnames = ["correct_index", "image_name"] + metric_columns()
    if extra_columns:
        fieldnames.extend(extra_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_indices(rows: list[dict[str, Any]]) -> list[int | str]:
    indices: list[int | str] = []
    for row in rows:
        index = row["correct_index"]
        try:
            indices.append(int(index))
        except (TypeError, ValueError):
            indices.append(index)
    return indices


def write_selection_json(path: Path, rows: list[dict[str, Any]], selection_rule: str) -> None:
    payload = {
        "index_type": "correct_index",
        "description": (
            "Indices are positions in the ResNet50-correct filtered set, "
            "ordered by original source_order."
        ),
        "selection_rule": selection_rule,
        "num_selected": len(rows),
        "indices": json_indices(rows),
        "image_names": [row["image_name"] for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resnet = read_model_csv(args.resnet_csv, "resnet50")
    vgg = read_model_csv(args.vgg_csv, "vgg16")
    merged = merge_tables(resnet, vgg)

    write_csv(args.output_dir / "merged_per_image_metrics.csv", merged)

    list1 = select_list1(merged)
    list1_rule = (
        "For both resnet50 and vgg16: Mu insdel > IG insdel and "
        "Mu insdel > IDG-PDF insdel."
    )
    write_csv(
        args.output_dir / "list1_mu_beats_ig_idg_insdel_both_models.csv",
        list1,
        ["list1_resnet50_margin", "list1_vgg16_margin", "list1_avg_margin"],
    )
    write_selection_json(
        args.output_dir / "list1_mu_beats_ig_idg_insdel_both_models.json",
        list1,
        list1_rule,
    )

    list2 = select_list2(merged)
    list2_rule = (
        "For both resnet50 and vgg16: Mu insertion_auc > IG and IDG-PDF, "
        "Mu deletion_auc < IG and IDG-PDF, and Mu insdel > IG and IDG-PDF."
    )
    list2_margin_columns = [
        f"list2_{model}_{metric}_margin"
        for model in MODELS
        for metric in ("insertion", "deletion", "insdel")
    ]
    write_csv(
        args.output_dir / "list2_mu_beats_ig_idg_all_metrics_both_models.csv",
        list2,
        list2_margin_columns + ["list2_avg_margin_score"],
    )
    write_selection_json(
        args.output_dir / "list2_mu_beats_ig_idg_all_metrics_both_models.json",
        list2,
        list2_rule,
    )

    print(f"Merged images: {len(merged)}")
    print(f"List 1 selected: {len(list1)}")
    print(f"List 2 selected: {len(list2)}")
    print(f"Outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
