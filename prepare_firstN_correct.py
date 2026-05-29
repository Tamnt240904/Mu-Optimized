#!/usr/bin/env python3
"""Export first-N ImageNet validation images that ResNet50 classifies correctly."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/"
    "image-models/imagenet_class_index.json"
)


FIELDNAMES = [
    "rank",
    "correct_index",
    "source_order",
    "image_path",
    "copied_image_path",
    "image_name",
    "ground_truth_label_idx",
    "ground_truth_label_name",
    "predicted_label_idx",
    "predicted_label_name",
    "top1_score",
    "wnid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Take the first N ImageNet validation examples, keep those "
            "classified correctly by torchvision ResNet50, and export selected.csv."
        )
    )
    parser.add_argument("--dataset-id", default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--first-count", type=int, default=5000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/imagenet_first5000_correct"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--class-index-json",
        type=Path,
        default=Path("data/imagenet_class_index.json"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_class_index(path: Path) -> tuple[dict[str, int], dict[int, tuple[str, str]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(CLASS_INDEX_URL, timeout=60) as response:
            path.write_bytes(response.read())
    raw = json.loads(path.read_text(encoding="utf-8"))
    idx_to_meta = {
        int(index): (value[0], value[1].replace("_", " "))
        for index, value in raw.items()
    }
    wnid_to_idx = {wnid: index for index, (wnid, _) in idx_to_meta.items()}
    return wnid_to_idx, idx_to_meta


def label_to_imagenet_index(
    dataset: Any,
    label: int | str,
    wnid_to_idx: dict[str, int],
    idx_to_meta: dict[int, tuple[str, str]],
) -> tuple[int, str]:
    if isinstance(label, str):
        if label in wnid_to_idx:
            return wnid_to_idx[label], label
        if label.isdigit() and int(label) in idx_to_meta:
            idx = int(label)
            return idx, idx_to_meta[idx][0]

    feature = dataset.features.get("label") if getattr(dataset, "features", None) else None
    if feature is not None and hasattr(feature, "int2str"):
        label_name = feature.int2str(int(label))
        if label_name in wnid_to_idx:
            return wnid_to_idx[label_name], label_name
        if label_name.isdigit() and int(label_name) in idx_to_meta:
            idx = int(label_name)
            return idx, idx_to_meta[idx][0]

    if isinstance(label, int) and label in idx_to_meta:
        return label, idx_to_meta[label][0]

    raise ValueError(f"Could not map dataset label {label!r} to ImageNet class index")


def save_rgb_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)


def collect_first_n_candidates(
    dataset_id: str,
    split: str,
    first_count: int,
    candidate_dir: Path,
    wnid_to_idx: dict[str, int],
    idx_to_meta: dict[int, tuple[str, str]],
    resume: bool,
) -> list[dict[str, Any]]:
    if first_count <= 0:
        raise ValueError(f"--first-count must be positive, got {first_count}")

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    records: list[dict[str, Any]] = []
    progress = tqdm(total=first_count, desc="Download first-N candidates")

    for source_order, row in enumerate(dataset):
        if len(records) >= first_count:
            break

        label_idx, wnid = label_to_imagenet_index(
            dataset, row["label"], wnid_to_idx, idx_to_meta
        )
        image_name = f"candidate_{source_order:05d}_{wnid}_c{label_idx:04d}.JPEG"
        image_path = candidate_dir / image_name
        if not resume or not image_path.exists():
            save_rgb_jpeg(row["image"], image_path)

        records.append({
            "source_order": source_order,
            "candidate_image_path": image_path,
            "image_name": image_name,
            "ground_truth_label_idx": label_idx,
            "ground_truth_label_name": idx_to_meta[label_idx][1],
            "wnid": wnid,
        })
        progress.update(1)

    progress.close()
    if len(records) != first_count:
        raise RuntimeError(
            f"Dataset ended after {len(records)} candidates; expected {first_count}."
        )
    return records


def load_resnet50(device: torch.device):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms(), weights.meta["categories"]


@torch.no_grad()
def filter_correct_resnet50(
    records: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")

    model, transform, categories = load_resnet50(device)
    correct: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(records), batch_size), desc="Score ResNet50"):
        batch = records[start:start + batch_size]
        tensors = []
        for record in batch:
            with Image.open(record["candidate_image_path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        x = torch.stack(tensors).to(device)
        probs = F.softmax(model(x), dim=1)
        scores, preds = probs.max(dim=1)

        for record, pred, score in zip(batch, preds.tolist(), scores.tolist()):
            gt_idx = int(record["ground_truth_label_idx"])
            if int(pred) != gt_idx:
                continue
            enriched = dict(record)
            enriched["predicted_label_idx"] = int(pred)
            enriched["predicted_label_name"] = categories[int(pred)]
            enriched["top1_score"] = float(score)
            enriched["correct_index"] = len(correct)
            correct.append(enriched)

    return correct


def export_selected(correct: list[dict[str, Any]], output_root: Path) -> Path:
    selected_dir = output_root / "images_correct"
    selected_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "selected.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, record in enumerate(correct, start=1):
        source = Path(record["candidate_image_path"])
        copied = selected_dir / source.name
        if source.resolve() != copied.resolve():
            shutil.copy2(source, copied)
        rows.append({
            "rank": rank,
            "correct_index": int(record["correct_index"]),
            "source_order": int(record["source_order"]),
            "image_path": str(copied),
            "copied_image_path": str(copied),
            "image_name": copied.name,
            "ground_truth_label_idx": int(record["ground_truth_label_idx"]),
            "ground_truth_label_name": record["ground_truth_label_name"],
            "predicted_label_idx": int(record["predicted_label_idx"]),
            "predicted_label_name": record["predicted_label_name"],
            "top1_score": f"{float(record['top1_score']):.10f}",
            "wnid": record["wnid"],
        })

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    output_root = args.output_root
    candidate_dir = output_root / "candidates_firstN"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    wnid_to_idx, idx_to_meta = load_imagenet_class_index(args.class_index_json)
    print(f"Device: {device}")
    print(f"Candidates -> {candidate_dir}")
    print(f"Selected   -> {output_root / 'images_correct'}")

    records = collect_first_n_candidates(
        args.dataset_id, args.split, args.first_count, candidate_dir,
        wnid_to_idx, idx_to_meta, args.resume)
    correct = filter_correct_resnet50(records, args.batch_size, device)
    csv_path = export_selected(correct, output_root)

    print(f"Exported {len(correct)} correct images from first {len(records)} candidates.")
    print(f"CSV -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
