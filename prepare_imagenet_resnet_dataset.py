#!/usr/bin/env python3
"""Build a ResNet-50-correct ImageNet subset from random validation images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import traceback
import urllib.request
from pathlib import Path

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
IMAGE_EXT = ".JPEG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download random ImageNet validation images, keep those correctly "
            "predicted by torchvision ResNet-50, and export a score-sorted CSV."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default="Tsomaros/Imagenet-1k_validation",
        help=(
            "Hugging Face dataset id. Use ILSVRC/imagenet-1k if your HF token "
            "has access to the gated official dataset."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--keep-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help=(
            "Streaming shuffle buffer. Larger values are more globally random "
            "but download more data before the first sample is yielded."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument(
        "--random-dir",
        type=Path,
        default=Path("generated_imagenet/imagenet_random_5000"),
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=Path("generated_imagenet/imagenet_resnet50_correct_1000"),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("generated_imagenet/imagenet_resnet50_correct_1000.csv"),
    )
    parser.add_argument(
        "--class-index-json",
        type=Path,
        default=Path("generated_imagenet/imagenet_class_index.json"),
        help="Local cache for WNID to ImageNet class index mapping.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing files in --random-dir when names match.",
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wnid_to_class_index(path: Path) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(CLASS_INDEX_URL, timeout=60) as response:
            path.write_bytes(response.read())
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return {value[0]: int(key) for key, value in raw.items()}


def label_to_wnid(dataset, label: int) -> str:
    feature = dataset.features.get("label") if dataset.features else None
    if feature is not None and hasattr(feature, "int2str"):
        return feature.int2str(int(label))
    if isinstance(label, str):
        return label
    raise ValueError(
        "Dataset label must be a WNID string or a ClassLabel with WNID names."
    )


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def save_rgb_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)


def download_random_images(
    dataset_id: str,
    split: str,
    sample_count: int,
    seed: int,
    shuffle_buffer: int,
    random_dir: Path,
    wnid_to_class_index: dict[str, int],
    resume: bool,
) -> list[dict[str, object]]:
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    shuffled = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    records: list[dict[str, object]] = []

    progress = tqdm(total=sample_count, desc="Downloading random images")
    for sample_id, row in enumerate(shuffled):
        if len(records) >= sample_count:
            break

        label = row["label"]
        wnid = label_to_wnid(dataset, label)
        if wnid not in wnid_to_class_index:
            raise KeyError(f"WNID {wnid!r} is not present in ImageNet class index")

        target_idx = wnid_to_class_index[wnid]
        filename = f"imagenet_random_{len(records):05d}_{wnid}_c{target_idx:04d}{IMAGE_EXT}"
        image_path = random_dir / filename
        if not resume or not image_path.exists():
            save_rgb_jpeg(row["image"], image_path)

        records.append(
            {
                "image_name": filename,
                "image_path": image_path,
                "source_order": sample_id,
                "wnid": wnid,
                "label_idx": target_idx,
            }
        )
        progress.update(1)

    progress.close()
    if len(records) < sample_count:
        raise RuntimeError(
            f"Dataset ended after {len(records)} usable images; expected {sample_count}."
        )
    return records


def load_model(device: torch.device):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms(), weights.meta["categories"]


@torch.no_grad()
def score_records(
    records: list[dict[str, object]],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    model, transform, categories = load_model(device)
    correct: list[dict[str, object]] = []

    for start in tqdm(range(0, len(records), batch_size), desc="Scoring with ResNet-50"):
        batch_records = records[start : start + batch_size]
        images = []
        for record in batch_records:
            with Image.open(record["image_path"]) as image:
                images.append(transform(image.convert("RGB")))
        x = torch.stack(images).to(device)
        probs = F.softmax(model(x), dim=1)
        scores, preds = probs.max(dim=1)

        for record, pred, score in zip(batch_records, preds.tolist(), scores.tolist()):
            label_idx = int(record["label_idx"])
            if pred == label_idx:
                enriched = dict(record)
                enriched["score"] = float(score)
                enriched["pred_idx"] = pred
                enriched["label_name"] = categories[label_idx]
                correct.append(enriched)

    correct.sort(key=lambda item: float(item["score"]), reverse=True)
    return correct


def export_selected(
    correct: list[dict[str, object]],
    keep_count: int,
    selected_dir: Path,
    csv_path: Path,
) -> None:
    if len(correct) < keep_count:
        raise RuntimeError(
            f"Only {len(correct)} images were predicted correctly; "
            f"cannot select {keep_count}."
        )

    selected_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected = correct[:keep_count]

    rows = []
    for rank, record in enumerate(selected, start=1):
        src = Path(record["image_path"])
        dst = selected_dir / src.name
        shutil.copy2(src, dst)
        rows.append(
            {
                "rank": rank,
                "image_name": dst.name,
                "score": f"{float(record['score']):.10f}",
                "label_idx": int(record["label_idx"]),
                "label_name": record["label_name"],
                "wnid": record["wnid"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "image_name", "score", "label_idx", "label_name", "wnid"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.keep_count > args.sample_count:
        raise ValueError("--keep-count must be <= --sample-count")

    device = choose_device(args.device)
    wnid_to_class_index = load_wnid_to_class_index(args.class_index_json)
    records = download_random_images(
        dataset_id=args.dataset_id,
        split=args.split,
        sample_count=args.sample_count,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        random_dir=args.random_dir,
        wnid_to_class_index=wnid_to_class_index,
        resume=args.resume,
    )
    correct = score_records(records, args.batch_size, device)
    export_selected(correct, args.keep_count, args.selected_dir, args.csv_path)
    print(f"Downloaded random images: {len(records)} -> {args.random_dir}")
    print(f"Correct predictions: {len(correct)}")
    print(f"Selected images: {args.keep_count} -> {args.selected_dir}")
    print(f"CSV: {args.csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
