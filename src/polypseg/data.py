"""Kvasir-SEG auditing, deterministic splitting, and PyTorch dataset access."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .image_io import binary_mask, read_image, resize_pair

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    image: Path
    mask: Path
    width: int
    height: int
    foreground_pixels: int
    split: str = ""


def _file_map(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def locate_dataset(raw_parent: str | Path) -> tuple[Path, Path]:
    """Require the documented Kvasir-SEG directory layout."""
    raw_parent = Path(raw_parent)
    images = raw_parent / "Kvasir-SEG" / "images"
    masks = raw_parent / "Kvasir-SEG" / "masks"
    if images.is_dir() and masks.is_dir():
        return images, masks
    raise FileNotFoundError(
        "Expected data/raw/Kvasir-SEG/images and data/raw/Kvasir-SEG/masks. "
        f"Missing under: {raw_parent}"
    )


def audit_records(raw_parent: str | Path) -> tuple[list[SampleRecord], dict[str, object]]:
    image_dir, mask_dir = locate_dataset(raw_parent)
    image_map = _file_map(image_dir)
    mask_map = _file_map(mask_dir)
    common_names = sorted(set(image_map) & set(mask_map))
    records: list[SampleRecord] = []
    invalid: list[dict[str, str]] = []

    for name in common_names:
        image_path = image_map[name]
        mask_path = mask_map[name]
        try:
            image = read_image(image_path)
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError) as exc:
            invalid.append({"name": name, "reason": str(exc)})
            continue
        if image.shape[:2] != mask.shape[:2]:
            invalid.append({"name": name, "reason": "image_mask_size_mismatch"})
            continue
        foreground_pixels = int(binary_mask(mask).sum())
        if foreground_pixels == 0:
            invalid.append({"name": name, "reason": "empty_mask"})
            continue
        height, width = image.shape[:2]
        records.append(SampleRecord(image_path.resolve(), mask_path.resolve(), width, height, foreground_pixels))

    report: dict[str, object] = {
        "image_directory": str(image_dir),
        "mask_directory": str(mask_dir),
        "images_found": len(image_map),
        "masks_found": len(mask_map),
        "paired_files": len(common_names),
        "images_without_masks": sorted(set(image_map) - set(mask_map)),
        "masks_without_images": sorted(set(mask_map) - set(image_map)),
        "invalid_pairs": invalid,
        "valid_pairs": len(records),
    }
    return records, report


def split_records(records: Iterable[SampleRecord], seed: int = 42) -> list[SampleRecord]:
    records = list(records)
    if len(records) < 3:
        raise ValueError("At least three valid image-mask pairs are required for train/val/test splitting")
    order = np.random.default_rng(seed).permutation(len(records))
    ordered = [records[index] for index in order]
    train_count = max(1, min(round(len(ordered) * 0.70), len(ordered) - 2))
    val_count = max(1, min(round(len(ordered) * 0.15), len(ordered) - train_count - 1))
    labels = ["train"] * train_count + ["val"] * val_count + ["test"] * (len(ordered) - train_count - val_count)
    return [
        SampleRecord(item.image, item.mask, item.width, item.height, item.foreground_pixels, split)
        for item, split in zip(ordered, labels)
    ]


def prepare_dataset(raw_parent: str | Path, processed_dir: str | Path, seed: int = 42) -> dict[str, object]:
    raw_parent = Path(raw_parent)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    records, audit = audit_records(raw_parent)
    records = split_records(records, seed=seed)
    project_root = processed_dir.parent.parent

    def portable_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            return path.name

    def manifest_relative_path(path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), manifest_path.parent.resolve())).as_posix()

    manifest_path = processed_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask", "width", "height", "foreground_pixels", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image": manifest_relative_path(record.image),
                    "mask": manifest_relative_path(record.mask),
                    "width": record.width,
                    "height": record.height,
                    "foreground_pixels": record.foreground_pixels,
                    "split": record.split,
                }
            )

    split_counts = {split: sum(record.split == split for record in records) for split in ("train", "val", "test")}
    foreground_ratios = [record.foreground_pixels / (record.width * record.height) for record in records]
    report: dict[str, object] = {
        "seed": seed,
        "valid_pairs": len(records),
        "split_counts": split_counts,
        "foreground_ratio": {
            "min": float(np.min(foreground_ratios)),
            "median": float(np.median(foreground_ratios)),
            "max": float(np.max(foreground_ratios)),
        },
        "pairing_audit": {
            **audit,
            "image_directory": portable_path(Path(str(audit["image_directory"]))),
            "mask_directory": portable_path(Path(str(audit["mask_directory"]))),
        },
    }
    (processed_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_records(manifest_path: str | Path, split: str) -> list[SampleRecord]:
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    records = [
        SampleRecord(
            resolve_path(row["image"]),
            resolve_path(row["mask"]),
            int(row["width"]),
            int(row["height"]),
            int(row["foreground_pixels"]),
            row["split"],
        )
        for row in rows
        if row["split"] == split
    ]
    if not records:
        raise ValueError(f"No records found for split '{split}' in {manifest_path}")
    return records


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: Iterable[SampleRecord], image_size: int = 256, augment: bool = False) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = read_image(record.image)
        mask = read_image(record.mask, cv2.IMREAD_GRAYSCALE)
        image, mask = resize_pair(image, mask, self.image_size)
        if self.augment and np.random.random() < 0.5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        if self.augment and np.random.random() < 0.5:
            image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return image_tensor, mask_tensor
