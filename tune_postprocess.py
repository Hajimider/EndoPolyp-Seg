"""Select mask threshold and simple cleanup settings on the validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polypseg.data import load_records
from polypseg.image_io import binary_mask, read_image
from polypseg.inference import load_torch_model, mask_from_probability, preprocess_bgr
from polypseg.metrics import binary_metrics, mean_metrics
from polypseg.postprocess import clean_mask


def collect_predictions(model: torch.nn.Module, records):
    samples = []
    with torch.no_grad():
        for record in records:
            image = read_image(record.image)
            tensor, original_shape = preprocess_bgr(image, 256)
            logits = model(torch.from_numpy(tensor)).numpy()[0, 0]
            probability = cv2.resize(1.0 / (1.0 + np.exp(-logits)), (original_shape[1], original_shape[0]))
            target = binary_mask(read_image(record.mask, cv2.IMREAD_GRAYSCALE))
            samples.append((probability, target))
    return samples


def score(samples, threshold: float, min_area: int, kernel_size: int) -> dict[str, float]:
    rows = []
    for probability, target in samples:
        prediction = clean_mask((probability >= threshold).astype(np.uint8), min_area, kernel_size)
        rows.append(binary_metrics(target, prediction))
    return mean_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "best.pt")
    args = parser.parse_args()
    model, _ = load_torch_model(args.weights)
    records = load_records(ROOT / "data" / "processed" / "manifest.csv", "val")
    samples = collect_predictions(model, records)
    candidates = []
    for threshold in (0.35, 0.45, 0.5, 0.55, 0.65):
        for min_area in (0, 32, 64, 128):
            for kernel_size in (0, 3, 5):
                metrics = score(samples, threshold, min_area, kernel_size)
                candidates.append({"threshold": threshold, "min_area": min_area, "kernel_size": kernel_size, **metrics})
    best = max(candidates, key=lambda row: (row["dice"], row["iou"]))
    report = {"split": "val", "weights": args.weights.resolve().relative_to(ROOT).as_posix(), "best": best, "candidates": candidates}
    (ROOT / "reports" / "postprocess_tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"split": "val", "best": best}, indent=2))


if __name__ == "__main__":
    main()
