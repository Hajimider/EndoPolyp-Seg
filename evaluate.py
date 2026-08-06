"""Evaluate the validation-selected U-Net once on the final split."""

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
from polypseg.image_io import binary_mask, read_image, write_image
from polypseg.inference import load_torch_model, mask_from_probability, overlay_mask, preprocess_bgr
from polypseg.metrics import binary_metrics, mean_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--save-samples", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing report for the selected split.")
    args = parser.parse_args()
    report_path = ROOT / "reports" / ("evaluation.json" if args.split == "test" else "validation_evaluation.json")
    if report_path.exists() and not args.overwrite:
        parser.error(f"{report_path.name} already exists. Use --overwrite only for a deliberate rerun.")
    model, checkpoint = load_torch_model(args.weights)
    image_size = int(checkpoint.get("image_size", 256))
    records = load_records(ROOT / "data" / "processed" / "manifest.csv", args.split)
    sample_dir = ROOT / "reports" / ("prediction_samples" if args.split == "test" else "validation_prediction_samples")
    rows = []

    with torch.no_grad():
        for index, record in enumerate(records):
            image = read_image(record.image)
            tensor, original_shape = preprocess_bgr(image, image_size)
            logits = model(torch.from_numpy(tensor)).numpy()[0, 0]
            probability = 1.0 / (1.0 + np.exp(-logits))
            prediction = mask_from_probability(probability, original_shape)
            target = binary_mask(read_image(record.mask, cv2.IMREAD_GRAYSCALE))
            rows.append(binary_metrics(target, prediction))
            if index < args.save_samples:
                stem = f"{index + 1:02d}_{record.image.stem}"
                write_image(sample_dir / f"{stem}_image.jpg", image)
                write_image(sample_dir / f"{stem}_target.png", target * 255)
                write_image(sample_dir / f"{stem}_prediction.png", prediction * 255)
                write_image(sample_dir / f"{stem}_overlay.jpg", overlay_mask(image, prediction))

    report = {
        "weights": args.weights.resolve().relative_to(ROOT).as_posix(),
        "split": args.split,
        "images": len(rows),
        "threshold": 0.5,
        **mean_metrics(rows),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
