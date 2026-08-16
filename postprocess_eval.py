"""Compare raw and validation-selected mask post-processing on the test split."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--tuning", type=Path, default=ROOT / "reports" / "postprocess_tuning.json")
    args = parser.parse_args()
    model, checkpoint = load_torch_model(args.weights)
    image_size = int(checkpoint.get("image_size", 256))
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["best"]
    records = load_records(ROOT / "data" / "processed" / "manifest.csv", "test")
    raw_rows, processed_rows = [], []
    with torch.no_grad():
        for record in records:
            image = read_image(record.image)
            tensor, original_shape = preprocess_bgr(image, image_size)
            logits = model(torch.from_numpy(tensor)).numpy()[0, 0]
            probability = 1.0 / (1.0 + np.exp(-logits))
            raw = mask_from_probability(probability, original_shape, float(tuning["threshold"]))
            processed = clean_mask(raw, int(tuning["min_area"]), int(tuning["kernel_size"]))
            target = binary_mask(read_image(record.mask, cv2.IMREAD_GRAYSCALE))
            raw_rows.append(binary_metrics(target, raw))
            processed_rows.append(binary_metrics(target, processed))
    report = {
        "weights": args.weights.resolve().relative_to(ROOT).as_posix(),
        "split": "test",
        "postprocess": {key: tuning[key] for key in ("threshold", "min_area", "kernel_size")},
        "raw": mean_metrics(raw_rows),
        "processed": mean_metrics(processed_rows),
        "images": len(records),
    }
    (ROOT / "reports" / "postprocess_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
