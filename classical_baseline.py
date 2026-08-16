"""OpenCV threshold-and-morphology reference for the final test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polypseg.data import load_records
from polypseg.image_io import binary_mask, read_image
from polypseg.metrics import binary_metrics, mean_metrics


def baseline_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, thresholded = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if count <= 1:
        return binary_mask(cleaned)
    largest_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return (labels == largest_label).astype("uint8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    records = load_records(ROOT / "data" / "processed" / "manifest.csv", args.split)
    rows = []
    for record in records:
        prediction = baseline_mask(read_image(record.image))
        target = binary_mask(read_image(record.mask, cv2.IMREAD_GRAYSCALE))
        rows.append(binary_metrics(target, prediction))
    report = {"method": "HSV saturation Otsu + morphology + largest component", "split": args.split, "images": len(rows), **mean_metrics(rows)}
    path = ROOT / "reports" / "classical_baseline.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
