"""Measure ONNX Runtime CPU model latency on held-out images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polypseg.data import load_records
from polypseg.image_io import read_image
from polypseg.inference import OnnxPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "best.onnx")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    records = load_records(ROOT / "data" / "processed" / "manifest.csv", "test")[: args.limit]
    if not records:
        parser.error("No test images are available. Run prepare_data.py first.")
    predictor = OnnxPredictor(args.model)
    images = [read_image(record.image) for record in records]
    for index in range(args.warmup):
        predictor.predict(images[index % len(images)])
    latencies = [predictor.predict(image)[1] for image in images]
    import numpy as np

    report = {
        "model": args.model.resolve().relative_to(ROOT).as_posix(),
        "images": len(images),
        "warmup_runs": args.warmup,
        "input_size": predictor.image_size,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
    }
    (ROOT / "reports" / "onnx_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
