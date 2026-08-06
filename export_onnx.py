"""Export the validation-selected U-Net and verify ONNX Runtime numerics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polypseg.inference import load_torch_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "best.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    model, checkpoint = load_torch_model(args.weights)
    image_size = int(checkpoint.get("image_size", 256))
    dummy = torch.randn(1, 3, image_size, image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        opset_version=args.opset,
    )
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    torch_output = model(dummy).detach().numpy()
    onnx_output = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
    difference = np.abs(torch_output - onnx_output)
    report = {
        "weights": args.weights.resolve().relative_to(ROOT).as_posix(),
        "onnx_model": args.output.resolve().relative_to(ROOT).as_posix(),
        "opset": args.opset,
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "providers": session.get_providers(),
    }
    (ROOT / "reports" / "onnx_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["max_abs_difference"] > 1e-4:
        raise RuntimeError(f"ONNX output differs from PyTorch by {report['max_abs_difference']:.6g}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
