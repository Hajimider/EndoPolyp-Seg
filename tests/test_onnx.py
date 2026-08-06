from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from polypseg.model import UNetSmall


def test_exported_onnx_matches_torch(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = UNetSmall(base_channels=4).eval()
    example = torch.randn(1, 3, 32, 32)
    output = tmp_path / "model.onnx"
    torch.onnx.export(model, example, output, input_names=["image"], output_names=["logits"], opset_version=17)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    torch_output = model(example).detach().numpy()
    onnx_output = session.run(None, {"image": example.numpy()})[0]
    assert np.max(np.abs(torch_output - onnx_output)) < 1e-4
