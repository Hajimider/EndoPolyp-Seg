"""Single-image Gradio demo backed by the exported ONNX model."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polypseg.inference import OnnxPredictor, overlay_mask


def load_postprocess_config() -> dict[str, int | float]:
    path = ROOT / "reports" / "postprocess_tuning.json"
    if not path.is_file():
        return {"threshold": 0.5, "min_area": 0, "kernel_size": 0}
    try:
        best = json.loads(path.read_text(encoding="utf-8"))["best"]
        return {key: best[key] for key in ("threshold", "min_area", "kernel_size")}
    except (OSError, KeyError, json.JSONDecodeError):
        return {"threshold": 0.5, "min_area": 0, "kernel_size": 0}


def build_demo() -> gr.Interface:
    predictor = OnnxPredictor(ROOT / "artifacts" / "best.onnx")

    def segment(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if image_rgb is None:
            raise gr.Error("Please upload an endoscopy image.")
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        options = load_postprocess_config()
        mask, elapsed_ms = predictor.predict(image_bgr, **options)
        overlay_rgb = cv2.cvtColor(overlay_mask(image_bgr, mask), cv2.COLOR_BGR2RGB)
        details = {"predicted_area_percent": round(float(mask.mean() * 100.0), 2), "onnx_cpu_latency_ms": round(elapsed_ms, 2)}
        return mask * 255, overlay_rgb, details

    return gr.Interface(
        fn=segment,
        inputs=gr.Image(type="numpy", label="Endoscopy image"),
        outputs=[gr.Image(label="Predicted mask"), gr.Image(label="Mask overlay"), gr.JSON(label="Inference details")],
        title="EndoPolyp-Seg",
        description="基于公开 Kvasir-SEG 数据的内镜息肉分割演示，仅用于算法学习与项目展示；模型未经临床验证，不作为医疗诊断依据。",
    )


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7895, inbrowser=False)
