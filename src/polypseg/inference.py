"""Shared preprocessing, checkpoint loading, ONNX inference, and visualization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

from .model import build_model
from .postprocess import clean_mask


def load_torch_model(checkpoint_path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in payload:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    model = build_model(str(payload.get("model_name", "unet")), pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def preprocess_bgr(image_bgr: np.ndarray, image_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("Expected a BGR color image")
    original_shape = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    return tensor, original_shape


def mask_from_probability(probability: np.ndarray, original_shape: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32).squeeze()
    height, width = original_shape
    resized = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    return (resized >= threshold).astype(np.uint8)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show a transparent green region and a red boundary on the source image."""
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask sizes must match for visualization")
    region = image_bgr.copy()
    region[mask.astype(bool)] = (0, 255, 0)
    overlay = cv2.addWeighted(image_bgr, 0.68, region, 0.32, 0)
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


class OnnxPredictor:
    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}. Run export_onnx.py first.")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if "CPUExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("ONNX Runtime did not select CPUExecutionProvider")
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.image_size = int(input_shape[-1]) if isinstance(input_shape[-1], int) else 256

    def predict(self, image_bgr: np.ndarray, threshold: float = 0.5, min_area: int = 0, kernel_size: int = 0) -> tuple[np.ndarray, float]:
        tensor, original_shape = preprocess_bgr(image_bgr, self.image_size)
        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: tensor})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probability = 1.0 / (1.0 + np.exp(-logits[0, 0]))
        mask = mask_from_probability(probability, original_shape, threshold)
        return clean_mask(mask, min_area=min_area, kernel_size=kernel_size), elapsed_ms
