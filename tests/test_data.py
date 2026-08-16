from pathlib import Path

import numpy as np

from polypseg.data import SegmentationDataset, load_records, prepare_dataset
from polypseg.image_io import read_image, write_image


def _make_dataset(raw_dir: Path, count: int = 12) -> None:
    image_dir = raw_dir / "Kvasir-SEG" / "images"
    mask_dir = raw_dir / "Kvasir-SEG" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(count):
        image = np.full((40, 60, 3), 20 + index, dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[8:24, 12:40] = 255
        write_image(image_dir / f"case_{index:02d}.png", image)
        write_image(mask_dir / f"case_{index:02d}.png", mask)


def test_prepare_dataset_is_deterministic_and_splits_all_samples(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir)
    first = tmp_path / "processed_one"
    second = tmp_path / "processed_two"
    report = prepare_dataset(raw_dir, first, seed=42)
    prepare_dataset(raw_dir, second, seed=42)
    assert report["valid_pairs"] == 12
    assert report["split_counts"] == {"train": 8, "val": 2, "test": 2}
    assert (first / "manifest.csv").read_text(encoding="utf-8") == (second / "manifest.csv").read_text(encoding="utf-8")


def test_dataset_returns_normalized_tensor_and_binary_mask(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir, count=3)
    processed = tmp_path / "processed"
    prepare_dataset(raw_dir, processed)
    record = load_records(processed / "manifest.csv", "train")
    image, mask = SegmentationDataset(record, image_size=32)[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert image.min() >= 0 and image.max() <= 1
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_prepare_dataset_records_pairing_problem(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir, count=3)
    image_dir = raw_dir / "Kvasir-SEG" / "images"
    write_image(image_dir / "missing_mask.png", np.zeros((20, 20, 3), dtype=np.uint8))
    report = prepare_dataset(raw_dir, tmp_path / "processed")
    assert report["pairing_audit"]["images_without_masks"] == ["missing_mask"]
    assert read_image(image_dir / "case_00.png").shape == (40, 60, 3)


def test_prepare_dataset_audits_invalid_pairs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir, count=3)
    image_dir = raw_dir / "Kvasir-SEG" / "images"
    mask_dir = raw_dir / "Kvasir-SEG" / "masks"
    (image_dir / "broken.png").write_bytes(b"not an image")
    (mask_dir / "broken.png").write_bytes(b"not an image")
    write_image(image_dir / "size_mismatch.png", np.zeros((30, 30, 3), dtype=np.uint8))
    write_image(mask_dir / "size_mismatch.png", np.zeros((20, 20), dtype=np.uint8))
    write_image(image_dir / "empty_mask.png", np.zeros((30, 30, 3), dtype=np.uint8))
    write_image(mask_dir / "empty_mask.png", np.zeros((30, 30), dtype=np.uint8))
    report = prepare_dataset(raw_dir, tmp_path / "processed")
    reasons = {row["reason"] for row in report["pairing_audit"]["invalid_pairs"]}
    assert "image_mask_size_mismatch" in reasons
    assert "empty_mask" in reasons
    assert any("Unable to decode image" in reason for reason in reasons)
