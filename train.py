"""Train the CPU-only small U-Net after running prepare_data.py."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polypseg.data import SegmentationDataset, load_records
from polypseg.inference import load_torch_model
from polypseg.metrics import binary_metrics, mean_metrics
from polypseg.model import build_model, dice_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            losses.append(float((loss_fn(logits, targets) + dice_loss(logits, targets)).item()))
            predictions = torch.sigmoid(logits).ge(0.5).cpu().numpy()
            for target, prediction in zip(targets.numpy(), predictions):
                rows.append(binary_metrics(target[0], prediction[0]))
    return float(np.mean(losses)), mean_metrics(rows)


def load_history(path: Path, before_epoch: int) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["epoch"]) < before_epoch]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs, from 1 to the approved maximum of 25.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=(256,), default=256, help="Fixed CPU project input size.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on a small data subset.")
    parser.add_argument("--resume", type=Path, help="Continue from a saved checkpoint without repeating completed epochs.")
    parser.add_argument("--model", choices=("unet", "resnet18_unet"), default="unet")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights for the ResNet18 experiment.")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune all ResNet18 encoder layers.")
    parser.add_argument("--tag", default="", help="Optional artifact name prefix for a separate experiment.")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "image_size", "patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.epochs > 25:
        parser.error("--epochs cannot exceed the approved maximum of 25")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    seed_everything(args.seed)

    manifest = ROOT / "data" / "processed" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError("Missing data/processed/manifest.csv. Run prepare_data.py first.")
    train_records = load_records(manifest, "train")
    val_records = load_records(manifest, "val")
    if args.smoke:
        train_records, val_records = train_records[:16], val_records[:8]
        args.epochs = 1
    train_set = SegmentationDataset(train_records, args.image_size, augment=True)
    val_set = SegmentationDataset(val_records, args.image_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=min(args.batch_size, len(val_set)), shuffle=False, num_workers=0)

    model = build_model(args.model, pretrained=args.model == "resnet18_unet" and not args.no_pretrained).cpu()
    if args.model == "resnet18_unet":
        model.set_encoder_trainable(full=args.unfreeze_encoder)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    best_dice, remaining_patience, start_epoch = -1.0, args.patience, 1
    resumed_from: int | None = None
    if args.resume:
        model, checkpoint = load_torch_model(args.resume)
        args.model = str(checkpoint.get("model_name", "unet"))
        if int(checkpoint.get("image_size", args.image_size)) != args.image_size:
            parser.error("Resume checkpoint image size does not match the fixed 256px input")
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        resumed_from = int(checkpoint.get("epoch", 0))
        start_epoch = resumed_from + 1
        best_dice = float(checkpoint.get("validation_metrics", {}).get("dice", -1.0))
    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    prefix = f"{args.tag}_" if args.tag else ""
    checkpoint_path = artifacts / (f"{prefix}smoke.pt" if args.smoke else f"{prefix}best.pt")
    last_checkpoint_path = artifacts / (f"{prefix}smoke_last.pt" if args.smoke else f"{prefix}last.pt")
    history_path = reports / (f"{prefix}smoke_history.csv" if args.smoke else f"{prefix}training_history.csv")
    history = load_history(history_path, start_epoch) if args.resume else []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets) + dice_loss(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "base_channels": 16,
            "model_name": args.model,
            "pretrained_encoder": args.model == "resnet18_unet" and not args.no_pretrained,
            "image_size": args.image_size,
            "epoch": epoch,
            "validation_metrics": val_metrics,
        }
        torch.save(payload, last_checkpoint_path)
        if val_metrics["dice"] > best_dice:
            best_dice, remaining_patience = val_metrics["dice"], args.patience
            torch.save(payload, checkpoint_path)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                print("Early stopping: validation Dice did not improve.")
                break

    if not history or int(history[-1]["epoch"]) < start_epoch:
        raise ValueError("Resume checkpoint already reached the configured epoch limit")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "last_checkpoint": last_checkpoint_path.relative_to(ROOT).as_posix(),
        "best_validation_dice": best_dice,
        "epochs_ran": int(history[-1]["epoch"]),
        "epochs_logged": len(history),
        "resumed_from_epoch": resumed_from,
    }
    (reports / (f"{prefix}smoke_summary.json" if args.smoke else f"{prefix}training_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
