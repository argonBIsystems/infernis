#!/usr/bin/env python3
"""Train the U-Net heatmap model on rasterized fire data.

Expects data/processed/heatmap/ to exist (run prepare_heatmap_data first).
Outputs: models/heatmap_v1.pt

Usage:
    python scripts/train_heatmap.py [--epochs 100] [--batch-size 4] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from infernis.training.heatmap_data import get_dataloaders
from infernis.training.heatmap_model import FireUNet, HeatmapTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_heatmap")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net heatmap model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    heatmap_dir = project_root / "data" / "processed" / "heatmap"
    output_path = project_root / "models" / "heatmap_v1.pt"

    if not heatmap_dir.exists():
        logger.error("Heatmap data not found at %s. Run prepare_heatmap_data first.", heatmap_dir)
        sys.exit(1)

    # Create dataloaders
    logger.info("Creating dataloaders (batch_size=%d)...", args.batch_size)
    train_loader, val_loader, test_loader = get_dataloaders(
        heatmap_dir,
        batch_size=args.batch_size,
        num_workers=0,  # MPS doesn't support multiprocess workers well
    )
    logger.info(
        "Data: %d train, %d val, %d test samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    # Create model and trainer
    model = FireUNet(base_filters=args.base_filters)
    trainer = HeatmapTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    logger.info("Model: FireUNet (base_filters=%d)", args.base_filters)
    logger.info("Device: %s", trainer.device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{n_params:,}")

    # Train
    logger.info("Starting training for %d epochs...", args.epochs)
    t0 = time.time()
    history = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        output_path=output_path,
    )
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # Evaluate on test set
    logger.info("Evaluating on test set (2024)...")
    test_metrics = trainer.validate(test_loader)
    logger.info(
        "Test: loss=%.4f, precision=%.4f, recall=%.4f, F1=%.4f",
        test_metrics["loss"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    # Compute AUC on test set
    try:
        from sklearn.metrics import roc_auc_score

        model.eval()
        import torch
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["features"].to(trainer.device)
                labels = batch["labels"]
                outputs = model(inputs).cpu().numpy()
                all_preds.append(outputs.flatten())
                all_labels.append(labels.numpy().flatten())

        preds_flat = np.concatenate(all_preds)
        labels_flat = np.concatenate(all_labels)
        # Only compute AUC where there's label variation
        if labels_flat.sum() > 0 and labels_flat.sum() < len(labels_flat):
            auc = roc_auc_score(labels_flat, preds_flat)
            logger.info("Test AUC-ROC: %.4f", auc)
            test_metrics["auc_roc"] = float(auc)
    except Exception as e:
        logger.warning("Could not compute AUC: %s", e)

    # Save training history and metrics
    metrics = {
        "train_loss_final": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss_final": history["val_loss"][-1] if history["val_loss"] else None,
        "val_f1_best": max(history["val_f1"]) if history["val_f1"] else None,
        "n_epochs_trained": len(history["train_loss"]),
        "test_metrics": test_metrics,
        "config": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "base_filters": args.base_filters,
            "device": str(trainer.device),
        },
        "elapsed_minutes": elapsed / 60,
    }

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Model saved to %s", output_path)


if __name__ == "__main__":
    main()
