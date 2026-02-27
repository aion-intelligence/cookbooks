#!/usr/bin/env python3
"""
Quick test of batch inference with Ray Data.
"""

import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
import numpy as np
from typing import Dict


class ImageClassifier:
    """Batch inference predictor."""

    def __init__(self):
        import torch
        from torchvision import models

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load pretrained ResNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get class labels
        self.labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import torch

        images = batch["image"]

        # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
        tensors = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0

        # Normalize for ResNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensors = (tensors - mean) / std
        tensors = tensors.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensors)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(1, dim=1)

        # Get predictions
        predictions = [self.labels[idx.item()] for idx in top_indices.squeeze()]
        confidences = top_probs.squeeze().cpu().numpy()

        return {
            **batch,
            "prediction": np.array(predictions),
            "confidence": np.array(confidences),
        }


def main():
    print("=" * 60)
    print("Ray Data - Batch Inference Test")
    print("=" * 60)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, ignore_reinit_error=True)

    # Create synthetic image dataset
    print("\nCreating synthetic image dataset...")
    n_images = 100
    images = []
    for i in range(n_images):
        # Random 224x224 RGB images
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append({"id": i, "image": img})

    # Create Ray Dataset
    ds = ray.data.from_items(images)
    print(f"Dataset: {ds.count()} images")

    # Run batch inference
    print("\nRunning batch inference with ResNet18...")
    print("-" * 60)

    results = ds.map_batches(
        ImageClassifier,
        batch_size=16,
        concurrency=1,
    )

    # Collect results
    predictions = list(results.take(10))

    print("\nSample Predictions (random images → random classes):")
    for row in predictions[:10]:
        print(f"  Image {row['id']:3d}: {row['prediction']:<20s} ({row['confidence']:.1%})")

    # Count unique predictions
    all_preds = [r["prediction"] for r in results.take_all()]
    unique_preds = set(all_preds)
    print(f"\nProcessed {len(all_preds)} images")
    print(f"Unique predicted classes: {len(unique_preds)}")

    print("\n" + "=" * 60)
    print("Batch Inference Complete!")
    print("=" * 60)

    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
