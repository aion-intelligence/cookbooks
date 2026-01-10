#!/usr/bin/env python3
"""
Quick test of distributed training on local machine.
Runs a small PyTorch training job with Ray Train.
"""

import os

# Suppress warnings for cleaner output
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

# Configuration
NUM_WORKERS = 1  # CPU-only, single worker
EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001


def train_func(config):
    """Training function that runs on each worker."""
    # All imports must be inside the function for Ray serialization
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from ray import train

    # Simple CNN for CIFAR-10
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Use MPS if available (Apple Silicon), otherwise CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Downloading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # Use subset for quick testing
    train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
    test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print(f"Training on {device} with {len(train_dataset)} samples...")

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = train_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total

        # Report to Ray Train
        train.report({
            "loss": avg_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "epoch": epoch + 1,
        })

        print(f"  Epoch {epoch+1}/{config['epochs']} - "
              f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%, Test Acc: {test_acc:.1f}%")


def main():
    print("=" * 60)
    print("Ray Distributed Training - Local Test")
    print("=" * 60)

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=4, ignore_reinit_error=True)
    print(f"Ray initialized with resources: {ray.cluster_resources()}")

    # Training config
    train_config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    }

    print(f"\nConfig: {train_config}")
    print(f"Workers: {NUM_WORKERS}")
    print("-" * 60)

    # Create trainer
    storage_path = os.path.abspath("./runs/distributed_training")
    os.makedirs(storage_path, exist_ok=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=False,  # CPU only for MacBook
        ),
        run_config=RunConfig(
            name="cifar10-local-test",
            storage_path=storage_path,
        ),
    )

    # Run training
    print("\nStarting training...")
    result = trainer.fit()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    if result.metrics:
        print(f"Final Loss: {result.metrics.get('loss', 'N/A'):.4f}")
        print(f"Final Train Accuracy: {result.metrics.get('train_accuracy', 'N/A'):.1f}%")
        print(f"Final Test Accuracy: {result.metrics.get('test_accuracy', 'N/A'):.1f}%")
    else:
        print("Metrics reported during training (see logs above)")

    ray.shutdown()
    print("\nRay shutdown complete.")


if __name__ == "__main__":
    main()
