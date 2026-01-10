#!/usr/bin/env python3
"""
Quick test of hyperparameter tuning with Ray Tune.
"""

import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def train_cifar(config):
    """Training function for HPO."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from ray import tune

    # Build model with tunable architecture
    model = nn.Sequential(
        nn.Conv2d(3, config["filters1"], 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(config["filters1"], config["filters2"], 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(config["filters2"] * 8 * 8, config["hidden"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden"], 10),
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    # Data (small subset for speed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # Use tiny subset for quick HPO
    train_data = torch.utils.data.Subset(train_data, range(2000))
    test_data = torch.utils.data.Subset(test_data, range(500))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    for epoch in range(3):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, predicted = model(inputs).max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / total
        # Report metrics to Ray Tune
        tune.report({"accuracy": accuracy, "epoch": epoch})


def main():
    print("=" * 60)
    print("Ray Tune - Hyperparameter Optimization Test")
    print("=" * 60)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, ignore_reinit_error=True)

    # Search space
    search_space = {
        "filters1": tune.choice([16, 32]),
        "filters2": tune.choice([32, 64]),
        "hidden": tune.choice([128, 256]),
        "dropout": tune.uniform(0.1, 0.4),
        "lr": tune.loguniform(1e-4, 1e-2),
    }

    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=3,
        grace_period=1,
    )

    print("\nSearch space:")
    for k, v in search_space.items():
        print(f"  {k}: {v}")
    print(f"\nRunning 6 trials with ASHA early stopping...")
    print("-" * 60)

    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(train_cifar, {"cpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=6,
            max_concurrent_trials=2,
        ),
    )

    results = tuner.fit()

    # Print results
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)

    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"\nBest Trial:")
    print(f"  Accuracy: {best.metrics['accuracy']:.2%}")
    print(f"  Config:")
    for k, v in best.config.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Show all trials
    print(f"\nAll Trials:")
    df = results.get_dataframe()
    for _, row in df.iterrows():
        print(f"  Acc: {row['accuracy']:.2%} | lr={row['config/lr']:.4f}, hidden={row['config/hidden']}")

    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
