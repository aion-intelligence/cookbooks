#!/usr/bin/env python3
"""
Quick smoke test for Ray cookbooks on CPU (e.g., MacBook).

Run with: python test_local.py
"""

import sys


def test_ray_init():
    """Test Ray cluster initialization."""
    print("Testing Ray initialization...")
    import ray

    if ray.is_initialized():
        ray.shutdown()

    context = ray.init(num_cpus=4, ignore_reinit_error=True)
    dashboard_url = context.dashboard_url if hasattr(context, 'dashboard_url') else "N/A"
    print(f"  Dashboard: {dashboard_url}")
    print(f"  Resources: {ray.cluster_resources()}")
    ray.shutdown()
    print("  ✓ Ray init works\n")


def test_ray_data():
    """Test Ray Data pipeline."""
    print("Testing Ray Data...")
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, ignore_reinit_error=True)

    # Create simple dataset
    ds = ray.data.range(100)

    # Apply transformation
    result = ds.map(lambda x: {"value": x["id"] * 2})
    count = result.count()

    print(f"  Processed {count} items")
    ray.shutdown()
    print("  ✓ Ray Data works\n")


def test_ray_train():
    """Test Ray Train with simple function."""
    print("Testing Ray Train...")
    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, ignore_reinit_error=True)

    def simple_train_func(config):
        # Minimal training function
        for epoch in range(2):
            train.report({"loss": 1.0 / (epoch + 1)})

    trainer = TorchTrainer(
        train_loop_per_worker=simple_train_func,
        train_loop_config={},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )

    result = trainer.fit()
    print(f"  Final loss: {result.metrics['loss']:.4f}")
    ray.shutdown()
    print("  ✓ Ray Train works\n")


def test_ray_tune():
    """Test Ray Tune with simple search."""
    print("Testing Ray Tune...")
    import ray
    from ray import tune

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, ignore_reinit_error=True)

    def objective(config):
        # Simple objective function
        score = config["x"] ** 2
        return {"score": score}

    tuner = tune.Tuner(
        objective,
        param_space={"x": tune.uniform(-10, 10)},
        tune_config=tune.TuneConfig(num_samples=5),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="score", mode="min")
    print(f"  Best x: {best.config['x']:.4f}, score: {best.metrics['score']:.4f}")
    ray.shutdown()
    print("  ✓ Ray Tune works\n")


def test_pytorch():
    """Test PyTorch availability."""
    print("Testing PyTorch...")
    import torch

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("  Apple Silicon MPS available")
    else:
        device = "cpu"
        print("  CPU only (no GPU)")

    # Simple tensor operation
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    print(f"  Tensor ops work on {device}")
    print("  ✓ PyTorch works\n")


def test_transformers():
    """Test HuggingFace transformers."""
    print("Testing Transformers...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokens = tokenizer("Hello, Ray!", return_tensors="pt")
        print(f"  Tokenized: {tokens['input_ids'].shape}")
        print("  ✓ Transformers works\n")
    except Exception as e:
        print(f"  ⚠ Transformers not installed or error: {e}\n")


def main():
    print("=" * 50)
    print("Ray Cookbooks - Local Smoke Test")
    print("=" * 50 + "\n")

    tests = [
        ("PyTorch", test_pytorch),
        ("Ray Init", test_ray_init),
        ("Ray Data", test_ray_data),
        ("Ray Train", test_ray_train),
        ("Ray Tune", test_ray_tune),
        ("Transformers", test_transformers),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} failed: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n✓ All tests passed! Ready to run notebooks.")
        print("\nNote: For GPU-heavy notebooks (LLM fine-tuning, vLLM),")
        print("use Anyscale or a cloud GPU instance.")
    else:
        print("\n⚠ Some tests failed. Check dependencies.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
