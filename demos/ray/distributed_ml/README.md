# Distributed ML/DL with Ray

Scale your machine learning workflows from a single GPU to a cluster with minimal code changes.

## What's Included

| Notebook | Description |
|----------|-------------|
| `distributed_training.ipynb` | Distributed PyTorch and XGBoost training |
| `hyperparameter_tuning.ipynb` | Hyperparameter optimization with Ray Tune |
| `batch_inference.ipynb` | Scalable batch inference with Ray Data |

## Why Ray for ML?

- **Minimal code changes**: Add a few decorators to existing training code
- **Automatic DDP**: No manual distributed process group setup
- **Fault tolerance**: Resume training after node failures
- **Unified stack**: Same framework for training, tuning, and inference

## Getting Started

### Prerequisites

- Python 3.9+
- 2-8 GPUs (for distributed training)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

1. **Distributed Training**:
   ```bash
   jupyter notebook distributed_training.ipynb
   ```

2. **Hyperparameter Tuning**:
   ```bash
   jupyter notebook hyperparameter_tuning.ipynb
   ```

3. **Batch Inference**:
   ```bash
   jupyter notebook batch_inference.ipynb
   ```

## Deployment Options

### Option A: Local Ray Cluster

For single-node multi-GPU setups:

```python
from utils import init_ray, ClusterMode

init_ray(mode=ClusterMode.LOCAL)
```

### Option B: Anyscale Platform

For production with managed infrastructure:

```python
from utils import init_ray, ClusterMode

init_ray(mode=ClusterMode.ANYSCALE)
```

## Notebooks Overview

### 1. Distributed Training (`distributed_training.ipynb`)

Scale model training across multiple GPUs with:

**PyTorch Training**:
- Automatic DistributedDataParallel (DDP)
- Data sharding with `prepare_data_loader`
- Checkpoint management
- ResNet on CIFAR-10 example

**XGBoost Training**:
- Native `XGBoostTrainer` support
- Distributed data loading
- Multi-node tree building
- Large tabular dataset example

### 2. Hyperparameter Tuning (`hyperparameter_tuning.ipynb`)

Find optimal hyperparameters efficiently with:

- **Search Spaces**: `tune.choice`, `tune.uniform`, `tune.loguniform`
- **ASHA Scheduler**: Early stopping of underperforming trials
- **Optuna Search**: Bayesian optimization
- **Parallel Trials**: Run multiple experiments concurrently
- **Result Analysis**: Visualizations and hyperparameter importance

### 3. Batch Inference (`batch_inference.ipynb`)

Process large datasets efficiently with:

- **Image Classification**: Pretrained ResNet on image batches
- **Tabular Inference**: Sklearn models at scale
- **CPU→GPU Pipelines**: Streaming preprocessing to inference
- **Output Formats**: Save to Parquet, JSON, or custom formats

## Project Structure

```
distributed_ml/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── distributed_training.ipynb   # Training notebook
├── hyperparameter_tuning.ipynb  # HPO notebook
├── batch_inference.ipynb        # Inference notebook
├── assets/                      # Images and visual assets
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── gpu.py                   # GPU detection
│   ├── cluster.py               # Ray/Anyscale setup
│   └── viz.py                   # Visualization utilities
└── runs/                        # Output directory (gitignored)
```

## Utility Functions

```python
from utils import (
    # GPU utilities
    print_gpu_status,           # Display GPU info
    get_recommended_workers,    # Optimal worker count

    # Cluster utilities
    init_ray,                   # Initialize cluster
    shutdown_ray,               # Shutdown cluster
    ClusterMode,                # LOCAL or ANYSCALE
    get_scaling_config,         # Ray Train scaling
    get_run_config,             # Ray Train run config

    # Visualization
    plot_training_curves,       # Training metrics plot
    plot_tune_results,          # HPO results plot
    print_best_trial,           # Best trial summary
)
```

## Hardware Requirements

| Workload | Min Resources | Recommended |
|----------|--------------|-------------|
| PyTorch DDP | 2 GPUs | 4-8 GPUs |
| XGBoost | 2 CPU workers | 4-8 CPU workers |
| Hyperparameter Tuning | 2 GPUs | 4+ GPUs |
| Batch Inference | 1 GPU | 2-4 GPUs |

## Resources

### Ray Documentation
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
- [Ray Data](https://docs.ray.io/en/latest/data/data.html)

### Tutorials
- [PyTorch Distributed Training](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)
- [XGBoost with Ray](https://docs.ray.io/en/latest/train/getting-started-xgboost.html)
- [ASHA Scheduler](https://docs.ray.io/en/latest/tune/examples/tune-asha.html)

### Anyscale
- [Anyscale Documentation](https://docs.anyscale.com/)
- [Anyscale Jobs](https://docs.anyscale.com/platform/jobs/)

## License

Apache 2.0
