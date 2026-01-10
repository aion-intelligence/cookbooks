# GenAI with Ray

Scale your generative AI workloads with Ray - from fine-tuning LLMs to deploying them at scale.

## What's Included

| Notebook | Description |
|----------|-------------|
| `llm_finetuning.ipynb` | Distributed LLM fine-tuning with Ray Train and DeepSpeed |
| `llm_serving.ipynb` | Deploy LLMs with Ray Serve and vLLM |
| `batch_inference_llm.ipynb` | Efficient offline batch inference with Ray Data |

## Why Ray for GenAI?

- **Scale seamlessly**: Same code works from 1 GPU to hundreds
- **Framework agnostic**: Works with HuggingFace, DeepSpeed, vLLM, and more
- **Production ready**: Built-in fault tolerance, checkpointing, and autoscaling
- **Cost efficient**: Automatic spot instance management, scale-to-zero

## Getting Started

### Prerequisites

- Python 3.9+
- 2-8 GPUs with 16GB+ VRAM (for 7B models)
- HuggingFace account (for gated models like Llama)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

1. **Fine-tune an LLM**:
   ```bash
   jupyter notebook llm_finetuning.ipynb
   ```

2. **Deploy for inference**:
   ```bash
   jupyter notebook llm_serving.ipynb
   ```

3. **Run batch inference**:
   ```bash
   jupyter notebook batch_inference_llm.ipynb
   ```

## Deployment Options

### Option A: Local Ray Cluster

For development and single-node multi-GPU setups:

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

### 1. LLM Fine-tuning (`llm_finetuning.ipynb`)

Fine-tune LLMs using Ray Train with:
- **DeepSpeed ZeRO**: Memory-efficient distributed training
- **LoRA/QLoRA**: Parameter-efficient fine-tuning
- **Multi-GPU scaling**: 2-8 GPUs with automatic data parallelism
- **Checkpointing**: Automatic checkpoint saving and resume

**Supported models**:
- TinyLlama (1.1B) - for development
- Phi-2 (2.7B) - balanced performance
- Llama-2 (7B) - production quality

### 2. LLM Serving (`llm_serving.ipynb`)

Deploy LLMs for real-time inference with:
- **vLLM integration**: High-throughput inference with PagedAttention
- **OpenAI-compatible API**: Drop-in replacement for OpenAI client
- **Autoscaling**: Automatic scale up/down based on traffic
- **Multi-model serving**: Route to different models from one endpoint

### 3. Batch Inference (`batch_inference_llm.ipynb`)

Process large datasets offline with:
- **Ray Data pipelines**: Streaming CPU preprocessing → GPU inference
- **Automatic batching**: Optimal batch sizes for your hardware
- **Text generation**: Summarization, completion, etc.
- **Classification**: Zero-shot classification with LLMs

## Project Structure

```
genai/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── llm_finetuning.ipynb        # Fine-tuning notebook
├── llm_serving.ipynb           # Serving notebook
├── batch_inference_llm.ipynb   # Batch inference notebook
├── assets/                     # Images and visual assets
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── gpu.py                  # GPU detection utilities
│   └── cluster.py              # Ray/Anyscale cluster setup
└── runs/                       # Output directory (gitignored)
```

## Utility Functions

The `utils/` package provides helper functions used across all notebooks:

```python
from utils import (
    # GPU utilities
    print_gpu_status,           # Display GPU information
    get_recommended_workers,    # Get optimal worker count
    get_deepspeed_config,       # Generate DeepSpeed config

    # Cluster utilities
    init_ray,                   # Initialize Ray cluster
    shutdown_ray,               # Shutdown Ray cluster
    ClusterMode,                # LOCAL or ANYSCALE
    get_scaling_config,         # Ray Train scaling config
    get_run_config,             # Ray Train run config
)
```

## Hardware Requirements

| Model Size | Min GPUs | Recommended | VRAM per GPU |
|------------|----------|-------------|--------------|
| 1-3B       | 1        | 2           | 16GB         |
| 7B         | 2        | 4           | 24GB         |
| 13B        | 4        | 8           | 40GB         |
| 70B        | 8        | 16          | 80GB         |

## Resources

### Ray Documentation
- [Ray Train](https://docs.ray.io/en/latest/train/train.html) - Distributed training
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Model serving
- [Ray Data](https://docs.ray.io/en/latest/data/data.html) - Data processing

### Tutorials
- [LLM Fine-tuning with DeepSpeed](https://docs.ray.io/en/latest/train/examples/pytorch/deepspeed_finetune/README.html)
- [Serving LLMs with Ray](https://docs.ray.io/en/latest/serve/llm/index.html)
- [Working with LLMs in Ray Data](https://docs.ray.io/en/latest/data/working-with-llms.html)

### Anyscale
- [Anyscale Documentation](https://docs.anyscale.com/)
- [Anyscale RayLLM](https://www.anyscale.com/product/library/ray-llm)
