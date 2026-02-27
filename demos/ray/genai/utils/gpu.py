"""
GPU detection and configuration utilities for Ray GenAI workloads.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def detect_gpus() -> Dict[str, Any]:
    """
    Detect available GPUs and return configuration info.

    Returns:
        Dictionary with GPU information including count, names, and memory.
    """
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "cuda_version": None,
        "driver_version": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda

            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })

            logger.info(f"Detected {gpu_info['count']} GPU(s)")
        else:
            logger.warning("No CUDA GPUs available")

    except ImportError:
        logger.warning("PyTorch not installed, cannot detect GPUs")
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")

    return gpu_info


def print_gpu_status() -> None:
    """Print formatted GPU status information."""
    info = detect_gpus()

    if info["available"]:
        print(f"GPU Status: Available")
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  GPU Count: {info['count']}")
        print()
        for device in info["devices"]:
            print(f"  GPU {device['id']}: {device['name']}")
            print(f"    Memory: {device['total_memory_gb']} GB")
            print(f"    Compute Capability: {device['compute_capability']}")
    else:
        print("GPU Status: Not Available")
        print("  Running in CPU-only mode")


def get_recommended_workers(gpu_count: Optional[int] = None) -> int:
    """
    Get recommended number of Ray workers based on available GPUs.

    Args:
        gpu_count: Override GPU count (auto-detect if None)

    Returns:
        Recommended number of workers for distributed training.
    """
    if gpu_count is None:
        info = detect_gpus()
        gpu_count = info["count"] if info["available"] else 0

    if gpu_count == 0:
        # CPU-only: use number of cores
        import multiprocessing
        return max(1, multiprocessing.cpu_count() // 2)

    # One worker per GPU is typical for LLM workloads
    return gpu_count


def configure_environment(
    visible_devices: Optional[str] = None,
    memory_fraction: float = 0.9,
) -> None:
    """
    Configure GPU environment variables for training.

    Args:
        visible_devices: Comma-separated GPU IDs (e.g., "0,1,2,3")
        memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
    """
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={visible_devices}")

    # PyTorch memory configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:512"

    # Disable tokenizers parallelism to avoid deadlocks with Ray
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_deepspeed_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> Dict[str, Any]:
    """
    Get DeepSpeed ZeRO configuration for distributed training.

    Args:
        stage: ZeRO stage (1, 2, or 3)
        offload_optimizer: Offload optimizer states to CPU
        offload_param: Offload parameters to CPU (stage 3 only)

    Returns:
        DeepSpeed configuration dictionary.
    """
    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }

    if offload_optimizer and stage >= 2:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if offload_param and stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return config
