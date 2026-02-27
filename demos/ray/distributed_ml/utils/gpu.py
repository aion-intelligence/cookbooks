"""
GPU detection and configuration utilities for Ray ML workloads.
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
        import multiprocessing
        return max(1, multiprocessing.cpu_count() // 2)

    return gpu_count


def configure_environment(
    visible_devices: Optional[str] = None,
) -> None:
    """
    Configure environment variables for training.

    Args:
        visible_devices: Comma-separated GPU IDs (e.g., "0,1,2,3")
    """
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={visible_devices}")

    # Optimize PyTorch memory
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
