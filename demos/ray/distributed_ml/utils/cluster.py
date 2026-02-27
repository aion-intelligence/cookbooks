"""
Ray cluster setup utilities supporting both local Ray and Anyscale.
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ClusterMode(Enum):
    """Cluster deployment mode."""
    LOCAL = "local"
    ANYSCALE = "anyscale"


def init_ray(
    mode: ClusterMode = ClusterMode.LOCAL,
    num_gpus: Optional[int] = None,
    num_cpus: Optional[int] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Initialize Ray cluster based on deployment mode.

    Args:
        mode: ClusterMode.LOCAL or ClusterMode.ANYSCALE
        num_gpus: Number of GPUs to use (local mode only)
        num_cpus: Number of CPUs to use (local mode only)
        runtime_env: Runtime environment configuration
        **kwargs: Additional arguments passed to ray.init()
    """
    import ray

    if ray.is_initialized():
        logger.info("Ray already initialized, skipping...")
        return

    if mode == ClusterMode.LOCAL:
        init_kwargs = {}

        if num_gpus is not None:
            init_kwargs["num_gpus"] = num_gpus
        if num_cpus is not None:
            init_kwargs["num_cpus"] = num_cpus
        if runtime_env is not None:
            init_kwargs["runtime_env"] = runtime_env

        init_kwargs.update(kwargs)

        ray.init(**init_kwargs)
        logger.info("Initialized local Ray cluster")
        print(f"Ray Dashboard: {ray.dashboard_url}")

    elif mode == ClusterMode.ANYSCALE:
        init_kwargs = {"address": "auto"}
        if runtime_env is not None:
            init_kwargs["runtime_env"] = runtime_env
        init_kwargs.update(kwargs)

        ray.init(**init_kwargs)
        logger.info("Connected to Anyscale cluster")

    _print_cluster_info()


def shutdown_ray() -> None:
    """Shutdown Ray cluster."""
    import ray

    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray cluster shutdown complete")


def _print_cluster_info() -> None:
    """Print cluster resource information."""
    import ray

    resources = ray.cluster_resources()

    print("\nCluster Resources:")
    print(f"  CPUs: {resources.get('CPU', 0):.0f}")
    print(f"  GPUs: {resources.get('GPU', 0):.0f}")
    print(f"  Memory: {resources.get('memory', 0) / (1024**3):.1f} GB")


def get_scaling_config(
    num_workers: int,
    use_gpu: bool = True,
    resources_per_worker: Optional[Dict[str, float]] = None,
):
    """
    Create a Ray Train ScalingConfig.

    Args:
        num_workers: Number of distributed training workers
        use_gpu: Whether to use GPUs
        resources_per_worker: Custom resource requirements per worker

    Returns:
        ray.train.ScalingConfig instance
    """
    from ray.train import ScalingConfig

    config_kwargs = {
        "num_workers": num_workers,
        "use_gpu": use_gpu,
    }

    if resources_per_worker is not None:
        config_kwargs["resources_per_worker"] = resources_per_worker

    return ScalingConfig(**config_kwargs)


def get_run_config(
    name: str,
    storage_path: Optional[str] = None,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    failure_config: Optional[Dict[str, Any]] = None,
):
    """
    Create a Ray Train RunConfig.

    Args:
        name: Experiment name
        storage_path: Path for checkpoints (local path or cloud URI)
        checkpoint_config: Checkpoint configuration options
        failure_config: Failure handling configuration

    Returns:
        ray.train.RunConfig instance
    """
    from ray.train import RunConfig, CheckpointConfig, FailureConfig

    config_kwargs = {"name": name}

    if storage_path is not None:
        config_kwargs["storage_path"] = storage_path

    if checkpoint_config is not None:
        config_kwargs["checkpoint_config"] = CheckpointConfig(**checkpoint_config)
    else:
        config_kwargs["checkpoint_config"] = CheckpointConfig(num_to_keep=2)

    if failure_config is not None:
        config_kwargs["failure_config"] = FailureConfig(**failure_config)
    else:
        config_kwargs["failure_config"] = FailureConfig(max_failures=3)

    return RunConfig(**config_kwargs)


def create_runtime_env(
    pip_packages: Optional[list] = None,
    env_vars: Optional[Dict[str, str]] = None,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a Ray runtime environment configuration.

    Args:
        pip_packages: Additional pip packages to install
        env_vars: Environment variables to set
        working_dir: Working directory for workers

    Returns:
        Runtime environment dictionary
    """
    runtime_env = {}

    if pip_packages:
        runtime_env["pip"] = pip_packages

    if env_vars:
        runtime_env["env_vars"] = env_vars

    if working_dir:
        runtime_env["working_dir"] = working_dir

    return runtime_env


def is_anyscale_environment() -> bool:
    """Check if running in Anyscale environment."""
    return os.environ.get("ANYSCALE_SESSION_ID") is not None
