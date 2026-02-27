"""
Utility modules for Ray Distributed ML cookbooks.
"""

from .gpu import (
    detect_gpus,
    print_gpu_status,
    get_recommended_workers,
    configure_environment,
)

from .cluster import (
    ClusterMode,
    init_ray,
    shutdown_ray,
    get_scaling_config,
    get_run_config,
    create_runtime_env,
    is_anyscale_environment,
)

from .viz import (
    plot_training_curves,
    plot_tune_results,
    print_best_trial,
    format_metrics_table,
    save_results_json,
)

__all__ = [
    # GPU utilities
    "detect_gpus",
    "print_gpu_status",
    "get_recommended_workers",
    "configure_environment",
    # Cluster utilities
    "ClusterMode",
    "init_ray",
    "shutdown_ray",
    "get_scaling_config",
    "get_run_config",
    "create_runtime_env",
    "is_anyscale_environment",
    # Visualization utilities
    "plot_training_curves",
    "plot_tune_results",
    "print_best_trial",
    "format_metrics_table",
    "save_results_json",
]
