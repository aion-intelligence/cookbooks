"""
Visualization utilities for Ray ML training metrics.
"""

from typing import List, Dict, Any, Optional
import json


def plot_training_curves(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: str = "Training Curves",
) -> None:
    """
    Plot training metrics over iterations.

    Args:
        results: List of result dictionaries from Ray Train/Tune
        metrics: List of metric names to plot (default: loss, accuracy)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    if metrics is None:
        metrics = ["loss", "accuracy"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = []
        for r in results:
            if metric in r:
                values.append(r[metric])

        if values:
            ax.plot(values, marker='o', markersize=3)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} over Training")
            ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_tune_results(
    results_df,
    x_metric: str = "training_iteration",
    y_metric: str = "loss",
    group_by: str = None,
) -> None:
    """
    Plot Ray Tune experiment results.

    Args:
        results_df: DataFrame from tune.run().results_df
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        group_by: Column to group trials by (e.g., hyperparameter name)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_by and group_by in results_df.columns:
        for name, group in results_df.groupby(group_by):
            ax.plot(group[x_metric], group[y_metric], label=f"{group_by}={name}", alpha=0.7)
        ax.legend()
    else:
        for trial_id in results_df["trial_id"].unique():
            trial_data = results_df[results_df["trial_id"] == trial_id]
            ax.plot(trial_data[x_metric], trial_data[y_metric], alpha=0.5)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"Ray Tune: {y_metric} vs {x_metric}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_best_trial(result) -> None:
    """
    Print information about the best trial from Ray Tune.

    Args:
        result: ResultGrid from tune.run()
    """
    best = result.get_best_result()

    print("Best Trial:")
    print(f"  Trial ID: {best.path}")
    print(f"  Metrics:")
    for key, value in best.metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    print(f"\n  Config:")
    for key, value in best.config.items():
        print(f"    {key}: {value}")


def format_metrics_table(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as a table string.

    Args:
        metrics: Dictionary of metric names to values
        precision: Decimal precision for floats

    Returns:
        Formatted table string
    """
    lines = ["Metrics:"]
    lines.append("-" * 40)

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:<20} {value:.{precision}f}")
        else:
            lines.append(f"  {key:<20} {value}")

    lines.append("-" * 40)
    return "\n".join(lines)


def save_results_json(
    results: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary to save
        filepath: Output file path
    """
    # Convert non-serializable types
    def convert(obj):
        if hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return str(obj)
        return obj

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"Results saved to {filepath}")
