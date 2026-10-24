"""Optional charts (matplotlib): survival curves, death-age histogram, weight."""

from __future__ import annotations

import numpy as np

from .simulate import SimulationResult

COLORS = {"current": "#4C78A8", "optimized": "#54A24B", "adverse": "#E45756"}
NAMES = {"current": "current", "optimized": "optimized", "adverse": "adverse"}


def plot_result(result: SimulationResult, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.arange(int(result.profile.age), 112, 2)
    for name, r in result.results.items():
        color = COLORS[name]
        axes[0].plot(r.ages, r.survival, color=color, label=NAMES[name])
        axes[1].hist(r.death_age, bins=bins, color=color, alpha=0.45, label=NAMES[name])
        axes[2].plot(r.ages, r.weight_kg, color=color, label=NAMES[name])
    axes[0].set(title="Survival", xlabel="age", ylabel="P(alive)")
    axes[1].set(title="Age at death (simulated lives)", xlabel="age", ylabel="lives")
    axes[2].set(title="Body weight (energy-balance model)", xlabel="age", ylabel="kg")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig.suptitle("Lifespan Predictor - simulated estimate, not for diagnosis", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
