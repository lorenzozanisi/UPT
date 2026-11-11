"""Plot performance metrics from results_*.yaml files in plots/trained_on_* folders.

This is the script that produced plots/metrics_summary/metrics.png (12 Jun 2026); the only change
is that the default --plots-dir points at <repo>/plots from its location in configs_handshake/code/.

For each metric (avg_abserr, avg_r2, avg_relerr), produces a grouped bar chart
where the x-axis groups are the test sets and bars within a group correspond to
different training sets.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
plt.rcParams['font.size'] = 25
METRICS = ["avg_r2"]
METRIC_LABELS = {
    "avg_r2": "Average R²",
}


def collect_results(plots_dir: Path) -> dict:
    """Walk plots/trained_on_*/<run>/<TRAIN>_on_<TEST>/.../results_*.yaml.

    Returns: {train_set: {test_set: {metric: value}}}
    """
    results: dict = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(r"^(?P<train>.+)_on_(?P<test>.+)$")

    for trained_on_dir in sorted(plots_dir.glob("trained_on_*")):
        train_set = trained_on_dir.name.replace("trained_on_", "")
        for yaml_path in trained_on_dir.rglob("results_*.yaml"):
            # The folder two levels above the yaml is "<TRAIN>_on_<TEST>"
            if "CD_ED" in str(yaml_path) or "CD_SX" in str(yaml_path) or "ED_SX" in str(yaml_path):
                continue
            on_dir = yaml_path.parent.parent.name
            m = pattern.match(on_dir)
            if not m:
                continue
            test_set = m.group("test")
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
            for metric in METRICS:
                if metric in data:
                    results[train_set][test_set][metric] = float(data[metric])

    return {k: dict(v) for k, v in results.items()}


def _plot_metric_on_axis(ax, results: dict, metric: str) -> None:
    # Multi-component train sets (e.g. "CD_ED") on the left, singles on the right.
    train_sets = sorted(results.keys(), key=lambda t: ("_" not in t, t))
    test_sets = sorted({t for r in results.values() for t in r.keys()})

    n_trains = len(train_sets)
    n_tests = len(test_sets)
    group_spacing = 1.5
    x = np.arange(n_trains) * group_spacing
    width = 0.8 / max(n_tests, 1)
    cmap = plt.get_cmap("tab10")

    for i, test_set in enumerate(test_sets):
        values = [results[t].get(test_set, {}).get(metric, np.nan) for t in train_sets]
        offset = (i - (n_tests - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=f"deployed on {test_set}",
            color=cmap(i % 10),
        )
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    # fontsize=11,
                    # rotation=90,
                )

    ax.axhline(0.86, ls='--', color='firebrick', label='full dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"trained on {t}" for t in train_sets],
        # fontsize=13,
        # rotation=45,
        ha="right",
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric)) #), fontsize=14)
    ax.set_title(METRIC_LABELS.get(metric, metric) )#, fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_all_metrics(results: dict, output_path: Path) -> None:
    n = len(METRICS)
    ncols = 2 #(n ) // 2
    fig, axes = plt.subplots(1, ncols, figsize=(24 , 10), squeeze=False)
    flat_axes = axes.flatten()
    for ax, metric in zip(flat_axes, METRICS):
        _plot_metric_on_axis(ax, results, metric)
    

    empty_axes = list(flat_axes[n:])
    if empty_axes:
        legend_ax = empty_axes[0]
        legend_ax.axis("off")
        handles, labels = flat_axes[0].get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            loc="upper left",
            #fontsize=16,
            frameon=True,
          #  title="Test set",
            #title_fontsize=17,
        )
        for ax in empty_axes[1:]:
            ax.axis("off")

        
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    default_plots = Path(__file__).resolve().parents[2] / "plots"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=default_plots,
        help="Path to the plots/ directory containing trained_on_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the bar charts (default: <plots-dir>/metrics_summary).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.plots_dir / "metrics_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(args.plots_dir)
    if not results:
        raise SystemExit(f"No results_*.yaml found under {args.plots_dir}")

    out_path = output_dir / "metrics.png"
    plot_all_metrics(results, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
