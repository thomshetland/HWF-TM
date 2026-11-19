#!/usr/bin/python
"""Compare the vanilla TM with the hierarchical TM on the three same-dataset pair tasks.

Reads the per-epoch CSVs in results/ written by tm/*.py and hierarchy_tm/*.py and writes
figures/tm_vs_htm_test_accuracy.{pdf,png}, plus a summary table on stdout.

Run from the repo root:  ./.venv/bin/python experiments/documents/plot_results.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# (panel title, file name of the task in results/tm and results/hierarchy_tm)
TASKS = [
    ("IMDB vs. financial news", "imdb_finance"),
    ("Amazon vs. financial news", "awz_finance"),
    ("IMDB vs. Amazon", "imdb_awz"),
]

# Reference categorical slots 1-3 (validated all-pairs); line style and marker carry identity
# too, so the figure survives greyscale printing.
MODELS = [
    ("TM", dict(color="#2a78d6", linestyle="-", marker="o")),
    ("HTM flat", dict(color="#eb6834", linestyle="--", marker="s")),
    ("HTM two-leaf", dict(color="#1baf7a", linestyle=":", marker="^")),
]


def load(path):
    """Columns of a metrics CSV as float arrays, skipping non-numeric trailing notes."""
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["epoch"].strip().isdigit()]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def runs(task):
    """{model: [run, ...]} for one task; the hierarchy may have several seeds."""
    _, stem = task
    out = {"TM": [load(RESULTS / "tm" / f"{stem}.csv")]}
    for name, structure in [("HTM flat", "flat"), ("HTM two-leaf", "two-leaf")]:
        out[name] = [
            load(p)
            for p in sorted((RESULTS / "hierarchy_tm").glob(f"{stem}_{structure}_seed*.csv"))
        ]
    return out


def summarise(model_runs):
    """Summary over seeds: (mean, std) of each statistic; std is 0 for a single run."""
    stats = {
        "train_acc": [r["train_accuracy"][-1] for r in model_runs],
        "test_acc": [r["test_accuracy"][-1] for r in model_runs],
        "last5": [r["test_accuracy"][-5:].mean() for r in model_runs],
        "best": [r["test_accuracy"].max() for r in model_runs],
        "f1": [r["test_f1"][-1] for r in model_runs],
        "train_min": [r["train_seconds"].sum() / 60 for r in model_runs],
    }
    return {k: (np.mean(v), np.std(v)) for k, v in stats.items()}, len(model_runs)


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), constrained_layout=True)

for ax, task in zip(axes, TASKS, strict=True):
    task_runs = runs(task)
    print(f"\n{task[0]}")
    print(f"  {'model':<13} n  train   test        last5       best   F1     train time (min)")
    for name, style in MODELS:
        model_runs = task_runs[name]
        acc = np.stack([r["test_accuracy"] for r in model_runs])
        epochs = np.arange(1, acc.shape[1] + 1)
        ax.plot(epochs, acc.mean(0), lw=1.5, ms=3, label=name, **style)
        if len(model_runs) > 1:
            # Seed spread: min-max band around the mean.
            ax.fill_between(epochs, acc.min(0), acc.max(0), color=style["color"], alpha=0.15, lw=0)

        s, n = summarise(model_runs)
        print(
            f"  {name:<13} {n}  {s['train_acc'][0]:6.2f}  "
            f"{s['test_acc'][0]:5.2f}±{s['test_acc'][1]:.2f}  "
            f"{s['last5'][0]:5.2f}±{s['last5'][1]:.2f}  {s['best'][0]:5.2f}  "
            f"{s['f1'][0]:.3f}  {s['train_min'][0]:6.1f}"
        )

    ax.set_title(task[0], fontsize=8.5)
    ax.set_xlabel("Epoch")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Test accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)

FIGURES.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(FIGURES / f"tm_vs_htm_test_accuracy.{ext}", dpi=200)
print(f"\nWrote {FIGURES / 'tm_vs_htm_test_accuracy.pdf'}")
