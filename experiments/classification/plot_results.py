#!/usr/bin/python
"""Plots for the six classification runs.

accuracy_per_epoch: train and test accuracy per epoch, one panel per machine. Colour and
marker identify the dataset; test is a solid line, train a dashed one.

best_test_accuracy: the best test accuracy over the epochs, TM against HTM per dataset.

Run from the repo root:  ./.venv/bin/python experiments/classification/plot_results.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

MACHINES = [("tm", "Tsetlin machine"), ("hierarchy_tm", "Flat hierarchical Tsetlin machine")]
# Colour plus a distinct marker, so orange and green stay apart for colour-blind readers.
DATASETS = [
    ("imdb", "IMDB", "tab:blue", "o"),
    ("amazon", "Amazon", "tab:orange", "s"),
    ("finance", "Finance", "tab:green", "^"),
]


def read(path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_accuracy"]) for r in rows]
    test = [float(r["test_accuracy"]) for r in rows]
    return epochs, train, test


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, (folder, title) in zip(axes, MACHINES, strict=True):
    for name, _label, colour, marker in DATASETS:
        epochs, train, test = read(RESULTS / folder / f"{name}.csv")
        ax.plot(epochs, test, color=colour, marker=marker, markersize=6, linewidth=2)
        ax.plot(
            epochs,
            train,
            color=colour,
            marker=marker,
            markersize=6,
            linewidth=2,
            linestyle="--",
            markerfacecolor="white",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

axes[0].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")

# Two legend groups: colour/marker = dataset, line style = split.
handles = [
    Line2D([], [], color=colour, marker=marker, linewidth=2, label=label)
    for _name, label, colour, marker in DATASETS
]
handles += [
    Line2D([], [], color="black", linewidth=2, label="Test"),
    Line2D([], [], color="black", linewidth=2, linestyle="--", label="Train"),
]
fig.tight_layout()
# Below the panels, so it never covers a curve.
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=len(handles),
    fontsize=11,
    frameon=False,
)
for suffix in (".pdf", ".png"):
    fig.savefig(FIGURES / f"accuracy_per_epoch{suffix}", dpi=300, bbox_inches="tight")
print(f"Wrote {FIGURES / 'accuracy_per_epoch.pdf'} and .png")

# Best test accuracy over the epochs, one bar per machine and dataset. Colour still means
# dataset, as above; the machine is told apart by the fill (TM solid, HTM hatched).
WIDTH = 0.38
fig, ax = plt.subplots(figsize=(7, 4.5))

for d, (name, _label, colour, _marker) in enumerate(DATASETS):
    for m, (folder, _title) in enumerate(MACHINES):
        best = max(read(RESULTS / folder / f"{name}.csv")[2])
        hatched = m == 1
        bar = ax.bar(
            d + (m - 0.5) * WIDTH,
            best,
            WIDTH,
            color="white" if hatched else colour,
            edgecolor=colour,
            hatch="///" if hatched else None,
            linewidth=1.5,
        )
        ax.bar_label(bar, fmt="%.2f", padding=3, fontsize=10)

ax.set_xticks(range(len(DATASETS)), [label for _name, label, _c, _m in DATASETS])
ax.tick_params(axis="x", labelsize=12)
ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
ax.set_ylabel("Best test accuracy (%)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)

ax.legend(
    handles=[
        Patch(facecolor="gray", edgecolor="gray", label="Tsetlin machine"),
        Patch(facecolor="white", edgecolor="gray", hatch="///", label="Flat hierarchical TM"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=11,
    frameon=False,
)

fig.tight_layout()
for suffix in (".pdf", ".png"):
    fig.savefig(FIGURES / f"best_test_accuracy{suffix}", dpi=300, bbox_inches="tight")
print(f"Wrote {FIGURES / 'best_test_accuracy.pdf'} and .png")
