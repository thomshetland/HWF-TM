#!/usr/bin/python
"""The figure and the summary table of the two-leaf experiment.

accuracy_per_epoch: train and test accuracy per epoch, one panel per dataset. Colour and marker
identify the structure; test is a solid line, train a dashed one. The distance between the two
lines of one colour is that structure's train-test gap.

The table on stdout gives each run's last epoch, and for the two-leaf machine also the flat
epoch whose train accuracy is closest to it: a smaller gap only means less overfitting if the
test accuracy is higher at the same train accuracy, not merely if training is slower.

Run from the repo root:  ./.venv/bin/python experiments/two_leaf_htm/plot_results.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SEED = 1

DATASETS = [("imdb", "IMDB"), ("amazon", "Amazon"), ("finance", "Finance")]
# The colours and markers that the two structures have in ../documents/plot_results.py.
STRUCTURES = [
    ("flat", "HTM flat", "#eb6834", "s"),
    ("two-leaf", "HTM two-leaf", "#1baf7a", "^"),
]


def load(dataset, structure):
    """Columns of one run's CSV as float arrays, or None if the run does not exist yet."""
    path = RESULTS / f"{dataset}_{structure}_seed{SEED}.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


runs = {(d, s): load(d, s) for d, _ in DATASETS for s, *_ in STRUCTURES}

fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4.2))
for ax, (dataset, title) in zip(axes, DATASETS, strict=True):
    for structure, _label, colour, marker in STRUCTURES:
        run = runs[dataset, structure]
        if run is None:
            continue
        style = dict(color=colour, marker=marker, markersize=6, linewidth=2)
        ax.plot(run["epoch"], run["test_accuracy"], **style)
        ax.plot(run["epoch"], run["train_accuracy"], linestyle="--", mfc="white", **style)
        ax.set_xticks(run["epoch"])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
axes[0].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")

# Two legend groups: colour/marker = structure, line style = split.
handles = [
    Line2D([], [], color=colour, marker=marker, linewidth=2, label=label)
    for _structure, label, colour, marker in STRUCTURES
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
FIGURES.mkdir(exist_ok=True)
for suffix in (".pdf", ".png"):
    fig.savefig(FIGURES / f"accuracy_per_epoch{suffix}", dpi=300, bbox_inches="tight")
print(f"Wrote {FIGURES / 'accuracy_per_epoch.pdf'} and .png\n")

print(
    f"{'dataset':<8} {'structure':<9} epoch  train   test    gap   best test  "
    f"words/clause  negated/clause  train time (min)"
)
for dataset, _title in DATASETS:
    for structure, *_ in STRUCTURES:
        run = runs[dataset, structure]
        if run is None:
            continue
        train, test = run["train_accuracy"][-1], run["test_accuracy"][-1]
        print(
            f"{dataset:<8} {structure:<9} {int(run['epoch'][-1]):>5}  {train:5.2f}  {test:5.2f}  "
            f"{train - test:5.2f}  {run['test_accuracy'].max():9.2f}  "
            f"{run['positive_literals'][-1]:12.2f}  {run['negated_literals'][-1]:14.2f}  "
            f"{run['train_seconds'].sum() / 60:16.1f}"
        )

print("\nThe flat epoch whose train accuracy is closest to the two-leaf machine's last epoch:")
for dataset, _title in DATASETS:
    flat, two_leaf = runs[dataset, "flat"], runs[dataset, "two-leaf"]
    if flat is None or two_leaf is None:
        continue
    target = two_leaf["train_accuracy"][-1]
    e = int(np.abs(flat["train_accuracy"] - target).argmin())
    print(
        f"{dataset:<8} two-leaf epoch {int(two_leaf['epoch'][-1])}: train {target:.2f}, "
        f"test {two_leaf['test_accuracy'][-1]:.2f}   |   flat epoch {int(flat['epoch'][e])}: "
        f"train {flat['train_accuracy'][e]:.2f}, test {flat['test_accuracy'][e]:.2f}"
    )
