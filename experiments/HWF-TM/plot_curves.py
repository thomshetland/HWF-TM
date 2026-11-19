#!/usr/bin/python
"""Figure for Appendix B: learning curves of the 200-label run.

Reads results/classify_sources_N200.csv (one label draw, 2000 clauses, 100 epochs) and writes
  figures/learning_curves_N200.{pdf,png}   test accuracy per epoch, one panel per dataset

Colors follow plot_results.py: the first three slots of the dataviz reference palette for the
family sources, neutral gray for the baseline TM, and dashed lines for random controls.

Run from the repo root:  ./.venv/bin/python experiments/HWF-TM/plot_curves.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "classify_sources_N200.csv"
FIGURE = HERE / "figures" / "learning_curves_N200"
DATASETS = (("imdb", "IMDB"), ("amazon", "Amazon"), ("finance", "Finance"))

INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (representation, legend label, color, dashed)
SERIES = (
    ("bow", "TM (set of words)", "#52514e", False),
    ("ppmi", "PPMI-TM", "#2a78d6", False),
    ("ppmi-perm", "PPMI control", "#2a78d6", True),
    ("profile", "HWF-TM (profile)", "#eb6834", False),
    ("profile-perm", "HWF-TM (profile) control", "#eb6834", True),
    ("same-dataset", "HWF-TM (same-dataset)", "#1baf7a", False),
    ("same-dataset-perm", "HWF-TM (same-dataset) control", "#1baf7a", True),
)

accuracy = defaultdict(dict)
with SOURCE.open() as fh:
    for row in csv.DictReader(fh):
        if row["seed"] == "1":
            key = (row["dataset"], row["representation"])
            accuracy[key][int(row["epoch"])] = float(row["test_accuracy"])

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
for ax, (dataset, title) in zip(axes, DATASETS, strict=True):
    for rep, label, color, dashed in SERIES:
        curve = accuracy[(dataset, rep)]
        epochs = sorted(curve)
        ax.plot(
            epochs,
            [curve[e] for e in epochs],
            label=label,
            color=color,
            linestyle="--" if dashed else "-",
            linewidth=1.3,
        )
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_xlim(1, 100)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Test accuracy (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.04),
)
fig.tight_layout(rect=(0, 0.17, 1, 1))
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(FIGURE.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"Wrote {FIGURE}.pdf and .png")
