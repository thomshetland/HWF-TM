#!/usr/bin/python
"""Figure for the report: test accuracy against the number of clauses, 1000 labels.

One panel per dataset, one line per input (mean over the label draws, error bars one standard
deviation). The random controls are not drawn here; they have their own figure
(plot_control.py). Colors are the first three slots of the dataviz reference palette, which are
documented to pass every colorblind check together; the baseline TM is a neutral gray because it
is the reference, not a category. Markers differ per series so identity never rests on color.

Run from the repo root:  ./.venv/bin/python experiments/HWF-TM/plot_results.py
Writes figures/results_clauses.{pdf,png}.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "classify_sources_N1000.csv"
OUT = HERE / "figures" / "results_clauses"
FINAL_EPOCH = 20
DATASETS = (("imdb", "IMDB"), ("amazon", "Amazon"), ("finance", "Finance"))
CLAUSES = (250, 500, 1000, 2000)

INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (representation, legend label, color, marker)
SERIES = (
    ("bow", "TM (set of words)", "#52514e", "o"),
    ("ppmi", "PPMI-TM", "#2a78d6", "s"),
    ("profile", "HWF-TM (profile)", "#eb6834", "D"),
    ("same-dataset", "HWF-TM (same-dataset)", "#1baf7a", "^"),
)

accuracy = defaultdict(list)
with SOURCE.open() as fh:
    for row in csv.DictReader(fh):
        if int(row["epoch"]) == FINAL_EPOCH:
            key = (row["dataset"], row["representation"], int(row["clauses"]))
            accuracy[key].append(float(row["test_accuracy"]))

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
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, (dataset, title) in zip(axes, DATASETS, strict=True):
    for rep, label, color, marker in SERIES:
        values = [accuracy[(dataset, rep, c)] for c in CLAUSES]
        mean = [np.mean(v) for v in values]
        std = [np.std(v) for v in values]
        ax.errorbar(
            CLAUSES,
            mean,
            yerr=std,
            label=label,
            color=color,
            linewidth=1.6,
            marker=marker,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            elinewidth=0.8,
            capsize=0,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(CLAUSES, [str(c) for c in CLAUSES])
    ax.minorticks_off()
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel("Clauses")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Test accuracy (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=(0, 0.1, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}.pdf and .png")
