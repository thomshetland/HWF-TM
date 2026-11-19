#!/usr/bin/python
"""Figure for the report: test accuracy against model size, 1000 labels.

One panel per dataset. Each input is a line through its four clause counts (250, 500, 1000, 2000,
left to right); x is the total number of included literals in the trained model (log scale) and
y the test accuracy, both the mean over the three label draws. A point up and to the left of
another is a smaller model that is at least as accurate. Marker size grows with the clause count,
and the TM's points in the first panel are annotated with it, where there is room; the other
panels are too crowded for labels. Colors match plot_results.py.

Run from the repo root:  ./.venv/bin/python experiments/HWF-TM/plot_literals.py
Writes figures/results_literals.{pdf,png}.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "classify_sources_N1000.csv"
OUT = HERE / "figures" / "results_literals"
FINAL_EPOCH = 20
DATASETS = (("imdb", "IMDB"), ("amazon", "Amazon"), ("finance", "Finance"))
CLAUSES = (250, 500, 1000, 2000)
MARKER_SIZE = dict(zip(CLAUSES, (3.2, 4.4, 5.6, 6.8), strict=True))
# Hand-placed label offsets for the TM points of the IMDB panel: (dx, dy, ha, va).
LABEL_OFFSETS = {
    250: (-6, 0, "right", "center"),
    500: (-4, 5, "right", "bottom"),
    1000: (4, -5, "left", "top"),
    2000: (0, -6, "center", "top"),
}
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (representation, legend label, color, marker)
SERIES = (
    ("bow", "TM (set of words)", "#52514e", "o"),
    ("ppmi", "PPMI-TM", "#2a78d6", "s"),
    ("profile", "HWF-TM (profile)", "#eb6834", "D"),
    ("same-dataset", "HWF-TM (same-dataset)", "#1baf7a", "^"),
)

accuracy, literals = defaultdict(list), defaultdict(list)
with SOURCE.open() as fh:
    for row in csv.DictReader(fh):
        if int(row["epoch"]) == FINAL_EPOCH:
            key = (row["dataset"], row["representation"], int(row["clauses"]))
            accuracy[key].append(float(row["test_accuracy"]))
            literals[key].append(float(row["total_literals"]))

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
    for rep, _label, color, marker in SERIES:
        x = [np.mean(literals[(dataset, rep, c)]) for c in CLAUSES]
        y = [np.mean(accuracy[(dataset, rep, c)]) for c in CLAUSES]
        ax.plot(x, y, color=color, linewidth=1.6)
        for xi, yi, c in zip(x, y, CLAUSES, strict=True):
            ax.plot(
                xi,
                yi,
                color=color,
                marker=marker,
                markersize=MARKER_SIZE[c],
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
            )
            if rep == "bow" and dataset == "imdb":
                dx, dy, ha, va = LABEL_OFFSETS[c]
                ax.annotate(
                    str(c),
                    (xi, yi),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=7,
                    color=INK_SOFT,
                )
    ax.set_xscale("log")
    ax.set_xticks([1e4, 3e4, 1e5, 3e5], ["10k", "30k", "100k", "300k"])
    ax.minorticks_off()
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel("Total literals")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Test accuracy (%)")

# Legend handles are built by hand because the data markers are drawn point by point.
handles = [
    Line2D([], [], color=color, marker=marker, markersize=4.4, linewidth=1.6, label=label)
    for _, label, color, marker in SERIES
]
fig.legend(
    handles=handles,
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
