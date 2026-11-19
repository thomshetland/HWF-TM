#!/usr/bin/python
"""Figure for the report: do the HWF-TM families beat their random control?

For each dataset, the difference in test accuracy between the TM with the learned families and the
TM with their random control (same family sizes, words replaced by random words of similar
frequency). With 1000 labels the difference is taken per label draw, since both inputs were trained
on the same draws, and shown as mean and standard deviation over the three draws; with all labels it
is a single run. Zero means the grouping carries nothing beyond merging words at all.

Run from the repo root:  ./.venv/bin/python experiments/HWF-TM/plot_control.py
Writes figures/control_difference.{pdf,png}.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "figures" / "control_difference"
DATASETS = (("imdb", "IMDB"), ("amazon", "Amazon"), ("finance", "Finance"))
CLAUSES = (250, 500, 1000, 2000)
SEEDS = ("1", "2", "3")
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (source, legend label, color, x offset); same hues as plot_results.py
SOURCES = (
    ("profile", "HWF-TM (profile)", "#eb6834", -0.12),
    ("same-dataset", "HWF-TM (same-dataset)", "#1baf7a", 0.12),
)
X_ALL = 4.6  # position of the all-labels point, set apart from the clause sweep


def final_accuracy(path, epoch, clauses=None):
    out = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if int(row["epoch"]) == epoch and (clauses is None or row["clauses"] == str(clauses)):
                out[(row["dataset"], row["representation"], int(row["clauses"]), row["seed"])] = (
                    float(row["test_accuracy"])
                )
    return out


sweep = final_accuracy(RESULTS / "classify_sources_N1000.csv", 20)
full = final_accuracy(RESULTS / "classify_sources_full.csv", 10)
# The same-dataset source was trained at all labels in its own run, under other names.
renamed = {"hwf": "same-dataset", "permuted": "same-dataset-perm"}
for (dataset, rep, clauses, seed), acc in final_accuracy(
    RESULTS / "classify_same-dataset_full.csv", 10, clauses=2000
).items():
    if rep in renamed:
        full[(dataset, renamed[rep], clauses, seed)] = acc

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
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
for ax, (dataset, title) in zip(axes, DATASETS, strict=True):
    ax.axhspan(-1, 1, color=GRID, alpha=0.6, linewidth=0)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9)
    ax.axvline((3 + X_ALL) / 2, color=INK_SOFT, linewidth=0.6, linestyle=":")
    for source, label, color, offset in SOURCES:
        diffs = [
            [
                sweep[(dataset, source, c, s)] - sweep[(dataset, f"{source}-perm", c, s)]
                for s in SEEDS
            ]
            for c in CLAUSES
        ]
        ax.errorbar(
            np.arange(len(CLAUSES)) + offset,
            [np.mean(d) for d in diffs],
            yerr=[np.std(d) for d in diffs],
            color=color,
            linestyle="none",
            marker="o",
            markersize=5,
            elinewidth=1,
            capsize=0,
            label=label,
        )
        full_seed = next(k[3] for k in full if k[0] == dataset and k[1] == source)
        ax.plot(
            X_ALL + offset,
            full[(dataset, source, 2000, full_seed)]
            - full[(dataset, f"{source}-perm", 2000, full_seed)],
            color=color,
            marker="D",
            markersize=5,
            linestyle="none",
        )
    ax.set_xticks([*range(len(CLAUSES)), X_ALL], [*map(str, CLAUSES), "all\nlabels"])
    ax.set_xlim(-0.5, X_ALL + 0.5)
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel("Clauses (1000 labels)")
axes[0].set_ylabel("Learned $-$ control (%)")

handles, labels = axes[0].get_legend_handles_labels()
handles.append(Patch(color=GRID, alpha=0.6, label="$\\pm 1\\%$"))
labels.append("$\\pm 1\\%$")
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.03),
)
fig.tight_layout(rect=(0, 0.1, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}.pdf and .png")
