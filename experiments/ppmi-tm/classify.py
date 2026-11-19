#!/usr/bin/python
"""Stage 2a of the PPMI-TM: sentiment classification with a standard TM on the word families.

For each dataset, number of labels N and label-sampling seed, up to four inputs are compared:

  bow           the benchmark's set of words: the 5000 most common words of the train texts.
  bow+families  bow plus one bit per family, "some word of this family occurs". No word loses
                its bit.
  fam+rest      the PPMI-TM: families *replace* their words. One bit per family, then one bit for
                every bow word that is in no family, so "great" and "excellent" in one family
                become one feature.
  random+rest   fam+rest with the random control: the same family sizes, filled with random words
                of similar frequency.

All inputs use the same TM hyperparameters and the same N labeled texts. Logistic regression on
the same matrices is recorded next to the TM as a reference: if it gains too, the families carry
the information and the question is only whether the TM uses it.

Run from the repo root, after build_families.py:

  ./.venv/bin/python experiments/ppmi-tm/classify.py --families families_ppmi --labels 200 1000
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path
from time import time

import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from src.common import (  # noqa: E402
    DATASETS,
    bag_of_words,
    dataset_vocabulary,
    labels,
    load_splits,
)
from src.family_readout import (  # noqa: E402
    family_features,
    permuted_families,
    read_out,
    word_family_encoding,
)

RESULTS = HERE / "results"
CONDITIONS = ("bow", "bow+families", "fam+rest", "random+rest")
# Families that fire on almost no text or almost every text are no use as features.
MIN_DF, MAX_DF = 0.005, 0.95

parser = argparse.ArgumentParser()
parser.add_argument("--families", required=True, help="results/<name>.json from build_families.py")
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
# 0 means every training label: the full-data point, comparable to experiments/classification/tm.
parser.add_argument("--labels", type=int, nargs="+", default=[200, 1000, 0])
parser.add_argument("--seeds", type=int, default=2)
parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
# The hyperparameters of experiments/classification/tm, so bow at N=all is that baseline.
parser.add_argument("--clauses", type=int, default=2000)
parser.add_argument("--T", type=int, default=1600)
parser.add_argument("--s", type=float, default=5.0)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--name", default=None, help="output name, default classify_<families>")
args = parser.parse_args()

with (RESULTS / f"{args.families}.json").open() as fh:
    stored = json.load(fh)
pre_vocabulary = {w: i for i, w in enumerate(stored["words"])}

families = read_out(stored["families"], "leaves")
random_families = permuted_families(families, len(pre_vocabulary), np.random.default_rng(0))
print(f"{len(families)} distinct families from results/{args.families}.json")


def keep_informative(F_train, F_test):
    """Drop near-constant family bits, judged on train texts only."""
    df = F_train.mean(axis=0)
    keep = np.flatnonzero((df >= MIN_DF) & (df <= MAX_DF))
    return F_train[:, keep], F_test[:, keep]


splits = load_splits()
DATA = {}
for name in args.datasets:
    train, test = splits[name]["train"], splits[name]["test"]
    classes = sorted({row["sentiment"] for row in train})
    vocabulary = dataset_vocabulary(train)

    tokens_train = [row["tokens"].split() for row in train]
    tokens_test = [row["tokens"].split() for row in test]
    # Families are word lists over the pre-training vocabulary, so they are evaluated on a
    # bag of words over that vocabulary, not the dataset's own.
    P_train = bag_of_words(tokens_train, pre_vocabulary)
    P_test = bag_of_words(tokens_test, pre_vocabulary)
    F_train, F_test = keep_informative(
        family_features(P_train, families), family_features(P_test, families)
    )

    B_train = bag_of_words(tokens_train, vocabulary)
    B_test = bag_of_words(tokens_test, vocabulary)
    DATA[name] = {
        "classes": classes,
        "Y_train": labels(train, classes),
        "Y_test": labels(test, classes),
        "bow": (B_train, B_test),
        "bow+families": (np.hstack((B_train, F_train)), np.hstack((B_test, F_test))),
    }
    print(
        f"{name}: {len(train)} train / {len(test)} test, classes {classes}, "
        f"{F_train.shape[1]} families kept for bow+families"
    )
    # Replacing: no keep_informative filter here, because dropping a family bit would drop
    # its words from the input altogether.
    for condition, fams in (("fam+rest", families), ("random+rest", random_families)):
        if condition in args.conditions:
            X_train, rest = word_family_encoding(tokens_train, fams, stored["words"], vocabulary)
            X_test, _ = word_family_encoding(tokens_test, fams, stored["words"], vocabulary)
            DATA[name][condition] = (X_train, X_test)
            print(
                f"  {condition}: {len(fams)} family bits + {len(rest)} rest words "
                f"(bow has {len(vocabulary)})"
            )


def run(job):
    name, condition, n_labels, seed = job
    data = DATA[name]
    X_train, X_test = data[condition]
    Y_train, Y_test = data["Y_train"], data["Y_test"]

    # The label sample depends on (dataset, N, seed) only, so all four conditions see the same
    # labelled reviews.
    rng = np.random.default_rng([seed, n_labels, DATASETS.index(name)])
    n = len(Y_train) if n_labels == 0 else min(n_labels, len(Y_train))
    chosen = rng.choice(len(Y_train), n, replace=False)
    X, Y = X_train[chosen], Y_train[chosen]

    # Small N needs more passes to get the same number of updates; the full-data runs get the
    # baseline's 10.
    epochs = int(np.clip(20000 // n, 10, 200))
    # pyTsetlinMachine shuffles with the global numpy RNG; each worker process has its own.
    np.random.seed(seed)
    start = time()
    tm = MultiClassTsetlinMachine(args.clauses, args.T, args.s, weighted_clauses=True)
    tm.fit(X, Y, epochs=epochs)
    tm_pred = tm.predict(X_test)
    seconds = time() - start

    lr = LogisticRegression(max_iter=2000).fit(X, Y)
    lr_pred = lr.predict(X_test)

    return {
        "dataset": name,
        "condition": condition,
        "n_labels": n,
        "seed": seed,
        "n_features": X_train.shape[1],
        "epochs": epochs,
        "tm_accuracy": float((tm_pred == Y_test).mean()),
        "tm_macro_f1": float(f1_score(Y_test, tm_pred, average="macro")),
        "lr_accuracy": float((lr_pred == Y_test).mean()),
        "lr_macro_f1": float(f1_score(Y_test, lr_pred, average="macro")),
        "tm_seconds": round(seconds, 1),
    }


jobs = [
    (name, condition, n_labels, seed)
    for name in args.datasets
    for n_labels in args.labels
    for seed in range(args.seeds)
    for condition in args.conditions
]
# Largest N first, so the slowest jobs do not end up alone at the tail.
jobs.sort(key=lambda job: -job[2])

name = args.name or f"classify_{args.families}"
out = RESULTS / f"{name}.csv"
# Rows are appended as they finish and finished jobs are skipped on a rerun, because the
# container restarts without warning and the full-label jobs take hours.
rows = []
if out.exists():
    with out.open(newline="") as fh:
        for row in csv.DictReader(fh):
            for key in ("n_labels", "seed", "n_features", "epochs"):
                row[key] = int(row[key])
            for key in ("tm_accuracy", "tm_macro_f1", "lr_accuracy", "lr_macro_f1", "tm_seconds"):
                row[key] = float(row[key])
            rows.append(row)
done = {(r["dataset"], r["condition"], r["n_labels"], r["seed"]) for r in rows}


def actual_n(name, n_labels):
    n_train = len(DATA[name]["Y_train"])
    return n_train if n_labels == 0 else min(n_labels, n_train)


jobs = [job for job in jobs if (job[0], job[1], actual_n(job[0], job[2]), job[3]) not in done]
print(f"{len(done)} jobs already in {out}, {len(jobs)} to run")
# fork, so the workers share DATA with this process instead of each rebuilding it.
with mp.get_context("fork").Pool(args.workers) as pool, out.open("a", newline="") as fh:
    writer = None
    for row in pool.imap_unordered(run, jobs):
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if not done:
                writer.writeheader()
        writer.writerow(row)
        fh.flush()
        rows.append(row)
        print(
            f"  {row['dataset']:8s} N={row['n_labels']:5d} seed {row['seed']} "
            f"{row['condition']:10s} TM {row['tm_accuracy']:.4f} LR {row['lr_accuracy']:.4f}",
            flush=True,
        )

print(f"\nWrote {out}\n")
print("Mean test accuracy over seeds (TM / LR); delta is against bow:\n")
for dataset in args.datasets:
    majority = np.bincount(DATA[dataset]["Y_test"]).max() / len(DATA[dataset]["Y_test"])
    print(f"{dataset} (majority class {majority:.3f})")
    for n_labels in sorted({r["n_labels"] for r in rows if r["dataset"] == dataset}):
        cells = {}
        present = {r["condition"] for r in rows if r["dataset"] == dataset}
        for condition in (c for c in CONDITIONS if c in present):
            group = [
                r
                for r in rows
                if r["dataset"] == dataset
                and r["n_labels"] == n_labels
                and r["condition"] == condition
            ]
            cells[condition] = (
                np.mean([r["tm_accuracy"] for r in group]),
                np.std([r["tm_accuracy"] for r in group]),
                np.mean([r["lr_accuracy"] for r in group]),
            )
        base_tm, _, base_lr = cells.get("bow", (np.nan, 0, np.nan))
        line = f"  N={n_labels:5d}"
        for condition, (tm_acc, tm_std, lr_acc) in cells.items():
            line += f"  {condition} {tm_acc:.3f}±{tm_std:.3f}/{lr_acc:.3f}"
            if condition != "bow" and "bow" in cells:
                line += f" ({tm_acc - base_tm:+.3f}/{lr_acc - base_lr:+.3f})"
        print(line)
