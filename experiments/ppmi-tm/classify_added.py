#!/usr/bin/python
"""Stage 2b of the PPMI-TM: the benchmark TM with the families *added* to the set of words.

Reproduces experiments/classification/tm/{imdb,amazon,finance}.py: the same splits, the same
5000-word vocabulary from the train texts, 2000 clauses, T = 1600, s = 5, weighted clauses, 10
epochs and all training labels. One bit per family ("any word of the family occurs") is appended,
so no word loses its own bit. Without --families it is the benchmark itself.

Writes one CSV row per epoch to results/added_<dataset>_<input>_seed<seed>.csv, so a killed run
keeps the epochs it finished. Run from the repo root:

  ./.venv/bin/python experiments/ppmi-tm/classify_added.py --dataset imdb --families families_ppmi
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from src.common import DATASETS, bag_of_words, dataset_vocabulary, labels, load_splits  # noqa: E402
from src.family_readout import family_features, permuted_families  # noqa: E402

RESULTS = HERE / "results"
MIN_DF, MAX_DF = 0.005, 0.95  # the same filter as classify.py

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=DATASETS)
parser.add_argument("--families", default=None, help="results/<name>.json; default no families")
parser.add_argument("--control", action="store_true", help="the random control of the families")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

splits = load_splits()
train, test = splits[args.dataset]["train"], splits[args.dataset]["test"]
classes = sorted({r["sentiment"] for r in train})
vocabulary = dataset_vocabulary(train)
tokens_train = [r["tokens"].split() for r in train]
tokens_test = [r["tokens"].split() for r in test]
X, X_test = bag_of_words(tokens_train, vocabulary), bag_of_words(tokens_test, vocabulary)
Y, Y_test = labels(train, classes), labels(test, classes)

tag = "bow"
if args.families:
    with (RESULTS / f"{args.families}.json").open() as fh:
        stored = json.load(fh)
    pre = {w: i for i, w in enumerate(stored["words"])}
    families = [f["leaves"] for f in stored["families"]]
    if args.control:
        families = permuted_families(families, len(pre), np.random.default_rng(0))
    F = family_features(bag_of_words(tokens_train, pre), families)
    F_test = family_features(bag_of_words(tokens_test, pre), families)
    df = F.mean(axis=0)
    keep = np.flatnonzero((df >= MIN_DF) & (df <= MAX_DF))
    X, X_test = np.hstack((X, F[:, keep])), np.hstack((X_test, F_test[:, keep]))
    tag = args.families + ("_control" if args.control else "")
    print(f"{len(families)} families, {len(keep)} kept after the df filter", flush=True)
print(f"{args.dataset}: {tag}, X {X.shape}, test {X_test.shape}", flush=True)

RESULTS.mkdir(parents=True, exist_ok=True)
path = RESULTS / f"added_{args.dataset}_{tag}_seed{args.seed}.csv"
np.random.seed(args.seed)
tm = MultiClassTsetlinMachine(2000, 1600, 5.0, weighted_clauses=True)
with path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["epoch", "train_accuracy", "test_accuracy", "train_seconds"])
    for epoch in range(args.epochs):
        start = time.time()
        tm.fit(X, Y, epochs=1, incremental=True)
        seconds = time.time() - start
        test_accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
        train_accuracy = 100 * (tm.predict(X) == Y).mean()
        writer.writerow(
            [epoch + 1, f"{train_accuracy:.4f}", f"{test_accuracy:.4f}", f"{seconds:.1f}"]
        )
        fh.flush()
        print(
            f"#{epoch + 1} train {train_accuracy:.2f} test {test_accuracy:.2f} ({seconds:.0f}s)",
            flush=True,
        )
