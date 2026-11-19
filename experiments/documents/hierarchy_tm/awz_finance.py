#!/usr/bin/python
"""The Amazon vs finance news pair task of ../tm/awz_finance.py, with the hierarchical TM.

Y=1 if both texts come from the same dataset (Amazon reviews or finance news), Y=0 if not.

The goal is to reproduce the vanilla machine with the hierarchical one, as ./imdb_awz.py does for
IMDB vs Amazon. Two structures:

  flat      one leaf over [A | B]. The same algorithm as the vanilla machine, so any difference
            from ../tm/awz_finance.py comes from the library, not from the hierarchy.
  two-leaf  one leaf per text, AND-ed at the root. The clause is still one conjunction over
            [A | B], but each leaf flips its own coin for whether it gets feedback
            (update_component_hierarchy), where a vanilla clause flips one for all its literals.

Both give an untrained clause a vote of exactly 1, as in the vanilla machine, so T carries over.

The pairs are identical to the vanilla run: the data RNG is seeded with 42 and called in the
same order. --seed only changes the machine's own randomness.

Run from the repo root:
  ./.venv/bin/python experiments/documents/hierarchy_tm/awz_finance.py --structure two-leaf
      --seed 1
"""

import argparse
import csv
import sys
from pathlib import Path
from time import time

import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as htm

# Make `src.preprocessing` importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.create_ds import build_vocabulary, create_ds  # noqa: E402
from src.preprocessing.tokenizing import load_amazon, load_finance, tokenize_document  # noqa: E402

VOCABULARY_SIZE = 5000  # words per review -> 2 * VOCABULARY_SIZE features
TRAIN_PAIRS = 20000
TEST_PAIRS = 5000
TEST_FRACTION = 0.2
EPOCHS = 20
DATA_SEED = 42

CLAUSES = 2000
T = 1600
S = 5.0

STRUCTURES = {
    # The library crashes on depth 1, so the flat machine gets a single-alternative OR on top.
    "flat": ((htm.AND_GROUP, 2 * VOCABULARY_SIZE), (htm.OR_ALTERNATIVES, 1)),
    "two-leaf": ((htm.AND_GROUP, VOCABULARY_SIZE), (htm.AND_GROUP, 2)),
}

parser = argparse.ArgumentParser()
parser.add_argument("--structure", choices=STRUCTURES, default="two-leaf")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "hierarchy_tm"
    / f"awz_finance_{args.structure}_seed{args.seed}.csv"
)

rng = np.random.default_rng(DATA_SEED)

# tokenize documents

print("Loading and tokenizing...")
finance = tokenize_document(load_finance(), save_document=False)
# Amazon has 3.6M reviews, so a random sample the size of finance news is used; the two
# datasets are then equally large and neither dominates the vocabulary.
amazon = tokenize_document(load_amazon(len(finance), rng), save_document=False)

# Shuffle, then split texts into train/test BEFORE pairing, so no text appears in both.
amazon = [amazon[i] for i in rng.permutation(len(amazon))]
finance = [finance[i] for i in rng.permutation(len(finance))]

n = len(finance)
n_test = int(n * TEST_FRACTION)
amazon_test, amazon_train = amazon[:n_test], amazon[n_test:]
finance_test, finance_train = finance[:n_test], finance[n_test:]
print(f"{n} texts per dataset: {n - n_test} train, {n_test} test")

# make set of words vocabulary

vocabulary = build_vocabulary([amazon_train, finance_train], VOCABULARY_SIZE)

# make training ds

X_train, Y_train = create_ds(amazon_train, finance_train, vocabulary, TRAIN_PAIRS, rng)

# make test ds

X_test, Y_test = create_ds(amazon_test, finance_test, vocabulary, TEST_PAIRS, rng)
print(f"X_train {X_train.shape}, X_test {X_test.shape}, positives {Y_train.mean():.2f}")

# MultiClassTsetlinMachine.fit shuffles and subsamples with the global, unseeded np.random.
np.random.seed(args.seed)

# boost_true_positive_feedback=1 is the vanilla default; the hierarchy demos use 0.
tm = htm.MultiClassTsetlinMachine(
    CLAUSES,
    T,
    S,
    weighted_clauses=True,
    hierarchy_structure=STRUCTURES[args.structure],
    boost_true_positive_feedback=1,
    number_of_state_bits=8,
    append_negated=True,
    seed=args.seed,
)
print(f"structure {args.structure}: {STRUCTURES[args.structure]}, seed {args.seed}")


def prf(y_true, y_pred):
    """Precision, recall and F1 for the positive class (same dataset, Y=1)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"\nAccuracy over {EPOCHS} epochs:\n")
with CSV_PATH.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(
        [
            "epoch",
            "train_accuracy",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "train_seconds",
            "test_seconds",
        ]
    )

    for i in range(EPOCHS):
        # fit() takes no epochs argument and is always incremental: one call is one epoch.
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        test_pred = tm.predict(X_test)
        stop_testing = time()
        train_pred = tm.predict(X_train)

        train_acc = 100 * (train_pred == Y_train).mean()
        result = 100 * (test_pred == Y_test).mean()
        precision, recall, f1 = prf(Y_test, test_pred)

        writer.writerow(
            [
                i + 1,
                f"{train_acc:.4f}",
                f"{result:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{stop_training - start_training:.2f}",
                f"{stop_testing - start_testing:.2f}",
            ]
        )
        # Flush per epoch so a long run can be inspected (or killed) mid-way.
        fh.flush()

        print(
            f"#{i + 1} Train: {train_acc:.2f}% "
            f"Accuracy: {result:.2f}% "
            f"F1: {f1:.4f} "
            f"Training: {stop_training - start_training:.2f}s "
            f"Testing: {stop_testing - start_testing:.2f}s",
            flush=True,
        )

print(f"\nWrote per-epoch metrics to {CSV_PATH}")
