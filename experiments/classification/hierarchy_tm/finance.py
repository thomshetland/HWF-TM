#!/usr/bin/python
"""Sentiment classification on financial news (negative/neutral/positive)
with a flat hierarchical Tsetlin machine.

The flat structure is one leaf over all words, i.e. the same algorithm as ../tm/finance.py.

Run from the repo root:  ./.venv/bin/python experiments/classification/hierarchy_tm/finance.py
"""

import csv
import sys
from pathlib import Path
from time import time

import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as htm

# Make `src.preprocessing` importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.create_ds import build_vocabulary  # noqa: E402
from src.preprocessing.tokenizing import load_finance, tokenize_document  # noqa: E402

VOCABULARY_SIZE = 5000
TEST_FRACTION = 0.2
EPOCHS = 10
SEED = 42
HERE = Path(__file__).resolve()
CSV_PATH = HERE.parents[1] / "results" / HERE.parent.name / f"{HERE.stem}.csv"

CLASSES = ["negative", "neutral", "positive"]

rng = np.random.default_rng(SEED)

print("Loading and tokenizing...")
sentences = tokenize_document(load_finance())
sentences = [sentences[i] for i in rng.permutation(len(sentences))]

n_test = int(len(sentences) * TEST_FRACTION)
test, train = sentences[:n_test], sentences[n_test:]

# The vocabulary is fitted on training sentences only.
vocabulary = build_vocabulary([train], VOCABULARY_SIZE)


def to_bits(rows):
    """One bit per vocabulary word: is the word in the sentence?"""
    X = np.zeros((len(rows), len(vocabulary)), dtype=np.uint32)
    Y = np.zeros(len(rows), dtype=np.uint32)
    for i, row in enumerate(rows):
        for word in row["tokens"].split():
            if word in vocabulary:
                X[i, vocabulary[word]] = 1
        Y[i] = CLASSES.index(row["sentiment"])
    return X, Y


X_train, Y_train = to_bits(train)
X_test, Y_test = to_bits(test)
print(f"X_train {X_train.shape}, X_test {X_test.shape}")

# The library crashes on depth 1, so the flat machine gets a single-alternative OR on top.
tm = htm.MultiClassTsetlinMachine(
    2000,
    1600,
    5.0,
    weighted_clauses=True,
    hierarchy_structure=((htm.AND_GROUP, X_train.shape[1]), (htm.OR_ALTERNATIVES, 1)),
    boost_true_positive_feedback=1,
    number_of_state_bits=8,
    append_negated=True,
)

print(f"\nAccuracy over {EPOCHS} epochs:\n")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with CSV_PATH.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["epoch", "train_accuracy", "test_accuracy", "train_seconds", "test_seconds"])

    for i in range(EPOCHS):
        start_training = time()
        tm.fit(X_train, Y_train)  # no epochs argument: one call is one epoch
        stop_training = time()

        start_testing = time()
        result = 100 * (tm.predict(X_test) == Y_test).mean()
        stop_testing = time()
        train_acc = 100 * (tm.predict(X_train) == Y_train).mean()

        writer.writerow(
            [
                i + 1,
                f"{train_acc:.4f}",
                f"{result:.4f}",
                f"{stop_training - start_training:.2f}",
                f"{stop_testing - start_testing:.2f}",
            ]
        )
        fh.flush()  # so a long run can be inspected mid-way

        print(
            f"#{i + 1} Train: {train_acc:.2f}% "
            f"Accuracy: {result:.2f}% "
            f"Training: {stop_training - start_training:.2f}s "
            f"Testing: {stop_testing - start_testing:.2f}s",
            flush=True,
        )

print(f"\nWrote per-epoch metrics to {CSV_PATH}")
