#!/usr/bin/python
"""Self-supervised pair task: are two texts from the same dataset?

Y=1 if both texts come from the same dataset (Amazon product reviews or finance news), Y=0 if
not. The labels come for free from which file a text was read from.

Run from the repo root:  ./.venv/bin/python experiments/documents/tm/awz_finance.py
"""

import csv
import sys
from pathlib import Path
from time import time

import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

# Make `src.preprocessing` importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.create_ds import build_vocabulary, create_ds  # noqa: E402
from src.preprocessing.tokenizing import load_amazon, load_finance, tokenize_document  # noqa: E402

VOCABULARY_SIZE = 5000  # words per review -> 2 * VOCABULARY_SIZE features
TRAIN_PAIRS = 20000
TEST_PAIRS = 5000
TEST_FRACTION = 0.2
EPOCHS = 20
SEED = 42
CSV_PATH = Path(__file__).resolve().parents[1] / "results" / "tm" / "awz_finance.csv"

rng = np.random.default_rng(SEED)

# tokenize documents

print("Loading and tokenizing...")
finance = tokenize_document(load_finance())
# Amazon has 3.6M reviews, so a random sample the size of finance news is used; the two
# datasets are then equally large and neither dominates the vocabulary.
amazon = tokenize_document(load_amazon(len(finance), rng))

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

tm = MultiClassTsetlinMachine(2000, 1600, 5.0, weighted_clauses=True)


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
        start_training = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
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
            f"Testing: {stop_testing - start_testing:.2f}s"
        )

print(f"\nWrote per-epoch metrics to {CSV_PATH}")
