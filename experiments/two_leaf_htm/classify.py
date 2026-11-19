#!/usr/bin/python
"""Sentiment classification with a two-leaf hierarchical Tsetlin machine, next to the flat one.

The benchmark (../classification) classifies a text with a flat HTM: one leaf over all 5000 words.
Here the vocabulary is divided between two leaves that are AND-ed at the root:

  flat      ((AND_GROUP, 5000), (OR_ALTERNATIVES, 1))   one leaf over all words
  two-leaf  ((AND_GROUP, 2500), (AND_GROUP, 2))         one leaf per half of the vocabulary

Both encode one conjunction over the same 5000 words, and both give an untrained clause a vote
of exactly 1, so T carries over from the benchmark. What differs is the feedback: every leaf
draws its own random number for whether it is updated (update_component_hierarchy), where a flat
clause draws one for all its literals. The question is whether that makes the machine memorise
the training texts less, i.e. whether the gap between train and test accuracy shrinks.

A text has no natural two parts, so the words are dealt to the leaves at random. The vocabulary
is in document-frequency order, and without the shuffle one leaf would hold all the common words
and the other all the rare ones. Both structures get the same shuffled columns.

Everything else is the benchmark: the same splits and vocabulary, 2000 clauses, T = 1600, s = 5,
weighted clauses and 10 epochs. The machine is checkpointed after every epoch, because the
container restarts without warning; a rerun of the same command resumes, and a finished run is
only reported again. Delete its file in models/ to train from scratch.

Run from the repo root:
  ./.venv/bin/python experiments/two_leaf_htm/classify.py --dataset imdb --structure two-leaf
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path
from time import time

import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as htm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from src.common import (  # noqa: E402
    DATA_SEED,
    DATASETS,
    bag_of_words,
    dataset_vocabulary,
    labels,
    load_splits,
    tokens,
)

CLASSES = {
    "imdb": ["negative", "positive"],
    "amazon": ["negative", "positive"],
    "finance": ["negative", "neutral", "positive"],
}

CLAUSES = 2000
T = 1600
S = 5.0

STRUCTURES = {
    # The library crashes on depth 1, so the flat machine gets a single-alternative OR on top.
    "flat": lambda n_features: ((htm.AND_GROUP, n_features), (htm.OR_ALTERNATIVES, 1)),
    # A GROUP level partitions the input: leaf 0 reads the first half of the columns, leaf 1
    # the second half.
    "two-leaf": lambda n_features: ((htm.AND_GROUP, n_features // 2), (htm.AND_GROUP, 2)),
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=DATASETS, required=True)
parser.add_argument("--structure", choices=STRUCTURES, default="two-leaf")
parser.add_argument("--seed", type=int, default=1, help="the machine's randomness, not the data's")
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

run = f"{args.dataset}_{args.structure}_seed{args.seed}"
csv_path = HERE / "results" / f"{run}.csv"
checkpoint = HERE / "models" / f"{run}.pkl"

# ---------------------------------------------------------------- data

split = load_splits()[args.dataset]
vocabulary = dataset_vocabulary(split["train"])  # fitted on training texts only
if len(vocabulary) % 2:
    sys.exit(f"the vocabulary has {len(vocabulary)} words; two leaves need an even number")

# The same shuffle for every run and structure, so a leaf always holds the same words.
columns = np.random.default_rng(DATA_SEED).permutation(len(vocabulary))


def to_bits(rows):
    X = bag_of_words([tokens(row) for row in rows], vocabulary)
    return X[:, columns], labels(rows, CLASSES[args.dataset])


X_train, Y_train = to_bits(split["train"])
X_test, Y_test = to_bits(split["test"])
print(f"{run}: X_train {X_train.shape}, X_test {X_test.shape}")

# ---------------------------------------------------------------- machine

structure = STRUCTURES[args.structure](X_train.shape[1])
print(f"structure {structure}")

tm = htm.MultiClassTsetlinMachine(
    CLAUSES,
    T,
    S,
    weighted_clauses=True,
    hierarchy_structure=structure,
    boost_true_positive_feedback=1,  # the standard TM's default, as in the benchmark
    number_of_state_bits=8,
    append_negated=True,
    seed=args.seed,
)


def write_checkpoint(history):
    """Written to a temporary file and moved into place, so a restart never finds half a file."""
    state = {
        "config": vars(args),
        "machine": tm.save(),
        "history": history,
        "numpy_state": np.random.get_state(),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(state, fh)
    tmp.replace(checkpoint)


def write_csv(history):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


history = []
if checkpoint.exists():
    with checkpoint.open("rb") as fh:
        saved = pickle.load(fh)
    if saved["config"] | {"epochs": args.epochs} != vars(args):
        sys.exit(f"{checkpoint} was written with a different config; delete it to start over")
    # load() rebuilds the per-class machines without a seed, so the GPU's random stream starts
    # anew: a resumed run is a valid continuation, but not bit-identical to an unbroken one.
    tm.load(saved["machine"])
    np.random.set_state(saved["numpy_state"])
    history = saved["history"]
    write_csv(history)  # the checkpoint is the record; a restart may have cut the CSV short
    print(f"resumed from {checkpoint.name} after epoch {len(history)}")
else:
    # MultiClassTsetlinMachine.fit shuffles and subsamples with the global, unseeded np.random.
    np.random.seed(args.seed)


def accuracy(X, Y):
    return 100 * float((tm.predict(X) == Y).mean())


def literals_per_clause():
    """Mean number of included words and included negated words per clause, over all classes.

    A clause that copies training texts fills up with literals, so this is the direct measure
    of memorisation. get_literals() is (clauses, leaves, literals per leaf), and within a leaf
    the first half are the words and the second half their negations (checked on the GPU with a
    planted rule in each leaf).
    """
    positive, negated = [], []
    for class_machine in tm.tms:
        included = class_machine.get_literals()
        half = included.shape[2] // 2
        positive.append(included[:, :, :half].sum(axis=(1, 2)).mean())
        negated.append(included[:, :, half:].sum(axis=(1, 2)).mean())
    return float(np.mean(positive)), float(np.mean(negated))


# ---------------------------------------------------------------- training

for epoch in range(len(history) + 1, args.epochs + 1):
    start = time()
    tm.fit(X_train, Y_train)  # no epochs argument: one call is one epoch
    train_seconds = time() - start

    start = time()
    test_accuracy = accuracy(X_test, Y_test)
    test_seconds = time() - start
    train_accuracy = accuracy(X_train, Y_train)
    positive, negated = literals_per_clause()

    history.append(
        {
            "epoch": epoch,
            "train_accuracy": f"{train_accuracy:.4f}",
            "test_accuracy": f"{test_accuracy:.4f}",
            "positive_literals": f"{positive:.2f}",
            "negated_literals": f"{negated:.2f}",
            "train_seconds": f"{train_seconds:.2f}",
            "test_seconds": f"{test_seconds:.2f}",
        }
    )
    write_checkpoint(history)
    write_csv(history)
    print(
        f"#{epoch} train {train_accuracy:.2f}% test {test_accuracy:.2f}% "
        f"literals per clause: {positive:.1f} words, {negated:.1f} negated "
        f"({train_seconds:.0f}s)",
        flush=True,
    )

best = max(history, key=lambda row: float(row["test_accuracy"]))
print(f"\nbest test accuracy {best['test_accuracy']}% at epoch {best['epoch']}; wrote {csv_path}")
