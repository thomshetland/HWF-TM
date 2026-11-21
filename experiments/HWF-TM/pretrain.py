#!/usr/bin/python
"""Stage 1 of the HWF-TM: self-supervised pretraining of a hierarchical TM on a pair task.

No sentiment label is used. Two spans A and B are shown together and the machine predicts whether
they belong together. There are two pretext tasks:

  --task dataset   two reviews, Y=1 if both come from the same corpus
                   (common.sample_dataset_pairs).
  --task profile   two usage profiles of words, Y=1 if both describe the same word
                   (common.sample_profile_pairs).

The structure, leaf first:

  (AND_GROUP, V)          a leaf: a conjunction of words of one span
  (OR_ALTERNATIVES, m)    m alternative leaves over the same span, OR-ed: one word family
  (AND_GROUP, 2)          span A and span B, each over its own V columns

so every clause reads (group over A) AND (group over B). After training, every OR group is
written out with its leaves; families.py and export_families.py turn those groups into families.

The defaults are the same-dataset recipe, with shared words removed. Each choice has a reason:
  s = 100                 an alternative that loses its last word is vacuously true and makes its
                          whole group true; at s = 5 or 30 this happens faster than it is repaired
                          and the machine sits at exactly 0.5000.
  m = 12                  a group needs about twice as many alternatives as its family has words,
                          because alternatives duplicate (coupon collector).
  T = 128                 an untrained clause votes m * m = 144, not 1, so T cannot be taken from
                          the flat machine; it was swept with m.
  append_negated=False    a family means "any of these words"; NOT(word) is true on almost every
                          sparse text and would make a family bit constant.
  min/max df 1 % / 50 %   rare words become dead literals, very common ones ride along in every
                          alternative (common.pretext_vocabulary).

Writes results/pretrain_<name>.json with the config, the per-epoch history, the overlap baseline
and every OR group. Run from the repo root (see run.sh for both pretexts):

  ./.venv/bin/python experiments/HWF-TM/pretrain.py --name same-dataset
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from time import time

import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as htm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from src.common import (  # noqa: E402
    MIN_TOKENS,
    PRETRAIN_DATASETS,
    load_splits,
    overlap_baseline,
    pair_matrix,
    pretext_vocabulary,
    remove_shared,
    sample_dataset_pairs,
    sample_profile_pairs,
    tokens,
)

RESULTS = HERE / "results"
MODELS = HERE / "models"

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="same-dataset")
parser.add_argument("--task", choices=("dataset", "profile"), default="dataset")
parser.add_argument("--remove-shared", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--vocabulary", type=int, default=5000, help="cap after the df filter")
parser.add_argument("--min-df", type=float, default=0.01, help="share of reviews a word must reach")
parser.add_argument("--max-df", type=float, default=0.5, help="share of reviews a word may reach")
parser.add_argument("--sample", type=int, default=100, help="profile: occurrences per profile")
parser.add_argument("--min-count", type=int, default=3, help="profile: word threshold")
parser.add_argument("--window", type=int, default=5, help="profile: context words per side")
parser.add_argument("--alternatives", type=int, default=12, help="m, leaves per OR group")
parser.add_argument("--clauses", type=int, default=200)
parser.add_argument("--T", type=int, default=128)
parser.add_argument("--s", type=float, default=100.0)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--train-pairs", type=int, default=20000)
parser.add_argument("--val-pairs", type=int, default=2000)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

# Checkpoint after every epoch: the container restarts without warning and kills every process.
# A rerun with the same --name resumes. The pairs are re-sampled identically from --seed, so only
# the machine, the history, the generator state and the word order are stored.
checkpoint = MODELS / f"pretrain_{args.name}.pkl"
saved = None
if checkpoint.exists():
    with checkpoint.open("rb") as fh:
        saved = pickle.load(fh)
    if saved["config"] | {"epochs": args.epochs} != vars(args):
        sys.exit(f"{checkpoint} was written with a different config; delete it to start over")

rng = np.random.default_rng(args.seed)
splits = load_splits()

# Only *train* reviews are used, and 10 % of them are held out for the validation pairs that
# choose T, so no downstream test review is ever seen here.
pool, val_pool = {}, {}
for name in PRETRAIN_DATASETS:
    reviews = [r for r in splits[name]["train"] if len(tokens(r)) >= MIN_TOKENS]
    reviews = [reviews[i] for i in rng.permutation(len(reviews))]
    n_val = len(reviews) // 10
    val_pool[name], pool[name] = reviews[:n_val], reviews[n_val:]

vocabulary = pretext_vocabulary(
    [r for name in pool for r in pool[name]], args.vocabulary, args.min_df, args.max_df
)
if saved:
    # The automata are tied to the columns, so a resumed machine must keep its word order.
    vocabulary = {word: i for i, word in enumerate(saved["words"])}
words = sorted(vocabulary, key=vocabulary.get)
V = len(vocabulary)
print(f"vocabulary {V} words, review df in [{args.min_df}, {args.max_df}]")

if args.task == "profile":
    train_pairs = sample_profile_pairs(
        pool, vocabulary, args.train_pairs, rng, args.sample, args.window, args.min_count
    )
    val_pairs = sample_profile_pairs(
        val_pool, vocabulary, args.val_pairs, rng, args.sample, args.window, args.min_count
    )
else:
    train_pairs = sample_dataset_pairs(pool, args.train_pairs, rng)
    val_pairs = sample_dataset_pairs(val_pool, args.val_pairs, rng)
if args.remove_shared:
    train_pairs, val_pairs = remove_shared(train_pairs), remove_shared(val_pairs)
if args.mirror:
    # Every pair also as (B, A), so a family is as likely to form on either side of a clause.
    train_pairs = train_pairs + [(b, a, y) for a, b, y in train_pairs]
X_train, Y_train = pair_matrix(train_pairs, vocabulary)
X_val, Y_val = pair_matrix(val_pairs, vocabulary)
overlap_acc, overlap_k = overlap_baseline(X_train, Y_train, X_val, Y_val)
print(f"pairs: train {X_train.shape}, val {X_val.shape}, positives {Y_train.mean():.2f}")
print(f"overlap baseline (share >= {overlap_k} words): {overlap_acc:.4f}")

m = args.alternatives
structure = ((htm.AND_GROUP, V), (htm.OR_ALTERNATIVES, m), (htm.AND_GROUP, 2))
print(f"structure {structure}, initial vote per clause {m * m}, T {args.T}")

tm = htm.TsetlinMachine(
    args.clauses,
    args.T,
    args.s,
    hierarchy_structure=structure,
    boost_true_positive_feedback=0,  # what the upstream hierarchy demos use
    number_of_state_bits=8,
    append_negated=False,
    seed=args.seed,
)

history = []
if saved:
    tm.load(saved["machine"])
    history = saved["history"]
    rng.bit_generator.state = saved["rng"]
    print(f"resumed from {checkpoint} after epoch {len(history)}")

for epoch in range(len(history), args.epochs):
    order = rng.permutation(len(Y_train))
    start = time()
    # fit() takes no epochs argument and does not shuffle: one call is one pass in the given order.
    tm.fit(X_train[order], Y_train[order])
    train_seconds = time() - start
    val_acc = float((tm.predict(X_val) == Y_val).mean())
    history.append({"epoch": epoch + 1, "val_accuracy": val_acc, "train_seconds": train_seconds})
    print(f"#{epoch + 1} val {val_acc:.4f} ({train_seconds:.0f}s)", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        state = {"config": vars(args), "machine": tm.save(), "history": history, "words": words}
        pickle.dump(state | {"rng": rng.bit_generator.state}, fh)
    tmp.replace(checkpoint)


def extract_groups(tm):
    """Every OR group of every clause, with its leaves written as word lists.

    Leaves are numbered with the lowest level varying fastest, so leaf j of a clause is
    alternative j % m of span j // m (checked with map_ta_id_to_feature_id). Empty leaves are
    counted, not stored: an empty leaf is vacuously true, the failure mode that keeps a group true
    on every input, so their number is the main health check of a run.
    """
    literals = tm.get_literals()  # (clauses, 2 * m, V); no negated half, append_negated=False
    weights = tm.save()["clause_weights"]
    groups = []
    for c in range(args.clauses):
        for span in range(2):
            leaves = [np.flatnonzero(literals[c, span * m + a]) for a in range(m)]
            groups.append(
                {
                    "clause": c,
                    "span": "AB"[span],
                    # Positive clauses vote "belong together", negative ones "do not".
                    "polarity": int(np.sign(weights[c])),
                    "leaves": [[words[k] for k in leaf] for leaf in leaves if len(leaf)],
                    "empty_leaves": int(sum(len(leaf) == 0 for leaf in leaves)),
                }
            )
    return groups


groups = extract_groups(tm)
empty_per_group = float(np.mean([g["empty_leaves"] for g in groups]))
RESULTS.mkdir(parents=True, exist_ok=True)
out = RESULTS / f"pretrain_{args.name}.json"
with out.open("w") as fh:
    json.dump(
        {
            "config": vars(args),
            "structure": [list(level) for level in structure],
            "overlap_baseline": overlap_acc,
            "history": history,
            "empty_per_group": empty_per_group,
            "words": words,
            "groups": groups,
        },
        fh,
    )
print(f"\nWrote {out}")
