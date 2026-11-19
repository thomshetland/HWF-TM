#!/usr/bin/python
"""Pretrain the two-slot HTM on a pretext built so that an OR group must be a word family.

diagnose.py shows why the HWF-TM pretexts give poor families: an OR group collects whatever is
interchangeable evidence for the pretext label. In the same-dataset task that is "any IMDB word",
and the profile task is solved by the same word on both sides, so a group is a bundle of
unrelated matchers. This pretext is designed so that neither shortcut exists:

  input    a word pair (a, b) as two one-hot slots [a | b], with a != b.
  Y = 1    a and b are neighbours: one is among the other's k nearest in a PPMI embedding.
  Y = 0    b is a word of about the same frequency as a neighbour would be, and neither word is
           among the other's 5k nearest.

The relation is symmetric and every pair is used in both orders, so a family can form in either
slot. Validation measures generalisation to relations the machine has not seen: every unordered
word pair is assigned to training or validation before sampling, so no validation pair, in
either order, occurs in training. A held-out neighbour pair can only be recognised through
families, from the other neighbours of a and b. With V words and k neighbours there are only
about V * k positive pairs, so without this split every validation positive is a training pair
and a lookup table scores 100 %.

The PPMI embedding is the one of ../ppmi-tm (document co-occurrence on the unlabelled IMDB and
Amazon train reviews, 100-d SVD), so the label is still label-free. A clause (a in G) AND (b in H)
can only vote "related" correctly if G and H hold words that are related to each other, and no
single-word overlap can stand in for that. Negatives are frequency-matched, so a clause cannot
work from word frequency either.

The machine is the HWF-TM's two-slot structure, on library 0.3.1 like every stored HTM run:

  (AND_GROUP, V) -> (OR_ALTERNATIVES, m) -> (AND_GROUP, 2)

  s = 100               as in the HWF-TM: an empty alternative is vacuously true and keeps its
                        whole group true, and at low s alternatives empty faster than they fill.
                        On eight planted families s = 30 reaches 0.74 and s = 100 0.88
                        (planted.py).
  append_negated=False  a family means "any of these words".

The new options of library 0.3.5 are not used. literal_budget = 1 gives bit-identical results on
planted families, and binary_inference lowers accuracy there from 0.96 to 0.80 (planted.py).

Checkpointed after every epoch; rerunning the same command resumes. Writes
results/pretrain_<name>.json (the HWF-TM format: config, history, words, OR groups) and
results/families_<name>.json (the format of ../HWF-TM/export_families.py, so
../HWF-TM/classify.py can use the families downstream). Run from the repo root:

  ./.venv/bin/python experiments/or_families/pretrain.py --name ppmi-pairs_V96 --vocabulary 96
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
sys.path.insert(0, str(HERE.parent / "HWF-TM"))

from families import word_families  # noqa: E402
from src.common import bag_of_words, load_splits, tokens  # noqa: E402

PPMI_WORDS = HERE.parent / "ppmi-tm" / "results" / "families_ppmi.json"
RESULTS, MODELS = HERE / "results", HERE / "models"

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--vocabulary", type=int, default=300, help="most frequent PPMI words")
parser.add_argument("--neighbours", type=int, default=10, help="k of the k-nearest neighbours")
parser.add_argument("--alternatives", type=int, default=16, help="m, alternatives per OR group")
parser.add_argument("--clauses", type=int, default=512)
parser.add_argument("--T", type=int, default=16)
parser.add_argument("--s", type=float, default=100.0)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--train-pairs", type=int, default=40000)
parser.add_argument("--val-pairs", type=int, default=1000)
parser.add_argument("--val-fraction", type=float, default=0.2, help="of the unordered pairs")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

checkpoint = MODELS / f"pretrain_{args.name}.pkl"
saved = None
if checkpoint.exists():
    with checkpoint.open("rb") as fh:
        saved = pickle.load(fh)
    if saved["config"] | {"epochs": args.epochs} != vars(args):
        sys.exit(f"{checkpoint} was written with a different config; delete it to start over")

# ---------------------------------------------------------------- the PPMI neighbours


def ppmi_neighbours(words, k):
    """Each word's k nearest neighbours by cosine in the PPMI embedding of ../ppmi-tm.

    The same construction as ppmi-tm/build_families.py with --no-balance (the stored run):
    c[u, v] = number of train reviews containing both, PPMI with a zero diagonal, a 100-d SVD.
    """
    splits = load_splits()
    texts = [tokens(row) for name in ("imdb", "amazon") for row in splits[name]["train"]]
    X = bag_of_words(texts, {w: i for i, w in enumerate(words)}).astype(np.float32)
    co = X.T @ X
    total = co.sum()
    p = co.sum(axis=1) / total
    ppmi = np.maximum(np.log((co / total) / (p[:, None] * p[None, :] + 1e-12) + 1e-12), 0)
    ppmi[np.arange(len(words)), np.arange(len(words))] = 0
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    embedding = U[:, :100] * S[:100]
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-9
    cosine = embedding @ embedding.T
    np.fill_diagonal(cosine, -np.inf)
    ranked = np.argsort(-cosine, axis=1)
    return ranked[:, :k], ranked[:, : 5 * k]


with PPMI_WORDS.open() as fh:
    words = json.load(fh)["words"][: args.vocabulary]  # document-frequency order
V = len(words)
neighbours, near = ppmi_neighbours(words, args.neighbours)
print(f"vocabulary {V} words, {args.neighbours} PPMI neighbours each")


# Symmetric relations: no ordered pair gets both labels once the pairs are mirrored.
related = np.zeros((V, V), dtype=bool)
related[np.arange(V)[:, None], neighbours] = True
related |= related.T
close = np.zeros((V, V), dtype=bool)
close[np.arange(V)[:, None], near] = True
close |= close.T
np.fill_diagonal(close, True)
upper = np.triu_indices(V, 1)
chance = float(related[upper].mean())  # P(neighbours) for two distinct words drawn at random

# Every unordered pair belongs to one side, so validation pairs are unseen in either order.
rng = np.random.default_rng(args.seed)
in_val = np.zeros((V, V), dtype=bool)
in_val[upper] = rng.random(len(upper[0])) < args.val_fraction
in_val |= in_val.T


def sample_pairs(n, rng, val):
    """n ordered (a, b, y) index triples from one side, half of them neighbours.

    A negative (a, c) replaces a positive (a, b) by a word c from b's block of document-frequency
    ranks, so it is unrelated to a but no rarer or more common than b. The blocks are a quarter
    of the vocabulary, so that they stay narrower than the vocabulary itself.
    """
    bin_size = max(V // 4, 1)
    side = in_val if val else ~in_val
    positives = np.argwhere(np.triu(related & side, 1))
    a, b = positives[rng.integers(len(positives), size=n)].T
    flip = rng.random(n) < 0.5
    a, b = np.where(flip, b, a), np.where(flip, a, b)
    y = np.arange(n) % 2 == 0
    for i in np.flatnonzero(~y):
        start = (b[i] // bin_size) * bin_size
        block = np.arange(start, min(start + bin_size, V))
        candidates = block[~close[a[i], block] & side[a[i], block]]
        if not len(candidates):  # a small side can leave a block empty; widen to all ranks
            candidates = np.flatnonzero(~close[a[i]] & side[a[i]])
        b[i] = rng.choice(candidates)
    a, b = np.concatenate([a, b]), np.concatenate([b, a])
    y = np.concatenate([y, y]).astype(np.uint32)
    order = rng.permutation(len(y))
    return a[order], b[order], y[order]


def one_hot(a, b):
    X = np.zeros((len(a), 2 * V), dtype=np.uint32)
    X[np.arange(len(a)), a] = 1
    X[np.arange(len(a)), V + b] = 1
    return X


a_train, b_train, Y_train = sample_pairs(args.train_pairs, rng, val=False)
a_val, b_val, Y_val = sample_pairs(args.val_pairs, rng, val=True)
X_train, X_val = one_hot(a_train, b_train), one_hot(a_val, b_val)
seen = {(int(u), int(v)) for u, v in zip(a_train, b_train, strict=True)}
leaked = sum((int(u), int(v)) in seen for u, v in zip(a_val, b_val, strict=True))
assert leaked == 0, f"{leaked} validation pairs occur in training"

# Baseline from the words alone: how often each word is in a positive training pair. It is
# what a machine could get without relating a to b, and frequency matching should keep it low.
rate = np.bincount(np.r_[a_train, b_train], np.r_[Y_train, Y_train], V) / np.maximum(
    np.bincount(np.r_[a_train, b_train], minlength=V), 1
)
threshold = np.median(rate[a_train] + rate[b_train])
word_baseline = float(((rate[a_val] + rate[b_val] > threshold) == Y_val).mean())
print(
    f"pairs: train {X_train.shape}, val {X_val.shape} (unseen), positives {Y_train.mean():.2f}; "
    f"word-rate baseline {word_baseline:.4f}; neighbours among random pairs {chance:.2%}"
)

# ---------------------------------------------------------------- the machine

m = args.alternatives
structure = ((htm.AND_GROUP, V), (htm.OR_ALTERNATIVES, m), (htm.AND_GROUP, 2))
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
print(f"structure {structure}, T {args.T}, s {args.s}")

history = []
if saved:
    tm.load(saved["machine"])
    history = saved["history"]
    rng.bit_generator.state = saved["rng"]
    print(f"resumed from {checkpoint.name} after epoch {len(history)}")


def empty_alternatives(tm):
    """Mean number of empty alternatives per OR group: the failure mode that keeps a group true."""
    included = tm.get_literals().sum(axis=2)  # (clauses, 2 * m leaves)
    return float((included == 0).sum() / (args.clauses * 2))


for epoch in range(len(history), args.epochs):
    order = rng.permutation(len(Y_train))
    start = time()
    # fit() takes no epochs argument and does not shuffle: one call is one pass in the given order.
    tm.fit(X_train[order], Y_train[order])
    seconds = time() - start
    val = float((tm.predict(X_val) == Y_val).mean())
    # On a fixed sample of training pairs: tells memorising the seen pairs from not learning.
    train = float((tm.predict(X_train[: len(Y_val)]) == Y_train[: len(Y_val)]).mean())
    history.append(
        {
            "epoch": epoch + 1,
            "train_accuracy": train,
            "val_accuracy": val,
            "empty_per_group": empty_alternatives(tm),
            "train_seconds": seconds,
        }
    )
    print(
        f"#{epoch + 1} train {train:.4f} val {val:.4f}, {history[-1]['empty_per_group']:.2f} "
        "empty alternatives "
        f"per group ({seconds:.0f}s)",
        flush=True,
    )
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        state = {"config": vars(args), "machine": tm.save(), "history": history}
        pickle.dump(state | {"rng": rng.bit_generator.state}, fh)
    tmp.replace(checkpoint)

# ---------------------------------------------------------------- read-out


def extract_groups(tm):
    """Every OR group with its leaves as word lists, in the HWF-TM's format.

    Leaf j of a clause is alternative j % m of slot j // m (lowest level fastest).
    """
    literals = tm.get_literals()  # (clauses, 2 * m, V)
    weights = tm.save()["clause_weights"]
    groups = []
    for c in range(args.clauses):
        for slot in range(2):
            leaves = [np.flatnonzero(literals[c, slot * m + a]) for a in range(m)]
            groups.append(
                {
                    "clause": c,
                    "span": "AB"[slot],
                    "polarity": int(np.sign(weights[c])),
                    "leaves": [[words[k] for k in leaf] for leaf in leaves if len(leaf)],
                    "empty_leaves": int(sum(len(leaf) == 0 for leaf in leaves)),
                }
            )
    return groups


groups = extract_groups(tm)
leaves = [leaf for g in groups for leaf in g["leaves"]]
print(
    f"\n{len(groups)} OR groups, {np.mean([g['empty_leaves'] for g in groups]):.2f} empty "
    f"alternatives per group, {np.mean([len(leaf) == 1 for leaf in leaves]):.0%} single-word "
    "leaves"
)

# Only positive clauses: they vote "related", so a group is what the machine treats as one family.
families = word_families(groups, polarity="positive")
index = {w: i for i, w in enumerate(words)}
pairs = [(index[u], index[v]) for f in families for i, u in enumerate(f) for v in f[i + 1 :]]
linked = float(np.mean([related[u, v] for u, v in pairs])) if pairs else 0.0
print(
    f"{len(families)} families covering {sum(map(len, families))} words; {linked:.0%} of the "
    f"pairs within a family are PPMI neighbours (random pairs: {chance:.0%})"
)
for family in families[:20]:
    print("   ", " | ".join(family))

RESULTS.mkdir(parents=True, exist_ok=True)
pretrain_out = RESULTS / f"pretrain_{args.name}.json"
with pretrain_out.open("w") as fh:
    json.dump(
        {
            "config": vars(args),
            "structure": [list(level) for level in structure],
            "overlap_baseline": 0.5,  # a != b in every pair, so no overlap exists to exploit
            "word_rate_baseline": word_baseline,
            "neighbour_pairs_chance": chance,
            "history": history,
            "empty_per_group": float(np.mean([g["empty_leaves"] for g in groups])),
            "neighbour_pairs_within_families": float(linked),
            "words": words,
            "groups": groups,
        },
        fh,
    )
families_out = RESULTS / f"families_{args.name}.json"
with families_out.open("w") as fh:
    json.dump(
        {
            "name": args.name,
            "source": "HTM pretrained on PPMI-neighbour word pairs (experiments/or_families)",
            "families": families,
            "ranked_words": words,
        },
        fh,
    )
print(f"\nwrote {pretrain_out} and {families_out}")
