#!/usr/bin/python
"""The two-slot HTM on planted word families: which library options and settings let it train?

The pair task of the early probes (X2, X7 in docs/findings-early-probes.md), on synthetic words:
n_sets families of six words, one word per slot, Y = 1 when both words come from the same
family. The rule to find is one clause per family, (family)(A) AND (family)(B). Every
configuration is trained with the same seed, and the result is the validation accuracy.

  options   four families: literal_budget (None or 1) x binary_inference x T, at s = 30.
  s         eight families: s = 30 or 100, with and without literal_budget = 1.
  scale     4 to 32 families at 16 clauses per family, s = 100, literal_budget = 1: how many
            families can one machine learn? A real vocabulary has hundreds.
  scale-s   the same at 8 to 32 families with s = 400 and 1000: is the limit at s = 100 a
            capacity limit or a specificity setting?

Validation accuracy alone does not say whether the families were found, so every row also
counts the recovered families: a planted family is recovered when some positive clause has, in
both slots, no empty alternative and single-word alternatives that cover at least three of its
six words at 80 % purity or more. An empty alternative keeps its group true for every word, so
such a group is no family. A leaf of two or more words never fires on a one-hot slot; those
dead leaves are counted too (Type II feedback includes absent words, so they do arise).

Needs library 0.3.5. Writes
results/planted.csv. Run from the repo root:

  ./.venv-htm035/bin/python experiments/or_families/planted.py
"""

import csv
from pathlib import Path

import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as htm

HERE = Path(__file__).resolve().parent
SIZE, M = 6, 12  # words per family; alternatives, about twice the family size (X2)

CONFIGS = [
    # (sweep, families, clauses per family, T, s, literal_budget, binary_inference, epochs)
    *[
        ("options", 4, 4, T, 30.0, budget, binary, 40)
        for budget in (None, 1)
        for binary in (False, True)
        for T in (4, 16)
    ],
    *[("s", 8, 4, 4, s, budget, False, 60) for s in (30.0, 100.0) for budget in (None, 1)],
    *[("scale", n, 16, 8, 100.0, 1, False, 60) for n in (4, 8, 16, 32)],
    *[("scale-s", n, 16, 8, s, 1, False, 60) for s in (400.0, 1000.0) for n in (8, 16, 32)],
]
FIELDS = ["sweep", "families", "clauses", "T", "s", "literal_budget", "binary_inference"]
OUT = HERE / "results" / "planted.csv"


def pairs(n, n_sets, rng):
    """Half the pairs from one family, half from two random families (which may coincide)."""
    V = n_sets * SIZE
    X = np.zeros((n, 2 * V), dtype=np.uint32)
    Y = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        a, b = rng.integers(n_sets, size=2) if i % 2 else (rng.integers(n_sets),) * 2
        X[i, a * SIZE + rng.integers(SIZE)] = 1
        X[i, V + b * SIZE + rng.integers(SIZE)] = 1
        Y[i] = a == b
    return X, Y


def recovered(tm, n_sets):
    """Planted families that some positive clause holds, cleanly, in both slots (see above)."""
    literals = tm.get_literals()  # (clauses, 2 * M leaves, V): word indices within the slot
    weights = tm.save()["clause_weights"]
    found = set()
    for c in np.flatnonzero(weights > 0):
        per_slot = []
        for slot in range(2):
            leaves = [np.flatnonzero(literals[c, slot * M + a]) for a in range(M)]
            if any(len(leaf) == 0 for leaf in leaves):
                per_slot.append(set())
                continue
            members = {int(leaf[0]) for leaf in leaves if len(leaf) == 1}
            families = {}
            for w in members:
                families.setdefault(w // SIZE, set()).add(w)
            per_slot.append(
                {f for f, ws in families.items() if len(ws) >= 3 and len(ws) >= 0.8 * len(members)}
            )
        found |= per_slot[0] & per_slot[1]
    return len(found)


# Resumable: a row is written as soon as it is done, and a rerun skips the rows already in OUT.
rows = []
if OUT.exists():
    with OUT.open() as fh:
        rows = list(csv.DictReader(fh))
done = {tuple(r[k] for k in FIELDS) for r in rows}
for sweep, n_sets, per_set, T, s, budget, binary, epochs in CONFIGS:
    key = (sweep, n_sets, per_set * n_sets, T, s, "" if budget is None else budget, binary)
    if tuple(map(str, key)) in done:
        continue
    rng = np.random.default_rng(0)
    X, Y = pairs(max(6000, 500 * n_sets), n_sets, rng)
    X_val, Y_val = pairs(2000, n_sets, rng)
    tm = htm.TsetlinMachine(
        per_set * n_sets,
        T,
        s,
        literal_budget=budget,
        binary_inference=binary,
        hierarchy_structure=(
            (htm.AND_GROUP, n_sets * SIZE),
            (htm.OR_ALTERNATIVES, M),
            (htm.AND_GROUP, 2),
        ),
        boost_true_positive_feedback=0,
        number_of_state_bits=8,
        append_negated=False,
        seed=1,
    )
    accuracy = []
    for _ in range(epochs):
        order = rng.permutation(len(Y))
        tm.fit(X[order], Y[order])  # one call is one pass, in the given order
        accuracy.append(float((tm.predict(X_val) == Y_val).mean()))
    leaf_sizes = tm.get_literals().sum(axis=2)
    empties = float((leaf_sizes == 0).sum() / (tm.number_of_clauses * 2))
    rows.append(
        {
            "sweep": sweep,
            "families": n_sets,
            "clauses": per_set * n_sets,
            "T": T,
            "s": s,
            "literal_budget": budget,
            "binary_inference": binary,
            "epochs": epochs,
            "final_val_accuracy": f"{accuracy[-1]:.4f}",
            "best_val_accuracy": f"{max(accuracy):.4f}",
            "empty_per_group": f"{empties:.2f}",
            "dead_leaves_per_group": f"{(leaf_sizes >= 2).sum() / (tm.number_of_clauses * 2):.2f}",
            "recovered_families": recovered(tm, n_sets),
        }
    )
    print(", ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
print(f"wrote {OUT}")
