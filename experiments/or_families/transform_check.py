#!/usr/bin/python
"""Would a transform() give better families than reading the leaves? Checked on the GPU.

Upstream `main` (0.3.4 onwards) has a transform(X) that returns each clause's root vote per
example; its CUDA kernel is still a stub. The family-level version is the OR node outputs: after
evaluate_hierarchy, hierarchy_votes[1] holds, per clause and slot, how many alternatives fire.

This script computes that OR-level transform on the stored same-dataset pretraining run of
../HWF-TM and compares it with what the leaves predict: a leaf fires when every word it includes
is present (an empty leaf always fires). If the two are identical, transform() only evaluates
what the machine learned and cannot produce better families than the read-out. It also checks
that the root vote is the product of the two OR outputs, as upstream's transform returns.

Random sparse inputs suffice, because the equality is per input. Library 0.3.1. Writes
results/transform_check.json. Run from the repo root:
  ./.venv/bin/python experiments/or_families/transform_check.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import PyHierarchicalTsetlinMachineCUDA.tm as htm

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "HWF-TM" / "models" / "pretrain_same-dataset.pkl"

with CHECKPOINT.open("rb") as fh:
    saved = pickle.load(fh)
config, V = saved["config"], len(saved["words"])
m = config["alternatives"]
tm = htm.TsetlinMachine(
    config["clauses"],
    config["T"],
    config["s"],
    hierarchy_structure=((htm.AND_GROUP, V), (htm.OR_ALTERNATIVES, m), (htm.AND_GROUP, 2)),
    boost_true_positive_feedback=0,
    number_of_state_bits=8,
    append_negated=False,
    seed=config["seed"],
)
tm.load(saved["machine"])


def node_outputs(tm, X):
    """(examples, clauses, slots) OR node outputs and (examples, clauses) root votes."""
    X = X.reshape(X.shape[0], X.shape[1], 1)
    encoded = cuda.mem_alloc(int(X.shape[0] * tm.number_of_literal_chunks * 4))
    tm.encode_X(X, encoded)
    n_or = int(tm.hierarchy_size[2])
    or_out = np.empty((X.shape[0], tm.number_of_clauses * n_or), dtype=np.float32)
    root = np.empty((X.shape[0], tm.number_of_clauses), dtype=np.float32)
    for e in range(X.shape[0]):
        tm.evaluate_hierarchy(encoded, e)
        cuda.memcpy_dtoh(or_out[e], tm.hierarchy_votes[1])
        cuda.memcpy_dtoh(root[e], tm.hierarchy_votes[tm.depth - 1])
    return or_out.reshape(X.shape[0], tm.number_of_clauses, n_or), root


rng = np.random.default_rng(0)
X = (rng.random((300, 2 * V)) < 0.03).astype(np.uint32)
or_out, root = node_outputs(tm, X)

# (clauses, 2 * m leaves, V); leaf j is alternative j % m of slot j // m
literals = tm.get_literals()
from_leaves = np.zeros_like(or_out)
for c in range(config["clauses"]):
    for slot in range(2):
        for a in range(m):
            words = np.flatnonzero(literals[c, slot * m + a])
            from_leaves[:, c, slot] += X[:, slot * V + words].all(axis=1)

result = {
    "or_transform_equals_leaf_readout": bool(np.array_equal(or_out, from_leaves)),
    "root_vote_equals_product_of_or_outputs": bool(
        np.array_equal(root, or_out[:, :, 0] * or_out[:, :, 1])
    ),
    "examples": int(X.shape[0]),
    "mean_firing_alternatives_per_group": float(or_out.mean()),
}
(HERE / "results" / "transform_check.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
