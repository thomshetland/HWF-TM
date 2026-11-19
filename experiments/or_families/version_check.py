#!/usr/bin/python
"""Does library 0.3.5, with its new options at their defaults, train exactly like 0.3.1?

Every stored HTM result was made with 0.3.1 (.venv). The new pretext needs 0.3.5 (.venv-htm035)
for literal_budget and binary_inference. Run this once in each environment: it trains the
HWF-TM's two-slot structure on a planted synonym task with a fixed seed and stores the automata
states and predictions. The second run compares with the first.

  ./.venv/bin/python experiments/or_families/version_check.py
  ./.venv-htm035/bin/python experiments/or_families/version_check.py
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as htm

HERE = Path(__file__).resolve().parent
V, M, CLAUSES = 32, 4, 20

# Two planted families of four words; Y = 1 when both slots hold a word of the same family.
rng = np.random.default_rng(0)
families = [np.arange(0, 4), np.arange(4, 8)]
X = np.zeros((4000, 2 * V), dtype=np.uint32)
Y = np.zeros(4000, dtype=np.uint32)
for i in range(len(X)):
    a, b = rng.integers(2, size=2)
    X[i, rng.choice(families[a])] = 1
    X[i, V + rng.choice(families[b])] = 1
    X[i, rng.choice(np.arange(8, V), 3)] = 1  # distractor words
    Y[i] = a == b

tm = htm.TsetlinMachine(
    CLAUSES,
    16,
    30.0,
    hierarchy_structure=((htm.AND_GROUP, V), (htm.OR_ALTERNATIVES, M), (htm.AND_GROUP, 2)),
    boost_true_positive_feedback=0,
    number_of_state_bits=8,
    append_negated=False,
    seed=7,
)
for _ in range(10):
    tm.fit(X, Y)

run = {"ta_states": tm.get_ta_states(), "scores": tm.score(X)}
this = version("PyHierarchicalTsetlinMachineCUDA")
print(f"library {this}: accuracy {(tm.predict(X) == Y).mean():.4f}")
np.savez(HERE / "results" / f"version_check_{this}.npz", **run)

others = [p for p in (HERE / "results").glob("version_check_*.npz") if this not in p.name]
for path in others:
    other = np.load(path)
    same = all(np.array_equal(run[k], other[k]) for k in run)
    print(f"identical to {path.stem}: {same}")
