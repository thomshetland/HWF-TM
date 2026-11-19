#!/usr/bin/python
"""Coherence of the families from pretrain.py, measured as diagnose.py measures the HWF-TM's.

For each results/families_<name>.json: sentiment and dataset coherence and document PMI against
20 frequency-matched permutations, plus the reference families of ../ppmi-tm restricted to the
same vocabulary, so a small pretraining vocabulary is compared with PPMI on equal terms.

CPU only. Writes results/coherence_<name>.json. Run from the repo root:
  ./.venv/bin/python experiments/or_families/evaluate.py ppmi-pairs_T16
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script
sys.path.insert(0, str(HERE.parent / "HWF-TM"))

from families import permuted_families  # noqa: E402
from word_stats import WordStats  # noqa: E402

PPMI = HERE.parent / "HWF-TM" / "results" / "families_ppmi.json"
CONTROLS = 20


def load(path):
    with path.open() as fh:
        return json.load(fh)


ppmi = load(PPMI)
for name in sys.argv[1:]:
    learned = load(HERE / "results" / f"families_{name}.json")
    vocabulary = set(learned["ranked_words"])
    # PPMI families cut down to the pretraining vocabulary; those left with one word drop out.
    reference = [
        kept for f in ppmi["families"] if len(kept := [w for w in f if w in vocabulary]) >= 2
    ]
    stats = WordStats(learned["ranked_words"])
    rng = np.random.default_rng(0)
    result = {}
    for label, families in (("learned", learned["families"]), ("ppmi, same vocabulary", reference)):
        controls = [
            permuted_families(families, learned["ranked_words"], rng) for _ in range(CONTROLS)
        ]
        result[label] = stats.coherence(families, controls) | {"words": sum(map(len, families))}
    out = HERE / "results" / f"coherence_{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n{name} (vocabulary {len(vocabulary)})")
    print(f"  {'family set':<24} {'n':>4} {'words':>5}  sentiment  dataset   PMI (control)")
    for label, c in result.items():
        print(
            f"  {label:<24} {c['families']:>4} {c['words']:>5}  {c['sentiment_coherence']:+9.2f}"
            f"  {c['dataset_coherence']:+7.2f}   {c['pmi']:.2f} ({c['pmi_control']:.2f})"
        )
