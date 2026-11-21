#!/usr/bin/python
"""Why are the OR groups of the pretrained HTMs poor word families? A diagnosis of the stored runs.

Reads the two pretraining runs of ../HWF-TM (results/pretrain_<name>.json) and the family files
that were used downstream (results/families_<name>.json), plus the PPMI families as the reference
for families that do help. Three questions:

  1. What does a clause look like? For each clause polarity: which way its two slots lean on the
     dataset axis (IMDB > 0 > Amazon), and how many words slot A and slot B share. A group that
     holds the same words as the group in the other slot is a bundle of "this word on both sides"
     matchers, whose members need have nothing in common.
  2. What are the leaves? Single words or conjunctions, duplicates within a group.
  3. Are the members of a group alike? Sentiment and dataset coherence, and document PMI, each
     against 20 frequency-matched permutations (word_stats.WordStats.coherence).

CPU only, about a minute. Writes results/diagnosis.json. Run from the repo root:
  ./.venv/bin/python experiments/or_families/diagnose.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script
sys.path.insert(0, str(HERE.parent / "HWF-TM"))

from families import permuted_families, word_families  # noqa: E402
from word_stats import WordStats  # noqa: E402

HWF = HERE.parent / "HWF-TM" / "results"
PRETEXTS = ("same-dataset", "profile")
CONTROLS = 20
rng = np.random.default_rng(0)


def load(path):
    with path.open() as fh:
        return json.load(fh)


runs = {name: load(HWF / f"pretrain_{name}.json") for name in PRETEXTS}
downstream = {name: load(HWF / f"families_{name}.json") for name in (*PRETEXTS, "ppmi")}
words = sorted(
    {w for run in runs.values() for w in run["words"]}
    | {w for f in downstream.values() for family in f["families"] for w in family}
)
stats = WordStats(words)


def clauses(run):
    """{clause: {"A": group, "B": group}}; both groups of a clause share its polarity."""
    by_clause = {}
    for group in run["groups"]:
        by_clause.setdefault(group["clause"], {})[group["span"]] = group
    return list(by_clause.values())


def slot_words(group):
    return {w for leaf in group["leaves"] for w in leaf}


def clause_structure(run):
    out = {}
    for polarity, label in ((1, "positive"), (-1, "negative")):
        selected = [c for c in clauses(run) if c["A"]["polarity"] == polarity]
        lean = np.array(
            [
                [np.mean(stats.dataset_log_odds[stats.ids(slot_words(c[s]))]) for s in "AB"]
                for c in selected
            ]
        )
        jaccard = [
            len(slot_words(c["A"]) & slot_words(c["B"]))
            / max(1, len(slot_words(c["A"]) | slot_words(c["B"])))
            for c in selected
        ]
        out[label] = {
            "clauses": len(selected),
            "slot_A_dataset_lean": float(lean[:, 0].mean()),
            "slot_B_dataset_lean": float(lean[:, 1].mean()),
            "slots_lean_opposite": float(np.mean(np.sign(lean[:, 0]) != np.sign(lean[:, 1]))),
            "slot_jaccard": float(np.mean(jaccard)),
        }
    return out


def leaf_structure(run):
    leaves = [leaf for g in run["groups"] for leaf in g["leaves"]]
    duplicates = [
        len(g["leaves"]) - len({tuple(leaf) for leaf in g["leaves"]}) for g in run["groups"]
    ]
    rank = {w: i for i, w in enumerate(run["words"])}
    return {
        "leaves": len(leaves),
        "single_word_leaves": float(np.mean([len(leaf) == 1 for leaf in leaves])),
        "words_per_leaf": float(np.mean([len(leaf) for leaf in leaves])),
        "duplicate_leaves_per_group": float(np.mean(duplicates)),
        "empty_leaves_per_group": run["empty_per_group"],
        "median_member_rank": float(np.median([rank[w] for leaf in leaves for w in leaf])),
        "vocabulary": len(run["words"]),
    }


def coherence(families, ranked_words):
    controls = [permuted_families(families, ranked_words, rng) for _ in range(CONTROLS)]
    return stats.coherence(families, controls)


result = {}
for name, run in runs.items():
    result[name] = {
        "pretext_accuracy": max(h["val_accuracy"] for h in run["history"]),
        "overlap_baseline": run["overlap_baseline"],
        "clause_structure": clause_structure(run),
        "leaf_structure": leaf_structure(run),
        "coherence": {
            f"{polarity} groups": coherence(
                word_families(run["groups"], polarity=polarity, overlap="keep"), run["words"]
            )
            for polarity in ("positive", "negative")
        },
    }
for name, f in downstream.items():
    result.setdefault(name, {}).setdefault("coherence", {})["families used downstream"] = coherence(
        f["families"], f["ranked_words"]
    )

out = HERE / "results" / "diagnosis.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2))

print(f"wrote {out}")
