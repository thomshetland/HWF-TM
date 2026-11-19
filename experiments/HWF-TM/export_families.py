#!/usr/bin/python
"""Stage 2 of the HWF-TM: write the family lists of every source in one format, for classify.py.

Each source becomes results/families_<name>.json:

  {"name": ..., "source": ..., "families": [[word, ...], ...], "ranked_words": [word, ...]}

`families` is a partition (no word in two families), as the replacing encoding needs.
`ranked_words` is the source's vocabulary in document-frequency order, which the random control
draws its frequency-matched substitutes from.

Sources:
  same-dataset  the pretraining run on the same-dataset pretext, partitioned by
                families.word_families.
  profile       the pretraining run on the profile pretext, with the `disjoint` read-out: whole OR
                groups, most consistent first, no word twice.
  ppmi          experiments/ppmi-tm: k-means over PPMI co-occurrence of the unlabeled IMDB + Amazon
                train reviews. Not a Tsetlin machine at all; it is the reference for what *good*
                label-free word families do to the downstream TM.

Run from the repo root:  ./.venv/bin/python experiments/HWF-TM/export_families.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from families import load_pretrain, word_families  # noqa: E402
from src.family_readout import read_out  # noqa: E402

RESULTS = HERE / "results"
PPMI_FAMILIES = HERE.parent / "ppmi-tm" / "results" / "families_ppmi.json"


def write(name, source, families, ranked_words):
    members = [w for family in families for w in family]
    if len(members) != len(set(members)):
        raise ValueError(f"{name}: families overlap; the replacing encoding needs a partition")
    out = RESULTS / f"families_{name}.json"
    with out.open("w") as fh:
        json.dump(
            {"name": name, "source": source, "families": families, "ranked_words": ranked_words},
            fh,
        )
    sizes = [len(f) for f in families]
    print(f"{name:13s} {len(families):4d} families, {len(members):5d} words, max size {max(sizes)}")
    print("              e.g. " + "   ".join(" | ".join(f) for f in families[:3]))


same_dataset = load_pretrain("same-dataset")
write(
    "same-dataset",
    "HTM pretrained on the same-dataset pretext, partitioned",
    word_families(same_dataset["groups"]),
    same_dataset["words"],
)

profile = load_pretrain("profile")
write(
    "profile",
    "HTM pretrained on the profile pretext, disjoint read-out",
    [sorted(leaf[0] for leaf in family) for family in read_out(profile["groups"], "disjoint")],
    profile["words"],
)

with PPMI_FAMILIES.open() as fh:
    ppmi = json.load(fh)
write(
    "ppmi",
    "experiments/ppmi-tm: PPMI k-means, not an HTM",
    [
        sorted(ppmi["words"][leaf[0]] for leaf in family)
        for family in read_out(ppmi["families"], "words")
    ],
    ppmi["words"],
)
