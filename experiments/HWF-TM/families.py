#!/usr/bin/python
"""Stage 2 of the HWF-TM: from OR groups to word families, and from word families to input bits.

The downstream input replaces words by families: one bit per family ("some member of the family
is in the text"), then one bit per vocabulary word that belongs to no family. With K families
covering M vocabulary words, the input is K + (5000 - M) bits wide instead of 5000. A clause
that needed one literal per synonym now needs one literal per family, which is how the families
can make clauses fewer and shorter.

Replacing (rather than appending) puts two requirements on the families:
  * A family must be a set of words. A leaf such as "film & was" is a conjunction, and there is
    no word bit it could replace, so only single-word leaves are read (measured: 60-70 % of the
    leaves of the recipe's runs).
  * A word should be in at most one family. The OR groups of a trained machine overlap heavily
    (in the recipe's runs every word sits in about three groups, "book" in 64), and with
    overlapping families one word would switch on dozens of bits. overlap="partition" therefore
    assigns each word to one family; overlap="keep" leaves the groups as learned.

Run directly to inspect the families of a pretraining run:

  ./.venv/bin/python experiments/HWF-TM/families.py --pretrain same-dataset
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

RESULTS = HERE / "results"


def load_pretrain(name):
    with (RESULTS / f"pretrain_{name}.json").open() as fh:
        return json.load(fh)


def word_families(groups, polarity="all", overlap="partition", min_size=2):
    """Word families, as sorted word lists, from the OR groups of a pretrained machine.

    polarity: "all", or only the groups of "positive" clauses (which vote "belong together" and
      so hold one thing both spans mention) or "negative" ones (which vote "do not belong" and
      often hold a union of several families).
    overlap: "partition" processes groups from largest to smallest and gives each word to the
      first group that contains it; a group left with fewer than min_size unassigned words is
      dropped and its words stay free for later groups. Largest first, because a large group
      replaces the most word bits with one family bit. "keep" returns the distinct groups as
      they are, overlaps included.
    """
    wanted = {"all": (-1, 1), "positive": (1,), "negative": (-1,)}[polarity]
    sets, seen = [], set()
    for group in groups:
        if group["polarity"] not in wanted:
            continue
        members = frozenset(leaf[0] for leaf in group["leaves"] if len(leaf) == 1)
        if len(members) >= min_size and members not in seen:
            seen.add(members)
            sets.append(members)

    if overlap == "keep":
        return [sorted(members) for members in sets]
    if overlap != "partition":
        raise ValueError(f"unknown overlap mode {overlap!r}")

    families, assigned = [], set()
    # sorted() is stable, so groups of equal size keep the order of the clauses they came from.
    for members in sorted(sets, key=len, reverse=True):
        new = sorted(members - assigned)
        if len(new) >= min_size:
            families.append(new)
            assigned.update(new)
    return families


def permuted_families(families, ranked_words, rng, bin_size=50):
    """The same families with every word swapped for a random word of similar frequency.

    The control for "any compression helps": it keeps the number of families, their sizes and
    their frequency profile, and destroys only *which* words were grouped. ranked_words is the
    pretraining vocabulary in document-frequency order; words are permuted within consecutive
    bins of bin_size ranks. One permutation is applied everywhere, so a partition stays a
    partition.
    """
    mapping = {}
    for start in range(0, len(ranked_words), bin_size):
        block = ranked_words[start : start + bin_size]
        shuffled = [block[i] for i in rng.permutation(len(block))]
        mapping.update(zip(block, shuffled, strict=True))
    return [sorted(mapping.get(w, w) for w in family) for family in families]


class FamilyEncoder:
    """The HWF-TM input: one bit per family, then one bit per word that is in no family.

    Columns [0, K) are the kept families; column K + i is the i-th vocabulary word (in the
    vocabulary's own order) that is not a member of any kept family. A family bit is true when
    any of its members occurs in the text, including members outside the dataset's vocabulary,
    so rare synonyms of common words still count.
    """

    def __init__(self, families, vocabulary, max_df=0.95):
        self.candidates = [list(f) for f in families]
        self.vocabulary = vocabulary
        self.max_df = max_df

    def fit(self, token_lists):
        """Keep the families whose bit is ever true and not almost always true, on train texts.

        A family that never fires wastes a column. One that fires on almost every text carries
        nothing, and since it replaces its members it would also erase their own bits, so its
        members are given back their word bits.
        """
        hits = self._family_bits(token_lists, self.candidates)
        df = hits.mean(axis=0)
        keep = (df > 0) & (df <= self.max_df)
        self.families = [f for f, k in zip(self.candidates, keep, strict=True) if k]
        members = {w for family in self.families for w in family}
        self.words = [
            w for w in sorted(self.vocabulary, key=self.vocabulary.get) if w not in members
        ]
        self.columns = ["{" + "|".join(f) + "}" for f in self.families] + self.words
        self.covered = len(members & set(self.vocabulary))
        return self

    @property
    def n_families(self):
        return len(self.families)

    def transform(self, token_lists, dtype=np.uint8):
        families = self._family_bits(token_lists, self.families, dtype)
        word_index = {w: i for i, w in enumerate(self.words)}
        words = np.zeros((len(token_lists), len(self.words)), dtype=dtype)
        for i, text in enumerate(token_lists):
            for w in set(text):
                j = word_index.get(w)
                if j is not None:
                    words[i, j] = 1
        return np.hstack((families, words))

    @staticmethod
    def _family_bits(token_lists, families, dtype=np.uint8):
        member_of = {}
        for f, family in enumerate(families):
            for w in family:
                member_of.setdefault(w, []).append(f)
        X = np.zeros((len(token_lists), len(families)), dtype=dtype)
        for i, text in enumerate(token_lists):
            for w in set(text):
                for f in member_of.get(w, ()):
                    X[i, f] = 1
        return X


if __name__ == "__main__":
    from src.common import DATASETS, dataset_vocabulary, load_splits, tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default="same-dataset", help="results/pretrain_<name>.json")
    parser.add_argument("--polarity", choices=("all", "positive", "negative"), default="all")
    parser.add_argument("--overlap", choices=("partition", "keep"), default="partition")
    parser.add_argument("--min-family-size", type=int, default=2)
    parser.add_argument("--show", type=int, default=30, help="families to print")
    args = parser.parse_args()

    pretrained = load_pretrain(args.pretrain)
    groups = pretrained["groups"]
    families = word_families(groups, args.polarity, args.overlap, args.min_family_size)
    sizes = [len(f) for f in families]
    print(
        f"{len(groups)} OR groups -> {len(families)} families ({args.polarity}, {args.overlap}), "
        f"{sum(sizes)} member words, size mean {np.mean(sizes):.1f}, max {max(sizes)}"
    )
    for family in families[: args.show]:
        print("  " + " | ".join(family))

    splits = load_splits()
    print("\nInput width per dataset (baseline: 5000 word bits):")
    for name in DATASETS:
        train = [tokens(r) for r in splits[name]["train"]]
        encoder = FamilyEncoder(families, dataset_vocabulary(splits[name]["train"])).fit(train)
        width = encoder.n_families + len(encoder.words)
        print(
            f"  {name:8s} {encoder.n_families:4d} family bits replace {encoder.covered:4d} "
            f"word bits -> {width} bits"
        )
