"""Word families: reading them out of a pretrained HTM, and turning them into input bits.

A pretraining run stores its OR groups, and a PPMI run stores its clusters, in the same form: a
list of entries whose "leaves" are lists of words or word indices. Downstream there are two ways
to use a set of families:

  added       family_features: the dataset's full set of words, plus one bit per family.
              No word loses its own bit.
  replacing   word_family_encoding: one bit per family, and one bit for every vocabulary word
              that is in no family. "great" and "excellent" become the same feature.

Pure Python and NumPy; imports without a GPU.
"""

from itertools import combinations

import numpy as np

from src.common import bag_of_words

READOUTS = ("leaves", "words", "disjoint")


def clean_groups(groups):
    """The OR groups that look like synonym sets: positive polarity, >= 2 single-word leaves.

    Positive polarity: the clause votes for "same" (Y=1), so its groups are what the machine
    uses to *match* the two sides, not to tell them apart. Only single-word leaves are kept;
    a leaf like "the & and & movie" is a conjunction, not a member of a word set.
    """
    out = []
    for g in groups:
        if g.get("polarity", 1) <= 0:
            continue
        group = sorted({leaf[0] for leaf in g["leaves"] if len(leaf) == 1})
        if len(group) >= 2:
            out.append(group)
    return out


def disjoint_groups(groups):
    """A partition made of whole OR groups: the most consistent groups first, no word twice.

    The clause weights cannot rank the groups (in the binary hierarchical machine they are
    fixed at +1 / -1), so a group is ranked by *consensus*: for each pair of its words, how many
    clean groups in the whole run contain both, averaged over the pairs. A group whose words the
    machine put together again and again, in many clauses, comes first. Then greedily: a group
    is kept if none of its words is in a group kept already, and skipped otherwise. Every family
    is therefore exactly an OR group the HTM learned, and no word is in two families.
    """
    distinct = [tuple(g) for g in dict.fromkeys(tuple(g) for g in clean_groups(groups))]
    together = {}
    for group in clean_groups(groups):  # with duplicates: recurrence is the evidence
        for pair in combinations(group, 2):
            together[pair] = together.get(pair, 0) + 1

    def consensus(group):
        return np.mean([together[pair] for pair in combinations(group, 2)])

    ranked = sorted(distinct, key=lambda g: (-consensus(g), -len(g)))
    taken, kept = set(), []
    for group in ranked:
        if taken.isdisjoint(group):
            kept.append([[w] for w in group])
            taken.update(group)
    return kept


def read_out(groups, readout):
    """Families as lists of leaves, duplicates removed.

    leaves     every entry as it is: fires when any leaf (a conjunction) fires.
    words      the union of an entry's leaf words, each as a one-word leaf.
    disjoint   disjoint_groups: whole clean groups, most consistent first, no word twice.
    """
    if readout == "leaves":
        out = [g["leaves"] for g in groups]
    elif readout == "words":
        out = [[[w] for w in sorted({w for leaf in g["leaves"] for w in leaf})] for g in groups]
    elif readout == "disjoint":
        out = disjoint_groups(groups)
    else:
        raise ValueError(f"unknown read-out {readout!r}, expected one of {READOUTS}")
    seen, distinct = set(), []
    for family in out:
        key = tuple(tuple(leaf) for leaf in family)
        if family and key not in seen:
            seen.add(key)
            distinct.append(family)
    return distinct


def family_features(X, families):
    """One bit per family: does any of its leaves fire on the text?

    X is a set-of-words matrix over the vocabulary the families index into. A family is a list of
    leaves and a leaf a list of word indices; a leaf fires when all of its words are present.
    """
    F = np.zeros((X.shape[0], len(families)), dtype=np.uint32)
    for f, family in enumerate(families):
        fires = np.zeros(X.shape[0], dtype=bool)
        for leaf in family:
            fires |= X[:, leaf].all(axis=1)
        F[:, f] = fires
    return F


def permuted_families(families, vocabulary_size, rng, bin_size=50):
    """The same families with every word swapped for a random word of similar frequency.

    The random control: it keeps the number of families, their sizes and the frequency profile,
    and destroys only *which* words were grouped. Vocabulary indices are document-frequency ranks,
    so permuting within consecutive bins of `bin_size` ranks keeps each word's frequency roughly
    fixed.
    """
    mapping = np.arange(vocabulary_size)
    for start in range(0, vocabulary_size, bin_size):
        block = mapping[start : start + bin_size]
        mapping[start : start + bin_size] = block[rng.permutation(len(block))]
    return [[sorted(int(mapping[w]) for w in leaf) for leaf in family] for family in families]


def word_family_encoding(token_lists, families, family_words, vocabulary):
    """[family bits | bits of the vocabulary words that are in no family], and the rest words.

    families      lists of single-word leaves, word indices into `family_words`.
    vocabulary    the dataset's own set-of-words vocabulary {word: index}.

    Bit f is 1 when any word of family f occurs in the text, whether or not that word is in the
    dataset's vocabulary. The remaining bits are the dataset's vocabulary minus every word that
    is in some family, in the vocabulary's own order. Every family keeps its bit, even one that
    never fires on this dataset; HWF-TM/families.py drops those.
    """
    if any(len(leaf) != 1 for family in families for leaf in family):
        raise ValueError("replacing words by families needs single-word leaves")
    covered = {family_words[leaf[0]] for family in families for leaf in family}
    rest = [w for w in sorted(vocabulary, key=vocabulary.get) if w not in covered]
    rest_vocabulary = {w: i for i, w in enumerate(rest)}
    P = bag_of_words(token_lists, {w: i for i, w in enumerate(family_words)})
    X = np.hstack((family_features(P, families), bag_of_words(token_lists, rest_vocabulary)))
    return X, rest
