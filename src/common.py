"""Data handling shared by the PPMI-TM and the HWF-TM.

The splits are exactly those of experiments/classification/tm/{imdb,amazon,finance}.py, so the
set-of-words numbers of every experiment are directly comparable to that benchmark:

  IMDB      80/20 split of the 50 000 reviews.
  Amazon    40 000 reviews sampled from train.csv, 10 000 from test.csv.
  Finance   80/20 split of the 4846 sentences. Never used for pretraining: keeping it out makes
            it the out-of-domain test.

Pretraining sees IMDB and Amazon *train* texts only, with the sentiment labels thrown away, so
the downstream test texts are never seen before evaluation.
"""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocessing.create_ds import build_vocabulary
from src.preprocessing.tokenizing import (
    AMAZON_RAW,
    load_amazon,
    load_finance,
    load_imdb,
    tokenize_document,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_SEED = 42
TEST_FRACTION = 0.2
SPLITS_PATH = ROOT / "data" / "processed" / f"splits_seed{DATA_SEED}.pkl"
DATASETS = ("imdb", "amazon", "finance")
PRETRAIN_DATASETS = ("imdb", "amazon")
VOCABULARY_SIZE = 5000  # the downstream vocabulary of the benchmark
MIN_TOKENS = 10  # shorter reviews leave a side almost empty once the shared words are deleted


def load_splits():
    """{dataset: {"train": rows, "test": rows}}, rows as {"tokens": str, "sentiment": str}.

    Cached, because sampling Amazon reads a 3.6M-row file. Each dataset gets a fresh generator,
    consumed in the same order as the benchmark scripts, so the splits are identical to theirs.
    """
    if SPLITS_PATH.exists():
        with SPLITS_PATH.open("rb") as fh:
            return pickle.load(fh)

    print("Loading and tokenizing (first run only, cached afterwards)...")
    splits = {}
    for name, load in (("imdb", load_imdb), ("finance", load_finance)):
        rng = np.random.default_rng(DATA_SEED)
        rows = tokenize_document(load())
        rows = [rows[i] for i in rng.permutation(len(rows))]
        n_test = int(len(rows) * TEST_FRACTION)
        splits[name] = {"test": rows[:n_test], "train": rows[n_test:]}

    rng = np.random.default_rng(DATA_SEED)
    splits["amazon"] = {
        "train": tokenize_document(load_amazon(40000, rng)),
        "test": tokenize_document(load_amazon(10000, rng, AMAZON_RAW.with_name("test.csv"))),
    }

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLITS_PATH.open("wb") as fh:
        pickle.dump(splits, fh)
    return splits


def tokens(row):
    return row["tokens"].split()


def labels(rows, classes):
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[row["sentiment"]] for row in rows], dtype=np.uint32)


def dataset_vocabulary(rows):
    """The benchmark's vocabulary: the VOCABULARY_SIZE words in the most training texts."""
    return build_vocabulary([rows], VOCABULARY_SIZE)


def bag_of_words(token_lists, vocabulary, dtype=np.uint32):
    """(len(token_lists), len(vocabulary)) bit matrix: is the word in the text?"""
    X = np.zeros((len(token_lists), len(vocabulary)), dtype=dtype)
    for i, words in enumerate(token_lists):
        for word in words:
            j = vocabulary.get(word)
            if j is not None:
                X[i, j] = 1
    return X


# ---------------------------------------------------------------- pretext pairs


def pretext_vocabulary(rows, size, min_df, max_df):
    """The `size` most frequent words among those that occur in [min_df, max_df] of the reviews.

    Both bounds exist for the OR groups. A word that is almost never present becomes a dead
    literal: Type II includes every false literal of a firing leaf, and only slow Type Ib decay
    removes it again. A word present in most reviews is dragged into every alternative by Type Ia,
    so families read "film & was" instead of "film". Indices stay in document-frequency order,
    which the permuted control relies on.
    """
    frequency = Counter()
    for row in rows:
        # In text order, not a set(): most_common breaks ties by first occurrence.
        frequency.update(dict.fromkeys(tokens(row), 1))
    lo, hi = min_df * len(rows), max_df * len(rows)
    words = [w for w, n in frequency.most_common() if lo <= n <= hi][:size]
    return {word: i for i, word in enumerate(words)}


def sample_dataset_pairs(reviews_by_dataset, n_pairs, rng):
    """Review pairs (A, B, Y) for the same-dataset pretext.

    A and B are two reviews drawn at random. Y=1 if both come from the same corpus (IMDB or
    Amazon); the four (corpus, corpus) cells are drawn equally often.
    """
    datasets = list(reviews_by_dataset)
    cells = [(a, b) for a in datasets for b in datasets]
    pairs = []
    for i in range(n_pairs):
        a, b = cells[i % len(cells)]
        first = reviews_by_dataset[a][rng.integers(len(reviews_by_dataset[a]))]
        second = reviews_by_dataset[b][rng.integers(len(reviews_by_dataset[b]))]
        pairs.append((tokens(first), tokens(second), int(a == b)))
    return [pairs[i] for i in rng.permutation(n_pairs)]


def sample_profile_pairs(reviews_by_dataset, vocabulary, n_pairs, rng, sample, window, min_count):
    """(profile, profile, Y) pairs for the profile pretext: same word, or two different words?

    A word's profile is the set of vocabulary words that occur within `window` tokens of at
    least `min_count` of `sample` randomly chosen occurrences of it. Y=1 pairs two profiles from
    disjoint occurrence samples of one word, Y=0 profiles of two different words; targets are
    drawn uniformly from the vocabulary words with enough occurrences, so frequency is no cue.

    This is distributional similarity in a form a clause can read: a profile pools many contexts,
    as PPMI does, and an OR group over profile words is a family of interchangeable context words.
    """
    documents = [
        row["tokens"].split() for name in reviews_by_dataset for row in reviews_by_dataset[name]
    ]
    occurrences = {w: [] for w in vocabulary}
    for d, doc in enumerate(documents):
        for i, w in enumerate(doc):
            if w in occurrences:
                occurrences[w].append((d, i))
    targets = [w for w, occ in occurrences.items() if len(occ) >= 4 * sample]

    def profile(word, chosen):
        counts = Counter()
        for d, i in chosen:
            doc = documents[d]
            counts.update(
                {u for u in doc[max(0, i - window) : i] + doc[i + 1 : i + 1 + window]} - {word}
            )
        return [u for u, c in counts.items() if c >= min_count and u in vocabulary]

    pairs = []
    for k in range(n_pairs):
        same = k % 2 == 0
        a = targets[rng.integers(len(targets))]
        b = a
        while not same and b == a:
            b = targets[rng.integers(len(targets))]
        occ_a, occ_b = occurrences[a], occurrences[b]
        pick_a = rng.choice(len(occ_a), 2 * sample if same else sample, replace=False)
        pick_b = pick_a[sample:] if same else rng.choice(len(occ_b), sample, replace=False)
        pairs.append(
            (
                profile(a, [occ_a[j] for j in pick_a[:sample]]),
                profile(b, [occ_b[j] for j in pick_b]),
                int(same),
            )
        )
    return [pairs[i] for i in rng.permutation(n_pairs)]


def remove_shared(pairs):
    """Delete from both spans every word the two share.

    With shared words gone, the only evidence that two spans belong together is related but
    *different* words, which is the relation an OR group exists to represent. With them, word
    overlap dominates the label and the pretext barely trains.
    """
    out = []
    for a, b, y in pairs:
        shared = set(a) & set(b)
        out.append(([w for w in a if w not in shared], [w for w in b if w not in shared], y))
    return out


def pair_matrix(pairs, vocabulary):
    """[A | B] bit matrix, the layout the (AND_GROUP, 2) top level of the hierarchy expects."""
    A = bag_of_words([p[0] for p in pairs], vocabulary)
    B = bag_of_words([p[1] for p in pairs], vocabulary)
    Y = np.array([p[2] for p in pairs], dtype=np.uint32)
    return np.hstack((A, B)), Y


def overlap_baseline(X_train, Y_train, X_test, Y_test):
    """Accuracy of "the spans share at least k words", k fitted on the training pairs.

    The reference for the pretext accuracy: a margin over this is what the hierarchy learned
    beyond word overlap. With shared words removed it is at chance by construction.
    """
    V = X_train.shape[1] // 2

    def shared(X):
        return (X[:, :V] & X[:, V:]).sum(axis=1)

    s_train, s_test = shared(X_train), shared(X_test)
    best_k = max(range(int(s_train.max()) + 2), key=lambda k: ((s_train >= k) == Y_train).mean())
    return float(((s_test >= best_k) == Y_test).mean()), best_k
