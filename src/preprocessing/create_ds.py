from collections import Counter

import numpy as np


def build_vocabulary(tokenized_documents, size):
    """The `size` words appearing in the most reviews, mapped to a column index.

    Pass training reviews only, so the vocabulary never sees the test set.
    """
    document_frequency = Counter()
    for document in tokenized_documents:
        for row in document:
            # Each word once per review, in text order. most_common breaks ties by first
            # occurrence, and counting a set() would make that order depend on the hash seed.
            document_frequency.update(dict.fromkeys(row["tokens"].split(), 1))

    words = [word for word, _ in document_frequency.most_common(size)]
    return {word: i for i, word in enumerate(words)}


def create_ds(document1, document2, vocabulary, n_pairs, rng):
    """Sample review pairs and turn them into a (n_pairs, 2 * len(vocabulary)) bit matrix.

    Y=1 if both reviews come from the same dataset, Y=0 if not. The four combinations
    (1,1), (2,2), (1,2), (2,1) are sampled equally often, so classes are balanced and
    neither dataset is always on the left.

    Columns [0, V) are the words of review A, [V, 2V) the words of review B.
    """
    V = len(vocabulary)
    documents = [document1, document2]
    combinations = [(0, 0), (1, 1), (0, 1), (1, 0)]

    X = np.zeros((n_pairs, V * 2), dtype=np.uint32)
    Y = np.zeros(n_pairs, dtype=np.uint32)

    for i in range(n_pairs):
        a, b = combinations[i % 4]
        index_a = rng.integers(len(documents[a]))
        index_b = rng.integers(len(documents[b]))
        while a == b and index_b == index_a:  # never pair a review with itself
            index_b = rng.integers(len(documents[b]))

        for offset, row in ((0, documents[a][index_a]), (V, documents[b][index_b])):
            for word in row["tokens"].split():
                if word in vocabulary:
                    X[i, offset + vocabulary[word]] = 1

        Y[i] = int(a == b)

    order = rng.permutation(n_pairs)
    return X[order], Y[order]
