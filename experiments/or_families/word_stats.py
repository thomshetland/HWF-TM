"""Per-word statistics of the unlabelled IMDB and Amazon train reviews, and family coherence.

A word family is good for sentiment if its words are alike in what the downstream task cares
about. Three per-word quantities are measured on the train texts of IMDB and Amazon:

  dataset log-odds     log P(word | IMDB review) - log P(word | Amazon review), document level.
                       The label of the same-dataset pretext.
  sentiment log-odds   log P(word | positive) - log P(word | negative), computed within each
                       dataset and averaged, so a movie word is not "positive" because IMDB is.
  document PMI         log P(u, v) / (P(u) P(v)) for a pair: do the two words occur in the same
                       reviews? This is what the PPMI families are built from.

Coherence compares a family set with its random control (HWF-TM's frequency-matched
permutations): 1 - (mean within-family variance) / (the same for the control). 0 means the words
of a family are no more alike than random words of the same frequency, 1 means identical.

Pure NumPy/SciPy; imports without a GPU.
"""

from itertools import combinations

import numpy as np
from scipy import sparse

from src.common import load_splits, tokens

PRETRAIN_DATASETS = ("imdb", "amazon")


class WordStats:
    def __init__(self, words):
        self.index = {w: i for i, w in enumerate(words)}
        splits = load_splits()
        matrices, positive = [], []
        for name in PRETRAIN_DATASETS:
            rows = splits[name]["train"]  # train texts only, as in pretraining
            matrices.append(self._documents(rows))
            positive.append(np.array([row["sentiment"] == "positive" for row in rows]))

        self.dataset_log_odds = _log_odds(matrices[0], matrices[1])
        self.sentiment_log_odds = np.mean(
            [_log_odds(M[y], M[~y]) for M, y in zip(matrices, positive, strict=True)], axis=0
        )
        self.documents = sparse.vstack(matrices).tocsc()
        self.p = np.asarray(self.documents.mean(axis=0)).ravel()

    def _documents(self, rows):
        """Binary (texts, words) matrix: does the word occur in the text?"""
        r, c = [], []
        for i, row in enumerate(rows):
            present = {self.index[w] for w in tokens(row) if w in self.index}
            r += [i] * len(present)
            c += list(present)
        shape = (len(rows), len(self.index))
        return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=shape)

    def ids(self, family):
        return [self.index[w] for w in family if w in self.index]

    def within_variance(self, families, axis):
        """Mean over families of the variance of a per-word statistic among its members."""
        values = [np.var(axis[ids]) for ids in map(self.ids, families) if len(ids) >= 2]
        return float(np.mean(values))

    def pmi(self, families):
        """Mean over families of the mean document PMI of its member pairs."""
        out = []
        for ids in map(self.ids, families):
            if len(ids) < 2:
                continue
            columns = self.documents[:, ids]
            joint = (columns.T @ columns).toarray() / self.documents.shape[0]
            pairs = list(combinations(range(len(ids)), 2))
            out.append(
                np.mean(
                    [
                        np.log((joint[i, j] + 1e-6) / (self.p[ids[i]] * self.p[ids[j]]))
                        for i, j in pairs
                    ]
                )
            )
        return float(np.mean(out))

    def coherence(self, families, controls):
        """Sentiment and dataset coherence against the controls, and PMI with the control's."""
        out = {"families": sum(len(self.ids(f)) >= 2 for f in families)}
        for name, axis in (
            ("sentiment", self.sentiment_log_odds),
            ("dataset", self.dataset_log_odds),
        ):
            control = np.mean([self.within_variance(c, axis) for c in controls])
            out[f"{name}_coherence"] = 1 - self.within_variance(families, axis) / control
        out["pmi"] = self.pmi(families)
        out["pmi_control"] = float(np.mean([self.pmi(c) for c in controls[:5]]))
        return out


def _log_odds(a, b):
    """Smoothed document-frequency log-odds of every column between two text matrices."""
    fa, fb = np.asarray(a.sum(axis=0)).ravel(), np.asarray(b.sum(axis=0)).ravel()
    return np.log((fa + 1) / (a.shape[0] + 2)) - np.log((fb + 1) / (b.shape[0] + 2))
