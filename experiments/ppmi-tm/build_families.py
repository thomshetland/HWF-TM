#!/usr/bin/python
"""Stage 1 of the PPMI-TM: label-free word families from PPMI co-occurrence.

Words that occur in the same texts get similar PPMI vectors, and k-means on those vectors groups
them. No labels and no Tsetlin machine are involved.

  1. Vocabulary: at most 5000 words, among those in 0.2 %-30 % of the texts.
  2. Co-occurrence: c[u, v] = number of texts containing both u and v.
  3. PPMI(u, v) = max(0, log P(u, v) / (P(u) P(v))), diagonal set to zero.
  4. A 100-d SVD with unit-length rows, then k-means.
  5. Clusters of 2-40 words are the families.

Datasets are weighted equally unless --no-balance is given. --word-order reads the vocabulary
order from an earlier result: k-means depends on the order of the rows, and the stored families
were built when the order of equally frequent words still depended on Python's hash seed. With
the stored order this script rebuilds them exactly.

Writes results/<name>.json, which classify.py reads with --families <name>. Run from the repo root:

  ./.venv/bin/python experiments/ppmi-tm/build_families.py --datasets imdb amazon --no-balance \\
      --k 1000 --name families_ppmi --word-order experiments/ppmi-tm/results/families_ppmi.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from src.common import DATASETS, bag_of_words, load_splits  # noqa: E402

RESULTS = HERE / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--vocabulary", type=int, default=5000)
parser.add_argument("--k", type=int, default=None, help="default vocabulary / words-per-family")
parser.add_argument("--words-per-family", type=int, default=5)
parser.add_argument("--word-order", default=None, help="result JSON to read the word order from")
parser.add_argument("--name", required=True)
args = parser.parse_args()

splits = load_splits()
texts, weights = [], []
for dataset in args.datasets:
    rows = splits[dataset]["train"]  # train texts only: the test split stays unseen
    texts += [row["tokens"].split() for row in rows]
    weights += [1.0 / len(rows)] * len(rows)
weights = np.array(weights) / np.sum(weights)

if args.word_order:
    with open(args.word_order) as fh:
        words = json.load(fh)["words"]
else:
    share = Counter()
    for text, w in zip(
        texts, weights if args.balance else [1 / len(texts)] * len(texts), strict=True
    ):
        for word in dict.fromkeys(text):  # text order: most_common breaks ties by first occurrence
            share[word] += w
    words = [w for w, x in share.most_common() if 0.002 <= x <= 0.3][: args.vocabulary]
index = {w: i for i, w in enumerate(words)}
print(f"{args.name}: {len(texts)} texts from {args.datasets}, vocabulary {len(words)}")

X = bag_of_words(texts, index).astype(np.float32)
co = X.T @ (X * weights[:, None].astype(np.float32)) if args.balance else X.T @ X
total = co.sum()
p = co.sum(axis=1) / total
ppmi = np.maximum(np.log((co / total) / (p[:, None] * p[None, :] + 1e-12) + 1e-12), 0)
ppmi[np.arange(len(words)), np.arange(len(words))] = 0
U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
embedding = U[:, :100] * S[:100]
embedding /= np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-9

k = args.k or max(2, len(words) // args.words_per_family)
clusters = KMeans(n_clusters=k, n_init=2, random_state=0).fit(embedding).labels_
families = [np.flatnonzero(clusters == c).tolist() for c in range(k)]
families = [f for f in families if 2 <= len(f) <= 40]
print(
    f"  {len(families)} families of 2-40 words, {sum(map(len, families))} words covered, "
    f"{np.mean([len(f) for f in families]):.1f} per family"
)

RESULTS.mkdir(parents=True, exist_ok=True)
out = RESULTS / f"{args.name}.json"
with out.open("w") as fh:
    json.dump(
        {
            "config": vars(args) | {"k": k},
            "words": words,
            "families": [
                {"leaves": [[w] for w in f], "clause": i, "half": 0, "polarity": 1}
                for i, f in enumerate(families)
            ],
        },
        fh,
    )
print(f"  wrote {out}")
