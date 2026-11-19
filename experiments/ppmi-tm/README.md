# PPMI-TM: a TM using synonyms

This is the folder of the PPMI-TM, a TM using synonyms. We used PPMI to find words that occur
together, clustered them into word families, and represented each family as a single literal to the
TM. The HTM is not involved. The experiment shows how much a TM can gain from label-free word
families when the families are good, which is the gain the HWF-TM would have to reproduce.

$$\operatorname{PPMI}(u, v) = \max\left(0,\ \log \frac{P(u, v)}{P(u)\,P(v)}\right)$$

Here $P(u, v)$ is estimated from how many reviews contain both words.

## Architecture

```text
Stage 1, no labels:
  IMDB + Amazon train reviews -> co-occurrence -> PPMI -> 100-d SVD -> k-means, k = 1000
                              -> 866 families covering 4534 words

Stage 2, labels:
  text -> one bit per family + one bit per word in no family -> standard TM -> sentiment
```

## Result

Test accuracy of the TM. Families help with few labels and cost accuracy when they replace words
with all labels, unless they are added next to the words. They do not transfer to Finance.

| Dataset | Labels | Set of words | PPMI-TM | Families added |
| --- | --- | --- | --- | --- |
| IMDB | 200 | 73.80 | **77.87** | 77.64 |
| | 1000 | 79.01 | 81.16 | **81.53** |
| | 40 000 | 87.18 | 83.97 | **87.32** |
| Amazon | 200 | 65.86 | **69.51** | 68.10 |
| | 1000 | 72.45 | 75.63 | **76.52** |
| | 40 000 | 81.92 | 80.06 | **83.11** |
| Finance | 200 | **64.55** | 63.93 | 64.19 |
| | 1000 | 68.78 | 68.47 | **69.35** |
| | 3877 | **71.72** | 70.59 | **71.72** |

## Layout

```text
build_families.py   stage 1: PPMI families; --word-order rebuilds the stored ones exactly
classify.py         stage 2: the TM on the set of words, the families added, or replacing words
classify_added.py   stage 2: all labels, families added, one row per epoch
run.sh              all runs
results/            families_ppmi.json, families_finance.json, classify_*.csv, added_*.csv
logs/               the logs of the runs
```

In `classify_*.csv`, `bow` is the set of words, `bow+families` the families added, `fam+rest` the
PPMI-TM and `random+rest` its random control.
