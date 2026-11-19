# HWF-TM: the Hierarchical Word-Family Tsetlin Machine

This is the folder of the proposed model. An HTM learns word families without labels, and a standard
TM classifies sentiment with those families as input. The goal is higher accuracy with fewer and
shorter clauses. The result is negative, and it locates the limitation in the families the HTM
learns, not in how they are given to the TM.

## Architecture

```text
Stage 1, no labels (pretrain.py, GPU):
  a pair of spans -> HTM with clauses (OR group over A) AND (OR group over B) -> belong together?

  same-dataset pretext: two half-reviews, from the same corpus?   m = 12, 200 clauses,  T = 128
  profile pretext:      two usage profiles, of the same word?     m = 8,  2000 clauses, T = 256

Stage 2, no labels (export_families.py):
  OR groups -> single-word leaves -> a partition of the vocabulary -> word families

Stage 3, labels (classify.py, CPU):
  text -> one bit per family + one bit per word in no family -> standard TM -> sentiment
```

The HTM is not used for classification. A family fires when any of its words occurs, and it replaces
the bits of its words.

## Result

Test accuracy and literals per clause with all labels and 2000 clauses. The inputs named "HWF-TM" are
below the TM on IMDB and Amazon, and their families are no better than random groups of equally
frequent words.

| Input | IMDB | Amazon | Finance |
| --- | --- | --- | --- |
| TM | 87.06 / 137 | 82.11 / 135 | 70.28 / 88 |
| PPMI-TM | 84.31 / 106 | 80.06 / 174 | 70.28 / 89 |
| HWF-TM, profile | 85.56 / 180 | 81.27 / 180 | 69.97 / 95 |
| HWF-TM, same-dataset | 83.76 / 120 | 77.58 / 170 | 70.69 / 98 |

With 1000 labels the PPMI-TM is the most accurate, and 250 of its clauses match a TM with 1000 to 2000
clauses on IMDB. Appendix B has every run, including the random controls.

## Layout

```text
pretrain.py          stage 1: both pretext tasks
families.py          stage 2: OR groups to families, and families to input bits
export_families.py   stage 2: the family lists of all sources in one format
classify.py          stage 3: the TM on every input, with accuracy and clause statistics per epoch
run.sh               all runs
results/             pretrain_*.json, families_*.json, classify_*.csv, clauses_*/
models/              the two pretrained HTMs, reloaded by pretrain.py instead of retrained
```

In `classify_*.csv`, `bow` is the set of words and `<source>-perm` the random control of a family
source. In `classify_same-dataset_full.csv`, `hwf` and `permuted` are the same-dataset families and
their control.
