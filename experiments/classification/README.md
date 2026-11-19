# Classification: the benchmark

This folder has the benchmark of the project: sentiment classification with the standard TM and a
flat HTM on IMDB, Amazon and Financial PhraseBank. Every later experiment is compared with these
numbers. The flat HTM has one leaf over all features, so each clause is an ordinary conjunction and
any difference to the TM comes from the implementation.

## Architecture

```text
text -> set of words, 5000 bits -> TM or flat HTM -> sentiment
```

Both machines use 2000 clauses, T = 1600, s = 5, weighted clauses and 10 epochs.

## Result

Best test accuracy over the epochs. The two machines perform alike.

| Machine | IMDB | Amazon | Finance |
| --- | --- | --- | --- |
| TM | 87.54 | 82.32 | 71.93 |
| Flat HTM | 86.93 | 82.46 | 70.38 |

## Layout

```text
tm/, hierarchy_tm/   one script per dataset and machine
run.sh               all runs
results/             per-epoch accuracy, one CSV per dataset and machine
```
