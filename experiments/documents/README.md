# Documents: the self-supervised pair task

This folder has the first self-supervised pretext experiment, where we try to classify if two
samples are from the same dataset using the TM and the HTM. The labels come from which file a text
was read from, so no annotation is needed. We use it to check that the hierarchical implementation
reproduces the standard machine before the hierarchy is given OR alternatives.

## Architecture

```text
text A, text B -> [A | B], 2 x 5000 bits -> TM, flat HTM or two-leaf HTM -> same dataset?
```

The flat HTM has one leaf over all 10 000 bits, and the two-leaf HTM has one leaf per text, joined by
AND. Both encode a single conjunction. All machines use 2000 clauses, T = 1600, s = 5 and 20 epochs.

## Result

Test accuracy after 20 epochs. The HTM reproduces the TM, and what is learned is mainly topic: the
task is easy when the datasets differ in subject and hard when both hold consumer reviews.

| Dataset pair | TM | Flat HTM | Two-leaf HTM |
| --- | --- | --- | --- |
| IMDB vs. financial news | 99.56 | 99.66 | 99.64 |
| Amazon vs. financial news | 95.36 | 94.74 | 94.20 |
| IMDB vs. Amazon | 82.38 | 81.53 | 81.09 |

The HTM values for IMDB vs. Amazon are the mean over three seeds.

## Layout

```text
tm/, hierarchy_tm/   one script per dataset pair and machine
run.sh               all runs and the figure
results/             per-epoch metrics, one CSV per dataset pair, machine, structure and seed
figures/             tm_vs_htm_test_accuracy
logs/                the logs of the IMDB vs. Amazon HTM runs
```
