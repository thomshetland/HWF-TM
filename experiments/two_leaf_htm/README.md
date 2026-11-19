# Two-leaf HTM: does a second leaf reduce overfitting?

This folder has the benchmark classification of [../classification/](../classification/) repeated
with a two-leaf HTM. The hypothesis, from Ole-Christoffer Granmo, is that two leaves overfit less
than one, because a clause no longer copies a training text in one piece. In the pair task of
[../documents/](../documents/) the two-leaf HTM had a smaller gap between train and test accuracy
than the flat HTM. Here we test whether that also holds on sentiment classification.

## Architecture

```text
text -> set of words, 5000 bits -> flat HTM or two-leaf HTM -> sentiment
```

| Structure | Hierarchy | Clause |
| --- | --- | --- |
| Flat | `((AND_GROUP, 5000), (OR_ALTERNATIVES, 1))` | one leaf over all words |
| Two-leaf | `((AND_GROUP, 2500), (AND_GROUP, 2))` | one leaf per half of the vocabulary, joined by AND |

Both structures encode a single conjunction over the same 5000 words, and an untrained clause
votes exactly 1 in both, so T carries over. The difference is in the feedback: each leaf decides
on its own whether it is updated, where a flat clause makes that decision once for all its
literals.

A text has no natural two parts, so the words are dealt to the two leaves at random, with the
same shuffle in every run. Without it one leaf would hold the 2500 most common words and the
other the 2500 rarest, because the vocabulary is in document-frequency order.

Everything else is the benchmark: the same splits and vocabulary, 2000 clauses, T = 1600, s = 5,
weighted clauses and 10 epochs, with one seed per run.

## Result

After 10 epochs, one seed. The gap is train minus test accuracy.

| Dataset | Structure | Train | Test | Gap | Best test | Words per clause | Negated words per clause |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IMDB | Flat | 94.60 | 86.82 | 7.78 | 87.37 | 0.42 | 127.4 |
| IMDB | Two-leaf | 94.07 | 86.52 | 7.55 | 86.80 | 0.46 | 119.5 |
| Amazon | Flat | 87.68 | 81.30 | 6.38 | 82.27 | 0.63 | 124.0 |
| Amazon | Two-leaf | 86.30 | 81.02 | 5.28 | 81.61 | 0.70 | 91.1 |
| Finance | Flat | 78.82 | 70.59 | 8.24 | 70.59 | 0.80 | 92.5 |
| Finance | Two-leaf | 75.01 | 70.38 | 4.62 | 70.38 | 0.82 | 64.1 |

The two-leaf HTM has the smaller gap on all three datasets, but only because it fits the training
data less. Its test accuracy is 0.2 to 0.3 points below the flat HTM at the last epoch and 0.2 to
0.7 below at the best epoch, which is less than either machine moves from one epoch to the next.
So the hypothesis holds for memorisation and not for generalisation: the second leaf does not
make the machine better on unseen texts.

Compared at the same train accuracy instead of the same epoch, the two-leaf HTM is not ahead
either. Its last epoch on IMDB (train 94.07, test 86.52) matches the flat HTM's epoch 8
(train 94.26, test 87.13), and on Amazon (train 86.30, test 81.02) the flat HTM's epoch 4
(train 86.05, test 81.60).

The clauses do differ. Both structures hold less than one word per clause, so neither copies the
words of a training text. What a clause accumulates is negated words, and the two-leaf HTM
accumulates fewer of them: 6 % fewer on IMDB, 26 % on Amazon and 31 % on Finance. The cause has
not been isolated.

The flat runs reproduce the benchmark, which has 86.93, 82.46 and 70.38 as best test accuracy.
The curves are noisy, most of all on Finance, where the two-leaf HTM drops to 59.03 at epoch 9
and is back at 70.38 at epoch 10. With one seed, a difference below one point is not a result.

## Layout

```text
classify.py       one run: --dataset imdb|amazon|finance, --structure flat|two-leaf
run.sh            all six runs
results/          per-epoch accuracy and literals per clause, one CSV per run
models/           the machine after its last finished epoch, one checkpoint per run
```

A run checkpoints after every epoch. Running the same command again resumes it, and a finished
run is only reported again. Delete its file in `models/` to train from scratch.
