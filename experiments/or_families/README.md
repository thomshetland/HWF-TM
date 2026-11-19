# OR families: why the HTM's OR groups are poor word families

The HWF-TM reads the OR groups of a pretrained HTM out as word families. Those families never
helped sentiment classification, while PPMI families did ([../HWF-TM/](../HWF-TM/),
[../ppmi-tm/](../ppmi-tm/)). This experiment asks why. We take apart the stored pretraining runs
and measure on planted families what the two-slot HTM can learn.

## How an OR group learns

The CUDA kernels are the same in library 0.3.1 and in the current upstream `main` (0.3.5).
Feedback to an `OR_ALTERNATIVES` node goes through the AND path. Every alternative that is true
gets Type I or Type II feedback, and every false one forgets at rate 1/s. The alternatives do not
compete. An alternative is therefore rewarded for any word that is present when its clause should
fire. An OR group becomes the set of words that are interchangeable evidence for the pretext
label, whatever that label is.

## Result 1: the stored groups are evidence for the pretext label, not families

`diagnose.py` reads both pretraining runs of the HWF-TM.

| Pretext | Positive clauses | Negative clauses | Slots A and B share words |
| --- | --- | --- | --- |
| Same dataset | 100 of 100 are "IMDB word in A and IMDB word in B" | 89 % contrast the slots (film words against `book`, `buy`, `price`) | 0.03 (Jaccard) |
| Profile | lean like the vocabulary | lean like the vocabulary | 0.36 in positive clauses, 0.00 in negative ones |

- **Same dataset.** A positive group means "some IMDB word". No clause recognises two Amazon
  texts. 42 % of the leaves are conjunctions such as `movie & has`, a group has 0.5 duplicate
  leaves on average, and the members are frequent words: the median rank is 39 of 732.
- **Profile.** The pretext is solved by the same word on both sides: the overlap baseline is
  0.849, and the HTM reaches 0.83. A positive clause `(a|b|c)(A) AND (a|b|c)(B)` is a bundle of
  such matchers, so the members of one group need have nothing in common.

The coherence of the groups is measured against 20 frequency-matched permutations. 0 means no more
alike than random words of the same frequency.

| Family set | Families | Sentiment coherence | Document PMI (control) |
| --- | --- | --- | --- |
| Same dataset, positive groups | 200 | -0.17 | 0.59 (0.45) |
| Same dataset, families used downstream | 137 | +0.09 | 0.50 (0.45) |
| Profile, positive groups | 1716 | +0.03 | 0.56 (0.49) |
| Profile, families used downstream | 112 | -0.00 | 0.63 (0.47) |
| **PPMI, families used downstream** | 866 | **+0.66** | **1.71 (0.36)** |

No HTM family set is more alike in sentiment than random words. The PPMI families are, because
words that occur in the same reviews tend to share their sentiment.

Dataset coherence is in `results/diagnosis.json` too, but it is a fragile measure here: most
words lean towards IMDB, so a group that mixes a few Amazon words with neutral ones scores below
its control. The claims about the dataset axis rest on the slot means above.

## Result 2: beyond a handful of families, no single group is a family

`planted.py` trains the two-slot HTM on synthetic families of six words. Each pair holds one word
per slot, and Y = 1 when both words come from the same family. That is the task of the early
probes (X2 and X7 in `docs/findings-early-probes.md`). After 60 epochs at 16 clauses per family
and T = 8, two things are measured:
- **Accuracy:** final validation accuracy.
- **Groups:** planted families held cleanly by one positive clause, in both slots. That needs at
  least three of the family's six words, 80 % purity and no empty alternative.

| Families | s | Accuracy | Groups | Empty alternatives per group |
| --- | --- | --- | --- | --- |
| 8 | 100 | 0.980 | 6 of 8 | 0.13 |
| 16 | 100 | 0.765 | 2 of 16 | 0.34 |
| 32 | 100 | 0.525 | 0 of 32 | 0.49 |
| 8 | 400 | 0.974 | 8 of 8 | 0.00 |
| 16 | 400 | 0.878 | 0 of 16 | 0.06 |
| 32 | 400 | 0.751 | 0 of 32 | 0.18 |
| 16 | 1000 | 0.836 | 0 of 16 | 0.01 |
| 32 | 1000 | 0.824 | 0 of 32 | 0.06 |

The fall to chance at 32 families is a setting, not a capacity limit. At s = 400 the machine
reaches 0.75, and at s = 1000 it reaches 0.82, because fewer alternatives empty (an external
audit of the project found this first). But above eight families no positive group holds a
family, although the groups are hardly ever empty. A positive group then mixes words from
several families, and the machine classifies by summing many such overlapping unions. A
read-out that takes one group as one family cannot find the families there. Every row has one
seed. The audit's second seed at 32 families gave 0.49 at s = 100 and 0.63 at s = 400, so the
level varies between seeds.

The same script measures two more things:
- **s.** On eight families s = 100 gives 0.88 and s = 30 gives 0.74.
- **The new options of library 0.3.5.**
  - `binary_inference` lowers four families from 0.96 to 0.80.
  - `literal_budget = 1` gives bit-identical results. That is not because leaves hold one word:
    Type II feedback adds absent words, so dead leaves of two or more words do occur.

## What this says about the HWF-TM

The OR mechanism is not the problem. Trained on the sentiment label itself, an HTM with an OR level
learns families such as `great | wonderful | perfect` at 99.5 % polarity purity (X11 in
`docs/findings-early-probes.md`). Our families fail for three reasons:

1. **The pretext label.** An OR group is an equivalence class of the pretext label. "Same dataset"
   makes every IMDB word equivalent, and a task that exact overlap can largely solve gives little
   reason to group words.
2. **The read-out.** Beyond a handful of families the machine keeps them as overlapping unions,
   and no single group is a family (Result 2). A read-out that takes one group as one family
   loses them. On the profile run the HWF-TM's consensus order combines groups, and its families
   occur together more often than random words (Result 1), but they are no more alike in
   sentiment. A read-out cannot give the families a meaning the pretext did not ask for.
3. **Frequency.** A one-word alternative forgets at 1/s per update and is reinforced only when its
   word occurs. Rare words cannot hold an alternative, so groups fill up with frequent words.

Upstream `main` (0.3.4 onwards) has a `transform(X)` that returns each clause's root vote; its CUDA
kernel is still a stub. `transform_check.py` computes the family-level version, the OR node
outputs, on the stored same-dataset run. It equals the leaf read-out exactly, and the root vote is
the product of the two OR outputs. So a transform evaluates the same groups, and gives no other
families than the read-out does.

## Layout

```text
diagnose.py        Result 1: clause structure and coherence of the stored HWF-TM runs (CPU)
planted.py         Result 2: planted families, options, s, scale (GPU, library 0.3.5)
word_stats.py      per-word dataset and sentiment log-odds, document PMI, coherence
version_check.py   library 0.3.5 with its new options off trains bit-identically to 0.3.1
transform_check.py an OR-level transform equals the leaf read-out (GPU, library 0.3.1)
run.sh             everything above, in order
results/           diagnosis.json, planted.csv, transform_check.json, version_check_*.npz
logs/              the logs of the runs
```

Library 0.3.5 is installed in its own environment, `.venv-htm035`, so the project's `.venv`
stays on 0.3.1:

```bash
uv venv --python 3.12 .venv-htm035
VIRTUAL_ENV=.venv-htm035 uv pip install numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.0 \
    pycuda==2026.1 PyHierarchicalTsetlinMachineCUDA==0.3.5 --no-binary pycuda
```

Every run has one seed.
