# Experiments

Each folder is one experiment in the report's chapter "Experiments and results". They build on each
other, in this order:

| Folder | Report section | Question |
| --- | --- | --- |
| [classification/](classification/) | Benchmark classification | How well do the TM and a flat HTM classify sentiment? |
| [documents/](documents/) | Self-supervised learning using the TM and using the HTM | Can both learn a pair task from dataset boundaries alone? |
| [ppmi-tm/](ppmi-tm/) | Do word families help the Tsetlin Machine? | Do good word families help a TM at all? |
| [HWF-TM/](HWF-TM/) | Empirical results, Appendix B | Can an HTM learn such families without labels? |
| [two_leaf_htm/](two_leaf_htm/) | Does a second leaf reduce overfitting? | Does a two-leaf HTM overfit less than a flat one on sentiment? |
| [or_families/](or_families/) | Why the OR groups are poor word families | Why are the HTM's OR groups poor word families? |

## Every experiment has the same layout

```text
<experiment>/
  README.md      what we did, the architecture and the result
  run.sh         the exact commands behind the stored results
  *.py           the code; tm/ and hierarchy_tm/ when one script exists per machine
  results/       the CSV and JSON files behind the report's numbers
  figures/       the generated figures, if the experiment has any
  logs/          the logs of the runs
  models/        trained machines, if the experiment keeps any
```

Code that several experiments use is in [../src/](../src/), and the datasets are described in
[../data/](../data/).

## Reproducing the results

Run an experiment with its `run.sh`. A full run takes hours and overwrites the stored results.
`plot_all.sh` in this folder only redraws the figures and the generated part of Appendix B from the
stored results.

- **Hardware.** Every standard TM runs on one CPU thread of an Intel Xeon Platinum 8168. Every HTM
  runs on an NVIDIA Tesla V100, one job at a time.
- **Exact and approximate results.** The word families and the two pretrained HTMs are rebuilt
  exactly. The accuracies come back within the variation between runs, not digit for digit, because
  the stored runs used a column order that depended on Python's hash seed, and `pyTsetlinMachine`
  exposes no seed. The results in `results/` are therefore the record of what the report states.
