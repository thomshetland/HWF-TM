# Logical self-supervised learning of language with hierarchical Tsetlin machines

ICT pre-project, University of Agder. Supervisor: Ole-Christoffer Granmo.

Can a hierarchical Tsetlin machine (HTM) learn synonyms without labels, and can those synonyms make a
Tsetlin machine (TM) more accurate and smaller in downstream text classification? We propose the
Hierarchical Word-Family Tsetlin Machine (HWF-TM): an HTM is trained on a self-supervised pretext
task, its OR groups are read out as word families, and each family replaces its words by one bit in
the input of a standard TM.

The HWF-TM does not beat the standard TM, and its families are no better than random groups of equally
frequent words. Word families built with PPMI, without any Tsetlin machine, do help when labels are
scarce. The encoding is therefore not the limitation; learning good families with the HTM is.

## Project layout

```text
experiments/   one folder per experiment, all with the same layout; start with experiments/README.md
src/           code shared by the experiments
data/          the datasets, which are downloaded separately
report/        the chapters of the report
docs/          background on the TM and the HTM, and the logs of what was tried along the way
```

## Getting started

Create the environment with the exact versions. This needs a GPU and `nvcc`, see
[docs/07-environment.md](docs/07-environment.md):

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r pyproject.toml --extra dev --no-binary pycuda
```

Then redraw every figure from the stored results, or rerun a whole experiment:

```bash
bash experiments/plot_all.sh
bash experiments/classification/run.sh
```
