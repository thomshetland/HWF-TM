#!/bin/bash
# Reproduce the two-leaf experiment: the flat and the two-leaf HTM on the three sentiment datasets.
# Run from anywhere. The GPU runs one job at a time; all six runs take about two hours.
# Every run checkpoints per epoch in models/, so after a container restart, run this again:
# finished runs are only reported and a broken one resumes. Delete models/ to train from scratch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
HERE=experiments/two_leaf_htm

for dataset in imdb amazon finance; do
    for structure in flat two-leaf; do
        $PY $HERE/classify.py --dataset $dataset --structure $structure --seed 1
    done
done
