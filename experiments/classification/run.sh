#!/bin/bash
# Reproduce the benchmark ("Benchmark classification"): the TM on the CPU, the flat HTM on the GPU.
# Run from anywhere. Each TM run takes up to an hour on one CPU thread.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
HERE=experiments/classification

for dataset in imdb amazon finance; do
    $PY $HERE/tm/$dataset.py
    $PY $HERE/hierarchy_tm/$dataset.py
done
$PY $HERE/plot_results.py
