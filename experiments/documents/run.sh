#!/bin/bash
# Reproduce the self-supervised pair task on the TM and the HTM ("Self-supervised learning using
# the Tsetlin Machine" and "... the Hierarchical Tsetlin Machine").
# Run from anywhere. The GPU runs one job at a time; the IMDB vs. Amazon runs take over an hour each.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
HERE=experiments/documents

for task in imdb_finance awz_finance imdb_awz; do
    $PY $HERE/tm/$task.py
done
for structure in flat two-leaf; do
    for task in imdb_finance awz_finance; do
        $PY $HERE/hierarchy_tm/$task.py --structure $structure --seed 1
    done
    for seed in 1 2 3; do
        $PY $HERE/hierarchy_tm/imdb_awz.py --structure $structure --seed $seed
    done
done
$PY $HERE/plot_results.py
