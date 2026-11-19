#!/bin/bash
# Reproduce the PPMI-TM experiment ("Do word families help the Tsetlin Machine?").
# Run from anywhere. The full-label runs take hours on one CPU thread per job.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
HERE=experiments/ppmi-tm

# Stage 1: word families. --word-order keeps the column order of the stored results.
$PY $HERE/build_families.py --datasets imdb amazon --no-balance --k 1000 --name families_ppmi \
    --word-order $HERE/results/families_ppmi.json
$PY $HERE/build_families.py --datasets finance --name families_finance \
    --word-order $HERE/results/families_finance.json

# Stage 2a: 200 and 1000 labels, and all labels.
$PY $HERE/classify.py --families families_ppmi --conditions bow bow+families fam+rest random+rest \
    --labels 200 1000 --seeds 2 --workers 4 --name classify_N200_N1000
$PY $HERE/classify.py --families families_ppmi --conditions fam+rest --labels 0 --seeds 1 \
    --datasets imdb amazon --workers 2 --name classify_full
$PY $HERE/classify.py --families families_ppmi --conditions bow fam+rest random+rest --labels 0 \
    --seeds 2 --datasets finance --workers 2 --name classify_full_finance
# Families built from the Finance sentences themselves.
$PY $HERE/classify.py --families families_finance --conditions fam+rest random+rest \
    --labels 200 1000 --seeds 2 --workers 2 --name classify_finance_families

# Stage 2b: the families added to the set of words, with all labels.
for dataset in imdb amazon finance; do
    $PY $HERE/classify_added.py --dataset "$dataset" --families families_ppmi
done
$PY $HERE/classify_added.py --dataset amazon --families families_ppmi --seed 1
$PY $HERE/classify_added.py --dataset amazon --seed 1
for dataset in imdb amazon; do
    $PY $HERE/classify_added.py --dataset "$dataset" --families families_ppmi --control
done
