#!/bin/bash
# Reproduce the HWF-TM experiment ("Empirical results" and Appendix B).
# Run from anywhere. The GPU runs one job at a time; the all-label sweeps take hours on the CPU.
# A finished checkpoint in models/ is reloaded instead of retrained, so delete it to train again.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
HERE=experiments/HWF-TM

# Stage 1: pretrain an HTM on each pretext (GPU).
$PY $HERE/pretrain.py --name same-dataset
# The profile vocabulary is the words in 2 % to 18 % of the reviews. With 1 % to 10 % it holds
# rarer words (1154 instead of 635), and the pretext then reaches 0.70 instead of 0.79.
$PY $HERE/pretrain.py --name profile --task profile --no-remove-shared --min-df 0.02 --max-df 0.18 \
    --alternatives 8 --clauses 2000 --T 256 --epochs 60 --train-pairs 60000

# Stage 2: read the OR groups out as word families, next to the PPMI families.
$PY $HERE/export_families.py

# Stage 3: the standard TM on every input, with each random control (CPU).
$PY $HERE/classify.py --pretrain same-dataset --name same-dataset_full --workers 5
# The TM baseline with all labels is the bow input of same-dataset_full above.
$PY $HERE/classify.py --families profile ppmi --name sources_full --labels 0 --epochs 10 \
    --seeds 1 --clauses 2000 --workers 5 \
    --representations profile profile-perm ppmi ppmi-perm
$PY $HERE/classify.py --families same-dataset profile ppmi --name sources_N1000 --labels 1000 \
    --epochs 20 --seeds 3 --clauses 250 500 1000 2000 --workers 5
$PY $HERE/classify.py --families same-dataset profile ppmi --name sources_N200 --labels 200 \
    --epochs 100 --seeds 1 --clauses 2000 --workers 5
