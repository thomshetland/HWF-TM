#!/bin/bash
# Reproduce the OR-family experiment: why the HWF-TM's OR groups are poor word families, and
# whether a pretext built to need families gets better ones.
# Run from anywhere. One GPU job at a time; the whole script takes about an hour.
# Pretraining checkpoints every epoch in models/, so after a container restart, run this again.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python               # library 0.3.1, as every stored HTM result
PY035=./.venv-htm035/bin/python     # library 0.3.5, for its new options (README.md)
HERE=experiments/or_families

mkdir -p $HERE/logs

# 1. Diagnosis of the stored HWF-TM pretraining runs (CPU).
$PY $HERE/diagnose.py | tee $HERE/logs/diagnose.log

# 2. An OR-level transform equals the leaf read-out; library 0.3.5 with its new options off
#    trains bit-identically to 0.3.1.
$PY $HERE/transform_check.py | tee $HERE/logs/transform_check.log
$PY $HERE/version_check.py
$PY035 $HERE/version_check.py

# 3. Planted families: which options and settings let the two-slot HTM train at all, and
#    whether its families can be read out group by group.
$PY035 $HERE/planted.py | tee $HERE/logs/planted.log

# 4. The PPMI-neighbour pretext: the 96 most frequent words, the size of the planted runs, and
#    the 300 most frequent with four times the clauses. Then their families' coherence (CPU).
$PY $HERE/pretrain.py --name ppmi-pairs_V96 --vocabulary 96 --neighbours 5 --alternatives 12 \
    --clauses 256 --T 8 --train-pairs 12000 --epochs 60 | tee $HERE/logs/pretrain_ppmi-pairs_V96.log
$PY $HERE/pretrain.py --name ppmi-pairs_V300 --vocabulary 300 --clauses 1024 --T 8 \
    --epochs 40 | tee $HERE/logs/pretrain_ppmi-pairs_V300.log
$PY $HERE/evaluate.py ppmi-pairs_V96 ppmi-pairs_V300 | tee $HERE/logs/evaluate.log
