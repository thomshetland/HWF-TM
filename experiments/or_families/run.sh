#!/bin/bash
# Reproduce the OR-family experiment: why the HWF-TM's OR groups are poor word families.
# Run from anywhere. One GPU job at a time. planted.py skips the rows it has already written, so
# after a container restart, run this again.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python               # library 0.3.1, as every stored HTM result
PY035=./.venv-htm035/bin/python     # library 0.3.5, for its new options (README.md)
HERE=experiments/or_families

# 1. Diagnosis of the stored HWF-TM pretraining runs (CPU).
$PY $HERE/diagnose.py

# 2. An OR-level transform equals the leaf read-out; library 0.3.5 with its new options off
#    trains bit-identically to 0.3.1.
$PY $HERE/transform_check.py
$PY $HERE/version_check.py
$PY035 $HERE/version_check.py

# 3. Planted families: which options and settings let the two-slot HTM train at all, and
#    whether its families can be read out group by group.
$PY035 $HERE/planted.py
