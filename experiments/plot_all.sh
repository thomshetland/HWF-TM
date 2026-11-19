#!/bin/bash
# Redraw every figure from the stored results.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python

$PY experiments/classification/plot_results.py
$PY experiments/documents/plot_results.py
$PY experiments/HWF-TM/plot_results.py
$PY experiments/HWF-TM/plot_control.py
$PY experiments/HWF-TM/plot_literals.py
$PY experiments/HWF-TM/plot_curves.py
$PY experiments/two_leaf_htm/plot_results.py
