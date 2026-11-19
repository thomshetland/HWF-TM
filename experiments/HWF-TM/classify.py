#!/usr/bin/python
"""Stage 3 of the HWF-TM: sentiment classification with a standard TM on word-family bits.

Three representations of the same texts, trained with the same TM:

  bow       the baseline: one bit per word of the dataset's 5000-word vocabulary.
  hwf       one bit per word family from pretraining, then one bit per vocabulary word that is in
            no family (families.FamilyEncoder).
  permuted  as hwf, with the families' words swapped for random words of similar frequency: the
            same number and sizes of families, the grouping destroyed. If this does as well as
            hwf, what helps is the narrower input, not what pretraining learned.

With --families, the families come from family lists written by export_families.py instead of a
pretraining run, so several sources can be compared in one sweep. Each source <s> then gives two
representations, <s> and <s>-perm (its permuted control), next to bow.

The claim under test is that families let a smaller machine with shorter clauses do the work of
a larger one, so every representation is trained at several clause counts, with T scaled to keep
the baseline's ratio (T = 1600 at 2000 clauses). After every epoch the script records accuracy
and the size of what was learned: included literals per clause, and how many of them are family
literals. At the end of a job the highest-weighted clauses are written out with family names.

Rows go to results/classify_<name>.csv, one per (job, epoch). A job whose last epoch is missing
is rerun from scratch, so the script can simply be restarted after the container dies.

pyTsetlinMachine runs on one CPU thread and exposes no seed, so the machine's own randomness is
not fixed; --seeds repeats every job. Keep --workers at 4 or below while a GPU job runs (the
container has a 6-CPU quota). Run from the repo root, after pretrain.py:

  ./.venv/bin/python experiments/HWF-TM/classify.py --pretrain same-dataset
  ./.venv/bin/python experiments/HWF-TM/classify.py --pretrain same-dataset --summary-only
  ./.venv/bin/python experiments/HWF-TM/classify.py --families profile ppmi --name sources_full
"""

import argparse
import csv
import fcntl
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # make `src` importable when run as a script

from families import (  # noqa: E402
    FamilyEncoder,
    load_pretrain,
    permuted_families,
    word_families,
)
from src.common import DATASETS, dataset_vocabulary, load_splits, tokens  # noqa: E402

RESULTS = HERE / "results"

KEY = ("dataset", "labels", "seed", "representation", "clauses")
FIELDS = KEY + (
    "T",
    "epoch",
    "features",
    "families",
    "train_accuracy",
    "test_accuracy",
    "train_seconds",
    "literals_per_clause",
    "positive_per_clause",
    "negated_per_clause",
    "family_literals_per_clause",
    "empty_clauses",
    "total_literals",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrain", default=None, help="results/pretrain_<name>.json (default same-dataset)"
)
parser.add_argument(
    "--families",
    nargs="+",
    default=None,
    help="results/families_<name>.json; several allowed",
)
parser.add_argument("--name", default=None, help="output name, default the pretrain name")
parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
parser.add_argument("--representations", nargs="+", default=None, help="default: all of them")
parser.add_argument("--clauses", type=int, nargs="+", default=[250, 500, 1000, 2000])
parser.add_argument("--T-per-clause", type=float, default=0.8, help="T = this * clauses")
parser.add_argument("--s", type=float, default=5.0)
parser.add_argument("--epochs", type=int, default=10)
# 0 means every training label, the point comparable to experiments/classification/tm.
parser.add_argument("--labels", type=int, nargs="+", default=[0])
parser.add_argument("--seeds", type=int, default=1, help="repeats of every job")
parser.add_argument("--polarity", choices=("all", "positive", "negative"), default="all")
parser.add_argument("--overlap", choices=("partition", "keep"), default="partition")
parser.add_argument("--min-family-size", type=int, default=2)
parser.add_argument("--family-max-df", type=float, default=0.95)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--show-clauses", type=int, default=5, help="top clauses per class to write")
parser.add_argument("--summary-only", action="store_true")
args = parser.parse_args()
if args.families and args.pretrain:
    parser.error("give --pretrain or --families, not both")
if args.families and not args.name:
    parser.error("--families needs --name")
if not args.families:
    args.pretrain = args.pretrain or "same-dataset"

NAME = args.name or args.pretrain
CSV_PATH = RESULTS / f"classify_{NAME}.csv"
CLAUSE_DIR = RESULTS / f"clauses_{NAME}"


def job_key(row):
    return tuple(str(row[k]) for k in KEY)


def finished_jobs():
    """Keys of jobs that reached their last epoch; rows of unfinished jobs are dropped from the CSV.

    A job killed halfway leaves some epoch rows behind. Rerunning it from epoch 1 would then
    write those epochs twice, so they are removed before anything is scheduled.
    """
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open() as fh:
        rows = list(csv.DictReader(fh))
    done = {job_key(r) for r in rows if int(r["epoch"]) == args.epochs}
    kept = [r for r in rows if job_key(r) in done]
    if len(kept) < len(rows):
        with CSV_PATH.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(kept)
    return done


def summarize():
    """Final-epoch test accuracy and literals per clause, mean over seeds, per dataset and size."""
    if not CSV_PATH.exists():
        print(f"no results in {CSV_PATH}")
        return
    with CSV_PATH.open() as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["epoch"]) == args.epochs]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["labels"], int(r["clauses"]), r["representation"])].append(r)
    representations = list(dict.fromkeys(r["representation"] for r in rows))
    print(f"\nFinal epoch ({args.epochs}), mean over seeds: test accuracy % / literals per clause")
    print(
        f"{'dataset':8s} {'labels':>6s} {'clauses':>7s}"
        + "".join(f"{r:>18s}" for r in representations)
    )
    for dataset, labels, clauses in sorted({k[:3] for k in cells}):
        line = f"{dataset:8s} {labels:>6s} {clauses:7d}"
        for rep in representations:
            group = cells.get((dataset, labels, clauses, rep))
            if group:
                acc = np.mean([float(r["test_accuracy"]) for r in group])
                lits = np.mean([float(r["literals_per_clause"]) for r in group])
                line += f"{acc:9.2f} / {lits:5.1f}"
            else:
                line += f"{'-':>18s}"
        print(line)


if args.summary_only:
    summarize()
    raise SystemExit


# ---------------------------------------------------------------- data, built once before forking

splits = load_splits()
# Each representation is a family list; bow has none, so its encoder is the plain bag of words.
# A permuted control is drawn once per source, so every dataset and job sees the same one.
SOURCES = {"bow": []}
if args.families:
    for source in args.families:
        with (RESULTS / f"families_{source}.json").open() as fh:
            exported = json.load(fh)
        SOURCES[source] = exported["families"]
        SOURCES[f"{source}-perm"] = permuted_families(
            exported["families"], exported["ranked_words"], np.random.default_rng(0)
        )
        print(f"{source}: {len(exported['families'])} families ({exported['source']})")
else:
    pretrained = load_pretrain(args.pretrain)
    families = word_families(
        pretrained["groups"], args.polarity, args.overlap, args.min_family_size
    )
    SOURCES["hwf"] = families
    SOURCES["permuted"] = permuted_families(families, pretrained["words"], np.random.default_rng(0))
    print(
        f"{len(pretrained['groups'])} OR groups -> {len(families)} families "
        f"({args.polarity}, {args.overlap})"
    )
args.representations = args.representations or list(SOURCES)

DATA = {}
for name in args.datasets:
    train, test = splits[name]["train"], splits[name]["test"]
    classes = sorted({row["sentiment"] for row in train})
    vocabulary = dataset_vocabulary(train)
    tokens_train = [tokens(r) for r in train]
    tokens_test = [tokens(r) for r in test]
    index = {c: i for i, c in enumerate(classes)}
    entry = {
        "classes": classes,
        "Y_train": np.array([index[r["sentiment"]] for r in train], dtype=np.uint32),
        "Y_test": np.array([index[r["sentiment"]] for r in test], dtype=np.uint32),
    }
    for rep in args.representations:
        encoder = FamilyEncoder(SOURCES[rep], vocabulary, args.family_max_df).fit(tokens_train)
        entry[rep] = {
            "X_train": encoder.transform(tokens_train),
            "X_test": encoder.transform(tokens_test),
            "columns": encoder.columns,
            "families": encoder.n_families,
        }
        width = entry[rep]["X_train"].shape[1]
        print(f"  {name:8s} {rep:13s} {width:5d} bits ({encoder.n_families} families)")
    DATA[name] = entry


# ---------------------------------------------------------------- one job


def clause_statistics(tm, n_families):
    """Included literals per clause, decoded from the automata's action bits of every class.

    get_state() returns, per class, the clause weights and the automaton states as bit planes
    [clause][chunk][state bit]; the most significant plane is the include action (checked
    against ta_action). With negated literals appended, features [0, F/2) are the inputs and
    [F/2, F) their negations.
    """
    F, C = tm.number_of_features, tm.number_of_clauses
    chunks, bits = tm.number_of_ta_chunks, tm.number_of_state_bits
    shifts = np.arange(32, dtype=np.uint32)
    include, weights = [], []
    for clause_weights, states in tm.get_state():
        msb = states.reshape(C, chunks, bits)[:, :, bits - 1]
        include.append(
            ((msb[:, :, None] >> shifts) & 1).reshape(C, chunks * 32)[:, :F].astype(bool)
        )
        weights.append(clause_weights.copy())
    include = np.stack(include)  # (classes, clauses, F)
    half = F // 2
    positive = include[:, :, :half].sum(axis=2)
    negated = include[:, :, half:].sum(axis=2)
    family = include[:, :, :n_families].sum(axis=2) + include[:, :, half : half + n_families].sum(
        axis=2
    )
    total = positive + negated
    stats = {
        "literals_per_clause": float(total.mean()),
        "positive_per_clause": float(positive.mean()),
        "negated_per_clause": float(negated.mean()),
        "family_literals_per_clause": float(family.mean()),
        "empty_clauses": float((total == 0).mean()),
        "total_literals": int(total.sum()),
    }
    return stats, include, np.stack(weights)


def render_clauses(include, weights, columns, classes, k):
    """The k highest-weighted clauses voting for each class, with literal names."""
    half = include.shape[2] // 2
    lines = []
    for c, label in enumerate(classes):
        # Even-indexed clauses vote for the class, odd ones against it (checked on a toy rule).
        voting_for = np.arange(0, include.shape[1], 2)
        for j in voting_for[np.argsort(-weights[c, voting_for])][:k]:
            literals = [columns[f] for f in np.flatnonzero(include[c, j, :half])]
            literals += ["NOT " + columns[f] for f in np.flatnonzero(include[c, j, half:])]
            lines.append(
                f"{label:9s} w={weights[c, j]:4d}  " + (" AND ".join(literals) or "(empty)")
            )
        lines.append("")
    return "\n".join(lines)


def run(job):
    dataset, labels, seed, rep, clauses = job
    data = DATA[dataset]
    X_train, Y_train = data[rep]["X_train"], data["Y_train"]
    if labels:
        # The same labeled texts for every representation and clause count of a seed.
        chosen = np.sort(np.random.default_rng(seed).choice(len(Y_train), labels, replace=False))
        X_train, Y_train = X_train[chosen], Y_train[chosen]
    X_test, Y_test = data[rep]["X_test"], data["Y_test"]
    T = round(args.T_per_clause * clauses)
    n_families = data[rep]["families"]

    # pyTsetlinMachine shuffles with NumPy's global generator, which every forked worker
    # inherits in the same state; seeding it per job makes --seeds real repeats.
    np.random.seed(seed)
    tm = MultiClassTsetlinMachine(clauses, T, args.s, weighted_clauses=True)
    for epoch in range(1, args.epochs + 1):
        start = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        train_seconds = time() - start
        stats, include, weights = clause_statistics(tm, n_families)
        row = {
            "dataset": dataset,
            "labels": labels,
            "seed": seed,
            "representation": rep,
            "clauses": clauses,
            "T": T,
            "epoch": epoch,
            "features": X_train.shape[1],
            "families": n_families,
            "train_accuracy": f"{100 * (tm.predict(X_train) == Y_train).mean():.4f}",
            "test_accuracy": f"{100 * (tm.predict(X_test) == Y_test).mean():.4f}",
            "train_seconds": f"{train_seconds:.2f}",
        } | {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in stats.items()}
        # Several workers append to one file; the lock keeps their rows whole.
        with CSV_PATH.open("a", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            fcntl.flock(fh, fcntl.LOCK_UN)
        print(
            f"{dataset} N={labels or 'all'} seed {seed} {rep} C={clauses} #{epoch}: "
            f"test {row['test_accuracy']} literals/clause {row['literals_per_clause']}",
            flush=True,
        )

    text = render_clauses(
        include, weights, data[rep]["columns"], data["classes"], args.show_clauses
    )
    path = CLAUSE_DIR / f"{dataset}_N{labels}_s{seed}_{rep}_C{clauses}.txt"
    path.write_text(text)
    return job


if __name__ == "__main__":
    CLAUSE_DIR.mkdir(parents=True, exist_ok=True)
    done = finished_jobs()
    jobs = [
        (dataset, labels, seed, rep, clauses)
        for clauses in sorted(args.clauses)  # small machines first: results arrive early
        for dataset in args.datasets
        for labels in args.labels
        for seed in range(1, args.seeds + 1)
        for rep in args.representations
    ]
    pending = [j for j in jobs if tuple(str(v) for v in j) not in done]
    print(f"\n{len(jobs)} jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run\n")
    # fork: the workers inherit DATA instead of rebuilding it.
    with mp.get_context("fork").Pool(args.workers) as pool:
        for _ in pool.imap_unordered(run, pending):
            pass
    summarize()
