#!/usr/bin/python
# ruff: noqa: E501  -- a report generator; the LaTeX f-strings read better unbroken.
"""Generate the result tables of the report's appendix from the HWF-TM result files.

Writes results/appendix_generated.tex: pretext training history, the complete lists of learned
word families, the full 1000-label sweep (mean +- sd over the three label draws, clause length,
model size, training time), the all-label runs, the 200-label learning curves and a few example
clauses. The report's appendixB.tex is the user's header followed by this file's content.

Numbers in running text and table cells are set in math mode, as the report requires.

Run from the repo root:  ./.venv/bin/python experiments/HWF-TM/make_appendix.py
"""

import csv
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "appendix_generated.tex"
DATASETS = (("imdb", "IMDB"), ("amazon", "Amazon"), ("finance", "Finance"))
CLAUSES = (250, 500, 1000, 2000)
# (representation in the CSVs, label in the tables)
INPUTS = (
    ("bow", "TM"),
    ("ppmi", "PPMI-TM"),
    ("ppmi-perm", "PPMI control"),
    ("profile", "HWF-TM (profile)"),
    ("profile-perm", "HWF-TM (profile) control"),
    ("same-dataset", "HWF-TM (same-dataset)"),
    ("same-dataset-perm", "HWF-TM (same-dataset) control"),
)
LABEL = dict(INPUTS)


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def m(x, nd=1):
    """A number in math mode."""
    return f"${x:.{nd}f}$"


def pm(values, nd=1):
    return f"${st.mean(values):.{nd}f} \\pm {st.pstdev(values):.{nd}f}$"


def words(ws):
    return "\\textit{" + ", ".join(ws) + "}"


def family_list(families):
    fams = sorted(families, key=lambda f: (-len(f), f))
    return (
        "\\begingroup\\small\\raggedright\\sloppy\n"
        + ";\n".join(words(f) for f in fams)
        + ".\n\\par\\endgroup\n"
    )


out = []
P = out.append

# ------------------------------------------------------------------ pretext training
same = json.load((RESULTS / "pretrain_same-dataset.json").open())
prof = json.load((RESULTS / "pretrain_profile.json").open())
hs = {e["epoch"]: e["val_accuracy"] for e in same["history"]}
hp = {e["epoch"]: e["val_accuracy"] for e in prof["history"]}
epochs = [1, 2, 3, 4, 5] + list(range(10, 61, 5))
P(r"""
\section{Self-supervised pretext training}
\label{app:pretext}

Tables~\ref{tab:app-pretext-epochs} and~\ref{tab:app-pretext-summary} give the training history
and the summary of the two HTM pretext runs whose OR groups are read out as word families
(Chapter~\ref{chap:hwftm}). Validation accuracy is measured on $2000$ held-out pairs after every
epoch. The lexical-overlap baseline is the accuracy of the best threshold on the number of words
the two texts of a pair share; a machine that does not exceed it has learned nothing beyond word
overlap.

\begin{table}[H]
\centering
\small
\caption{Validation accuracy (\%) of the HTM on the two pretext tasks per epoch. The same-dataset
pretext was trained for $40$ epochs and the profile pretext for $60$.}
\label{tab:app-pretext-epochs}
\begin{tabular}{rcc}
\toprule
Epoch & Same-dataset pretext & Profile pretext \\
\midrule
""")
for e in epochs:
    a = m(100 * hs[e]) if e in hs else "--"
    b = m(100 * hp[e]) if e in hp else "--"
    P(f"${e}$ & {a} & {b} \\\\\n")
P(r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\caption{Summary of the two pretext runs. Empty leaves are the fraction of OR alternatives that
include no literal after training; such leaves are vacuously true. Training time is the sum of
the per-epoch training times on the GPU.}
\label{tab:app-pretext-summary}
\begin{tabular}{lcc}
\toprule
 & Same-dataset pretext & Profile pretext \\
\midrule
""")
cs, cp = same["config"], prof["config"]
P(
    f"Training pairs & ${cs['train_pairs'] // 1000}\\,000$ & ${cp['train_pairs'] // 1000}\\,000$ \\\\\n"
)
P(
    f"Pretext vocabulary & ${same['structure'][0][1]}$ words & ${prof['structure'][0][1]}$ words \\\\\n"
)
P(f"Alternatives per OR group ($m$) & ${cs['alternatives']}$ & ${cp['alternatives']}$ \\\\\n")
P(
    f"Clauses, $T$, $s$ & ${cs['clauses']}$, ${cs['T']}$, ${cs['s']:.0f}$ & ${cp['clauses']}$, ${cp['T']}$, ${cp['s']:.0f}$ \\\\\n"
)
P(f"Epochs & ${cs['epochs']}$ & ${cp['epochs']}$ \\\\\n")
P(f"Final validation accuracy & {m(100 * hs[max(hs)])}\\% & {m(100 * hp[max(hp)])}\\% \\\\\n")
P(
    f"Best validation accuracy (epoch) & {m(100 * max(hs.values()))}\\% (${max(hs, key=hs.get)}$) & {m(100 * max(hp.values()))}\\% (${max(hp, key=hp.get)}$) \\\\\n"
)
P(
    f"Lexical-overlap baseline & {m(100 * same['overlap_baseline'])}\\% & {m(100 * prof['overlap_baseline'])}\\% \\\\\n"
)
P(
    f"Empty leaves after training & {m(100 * same['empty_per_group'] / cs['alternatives'])}\\% & {m(100 * prof['empty_per_group'] / cp['alternatives'])}\\% \\\\\n"
)
ts = sum(e["train_seconds"] for e in same["history"])
tp = sum(e["train_seconds"] for e in prof["history"])
P(f"Training time & ${ts / 60:.1f}$ min & ${tp / 60:.1f}$ min \\\\\n")
P(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ families
fam = {
    n: json.load((RESULTS / f"families_{n}.json").open())
    for n in ("profile", "same-dataset", "ppmi")
}
P(r"""
\section{Word families}
\label{app:families}

This section lists the word families used in Chapter~\ref{chap:experiments}, each family in
italics and families separated by semicolons, largest first. A family replaces its member words by
one input bit in the word-family encoding.
""")
for key, title in (
    ("profile", "Families of the profile pretext"),
    ("same-dataset", "Families of the same-dataset pretext"),
):
    fams = fam[key]["families"]
    P(
        f"\n\\subsection{{{title}}}\n\nAll ${len(fams)}$ families covering ${sum(map(len, fams))}$ words.\n\n"
    )
    P(family_list(fams))
fams = fam["ppmi"]["families"]
sample = random.Random(0).sample(fams, 40)
P(
    f"\n\\subsection{{PPMI families}}\n\nA random sample of $40$ of the ${len(fams)}$ PPMI families, which together cover ${sum(map(len, fams))}$ words. The complete list is part of the project repository.\n\n"
)
P(family_list(sample))

# ------------------------------------------------------------------ 1000 labels
R = defaultdict(list)
for r in rows(RESULTS / "classify_sources_N1000.csv"):
    R[(r["dataset"], r["representation"], int(r["clauses"]), int(r["epoch"]))].append(r)


def col(key, field):
    return [float(r[field]) for r in R[key]]


def time_total(dataset, rep, clauses, epochs):
    per_seed = defaultdict(float)
    for e in range(1, epochs + 1):
        for r in R[(dataset, rep, clauses, e)]:
            per_seed[r["seed"]] += float(r["train_seconds"])
    return st.mean(per_seed.values())


P(r"""
\section{Downstream classification with 1000 labeled texts}
\label{app:n1000}

Tables~\ref{tab:app-n1000-accuracy} and~\ref{tab:app-n1000-size} give the complete results of the
clause sweep with $1000$ labeled training texts (Section~\ref{sec:results}): every input,
including the random controls, at $250$ to $2000$ clauses, after $20$ epochs, as the mean over the
three label draws. The TM uses $s = 5.0$, weighted clauses and $T = 0.8 \times$ the number of
clauses.

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Test accuracy (\%) with $1000$ labeled training texts, mean $\pm$ standard deviation over
three label draws.}
\label{tab:app-n1000-accuracy}
\begin{tabular}{llcccc}
\toprule
Dataset & Input & $250$ clauses & $500$ clauses & $1000$ clauses & $2000$ clauses \\
\midrule
""")
for i, (d, D) in enumerate(DATASETS):
    for j, (rep, label) in enumerate(INPUTS):
        cells = " & ".join(pm(col((d, rep, c, 20), "test_accuracy")) for c in CLAUSES)
        P(f"{D if j == 0 else ''} & {label} & {cells} \\\\\n")
    if i < 2:
        P("\\midrule\n")
P(r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Clause length, model size and training time with $1000$ labeled training texts, mean over
three label draws. Literals per clause are the included literals, positive and negated, averaged
over all clauses; \emph{pos.} is the positive literals per clause at $2000$ clauses. Total literals
are given in thousands. Training time is the sum over the $20$ epochs on one CPU thread.}
\label{tab:app-n1000-size}
\begin{tabular}{llrrrrrrr}
\toprule
 & & \multicolumn{4}{c}{Literals per clause} & \multicolumn{3}{c}{$2000$ clauses} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-9}
Dataset & Input & $250$ & $500$ & $1000$ & $2000$ & pos. & total (k) & time (s) \\
\midrule
""")
for i, (d, D) in enumerate(DATASETS):
    for j, (rep, label) in enumerate(INPUTS):
        lits = " & ".join(
            m(st.mean(col((d, rep, c, 20), "literals_per_clause")), 0) for c in CLAUSES
        )
        pos = m(st.mean(col((d, rep, 2000, 20), "positive_per_clause")), 2)
        tot = m(st.mean(col((d, rep, 2000, 20), "total_literals")) / 1000, 0)
        tt = m(time_total(d, rep, 2000, 20), 0)
        P(f"{D if j == 0 else ''} & {label} & {lits} & {pos} & {tot} & {tt} \\\\\n")
    if i < 2:
        P("\\midrule\n")
P(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ all labels
F = {}
for r in rows(RESULTS / "classify_sources_full.csv"):
    F.setdefault((r["dataset"], r["representation"], int(r["clauses"])), []).append(r)
renamed = {"bow": "bow", "hwf": "same-dataset", "permuted": "same-dataset-perm"}
M = {}
for r in rows(RESULTS / "classify_same-dataset_full.csv"):
    M.setdefault((r["dataset"], renamed[r["representation"]], int(r["clauses"])), []).append(r)
for k, v in M.items():
    F.setdefault(k, v)


def final(key, epoch=10):
    return next(r for r in F[key] if int(r["epoch"]) == epoch)


def total_time(key):
    return sum(float(r["train_seconds"]) for r in F[key])


P(r"""
\section{Downstream classification with all labeled texts}
\label{app:full}

Table~\ref{tab:app-full} gives the complete results with the full training split and $2000$
clauses, the benchmark setting, after $10$ epochs; one run per input.
Table~\ref{tab:app-full-sweep} gives the clause sweep with all labels, which was run for the TM
and the same-dataset families only.

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{All labeled texts, $2000$ clauses, $T = 1600$, $10$ epochs. Width is the number of input
features; total literals are in thousands; training time is the sum over the $10$ epochs on one
CPU thread.}
\label{tab:app-full}
\begin{tabular}{llrrrrrrr}
\toprule
Dataset & Input & Width & Train acc. & Test acc. & Lit./clause & pos. & total (k) & time (min) \\
\midrule
""")
for i, (d, D) in enumerate(DATASETS):
    for j, (rep, label) in enumerate(INPUTS):
        r = final((d, rep, 2000))
        P(
            f"{D if j == 0 else ''} & {label} & ${r['features']}$ & {m(float(r['train_accuracy']), 2)} & {m(float(r['test_accuracy']), 2)} & "
            f"{m(float(r['literals_per_clause']), 0)} & {m(float(r['positive_per_clause']), 2)} & {m(float(r['total_literals']) / 1000, 0)} & "
            f"{m(total_time((d, rep, 2000)) / 60, 0)} \\\\\n"
        )
    if i < 2:
        P("\\midrule\n")
P(r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Clause sweep with all labeled texts and $T = 0.8 \times$ the number of clauses, $10$
epochs, one run per cell: test accuracy (\%) and literals per clause.}
\label{tab:app-full-sweep}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{Test accuracy (\%)} & \multicolumn{4}{c}{Literals per clause} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
Dataset & Input & $250$ & $500$ & $1000$ & $2000$ & $250$ & $500$ & $1000$ & $2000$ \\
\midrule
""")
for i, (d, D) in enumerate(DATASETS):
    for j, rep in enumerate(("bow", "same-dataset", "same-dataset-perm")):
        acc = " & ".join(m(float(final((d, rep, c))["test_accuracy"]), 2) for c in CLAUSES)
        lit = " & ".join(m(float(final((d, rep, c))["literals_per_clause"]), 0) for c in CLAUSES)
        P(f"{D if j == 0 else ''} & {LABEL[rep]} & {acc} & {lit} \\\\\n")
    if i < 2:
        P("\\midrule\n")
P(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ 200 labels
N = defaultdict(dict)
for r in rows(RESULTS / "classify_sources_N200.csv"):
    N[(r["dataset"], r["representation"])][int(r["epoch"])] = float(r["test_accuracy"])
P(r"""
\section{Learning curves with 200 labeled texts}
\label{app:learning-curves}

Figure~\ref{fig:learning-curves} and Table~\ref{tab:learning-curves} show the test accuracy of
the TM with each input after every epoch, trained on $200$ labeled texts with $2000$ clauses and
$T = 1600$ for $100$ epochs. Each curve is a single run on the same $200$ texts.

\begin{figure}[H]
\centering
% Generated by experiments/HWF-TM/plot_curves.py.
\includegraphics[width=\linewidth]{Figures/learning_curves_N200.pdf}
\caption{Test accuracy per epoch with $200$ labeled training texts and $2000$ clauses. Dashed lines
are the random controls of the families of the same color. Each panel has its own y-axis.}
\label{fig:learning-curves}
\end{figure}

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Test accuracy (\%) per epoch with $200$ labeled training texts and $2000$ clauses, at
selected epochs. The best input per row is in bold. \emph{Control}: the random control of the
family source to its left.}
\label{tab:learning-curves}
\begin{tabular}{llccccccc}
\toprule
 & & & \multicolumn{2}{c}{PPMI} & \multicolumn{4}{c}{HWF-TM} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-9}
Dataset & Epoch & TM & PPMI-TM & control & profile & control & same-ds. & control \\
\midrule
""")
reps200 = [rep for rep, _ in INPUTS]
for i, (d, D) in enumerate(DATASETS):
    for j, e in enumerate([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        vals = [N[(d, rep)][e] for rep in reps200]
        best = max(vals)
        cells = " & ".join(f"$\\mathbf{{{v:.1f}}}$" if abs(v - best) < 0.05 else m(v) for v in vals)
        P(f"{D if j == 0 else ''} & ${e}$ & {cells} \\\\\n")
    if i < 2:
        P("\\midrule\n")
P(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ example clauses
LINE = re.compile(r"^(\w+)\s+w=\s*(\d+)\s+(.*)$")


def parse(path):
    out = []
    for line in path.read_text().splitlines():
        mm = LINE.match(line)
        if not mm:
            continue
        cls, w, expr = mm.group(1), int(mm.group(2)), mm.group(3)
        lits = []
        for term in expr.split(" AND "):
            neg = term.startswith("NOT ")
            term = term[4:] if neg else term
            fam = term[1:-1].split("|") if term.startswith("{") else [term]
            lits.append((neg, fam))
        out.append((cls, w, lits))
    return out


def render(lits, keep=10):
    parts = []
    for neg, fam in lits[:keep]:
        t = (
            words(fam)
            if len(fam) == 1
            else "(" + " $\\lor$ ".join(f"\\textit{{{w}}}" for w in fam) + ")"
        )
        parts.append(("$\\lnot$" if neg else "") + t)
    s = " $\\land$ ".join(parts)
    if len(lits) > keep:
        s += f" $\\land \\dots$ (${len(lits) - keep}$ more negated literals)"
    return s


P(r"""
\section{Example clauses}
\label{app:clauses}

The following clauses were learned on IMDB with $1000$ labeled texts, $250$ clauses and the first
label draw. For each input and each class, the clause with the largest weight among those that
contain at least one positive literal is shown. Families appear in parentheses, joined by $\lor$;
only the first ten literals are written out. The clauses illustrate Section~\ref{sec:results-clauses}:
a clause is one or two positive literals followed by a long list of negated ones.
""")
for rep, label in (("bow", "TM"), ("ppmi", "PPMI-TM"), ("profile", "HWF-TM (profile)")):
    clauses = parse(RESULTS / "clauses_sources_N1000" / f"imdb_N1000_s1_{rep}_C250.txt")
    P(f"\n\\subsection{{{label}}}\n\\begin{{itemize}}\n")
    for cls in ("positive", "negative"):
        cands = [c for c in clauses if c[0] == cls and any(not neg for neg, _ in c[2])]
        if not cands:
            continue
        _, w, lits = max(cands, key=lambda c: c[1])
        npos = sum(1 for neg, _ in lits if not neg)
        P(
            f"  \\item Votes for \\emph{{{cls}}}, weight ${w}$, ${len(lits)}$ literals (${npos}$ positive): {render(lits)}\n"
        )
    P("\\end{itemize}\n")

OUT.write_text("".join(out))
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
