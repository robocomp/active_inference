#!/usr/bin/env python3
"""E2 (paper §5.3): the function evidence Λ_f of Eq. (lh_function_evidence) on a labelled room corpus.

Corpus conditions (as the paper states): every annotated object is a certain detection (p_c = z_c ∈ {0, 1}) and
every class not annotated was searched (d_c = 1). Units are the corpus classes (columns of the rooms CSV); the corpus
has no level-3 aggregates, so the members of a kitchen run count separately, which makes Λ_f more overconfident
than on the robot. With p_c ∈ {0, 1} and d_c = 1:
    Λ_f(r) = logit P(f) + Σ_c [ z_c log(π_cf / π_c¬f) + (1 − z_c) log((1 − π_cf) / (1 − π_c¬f)) ]
A function is claimed when Λ_f > 0. All sources share the elicited P(f) (tools/e1_out/function_prior.csv), so they
differ only in π. Sources of π_cf, π_c¬f:
    vlm          elicited (E1); π_c¬f = mean of π_cg over the other functions
    vlm_cal      the VLM's five answer levels mapped once to frequencies measured on the calibration corpus
                 (ProcTHOR-10k by default; pooled over every (class, function) pair given that level, made monotone by
                 isotonic regression); π_c¬f derived from the calibrated π as for the VLM. Held out when --rooms is
                 another corpus; in-sample (flagged) when it is the calibration corpus itself.
    vlm_beta     Eq. (lh_beta_prior) with no counts: (α + κβ π)/(2α + κβ), α=1, κ=4, β=0.5 (no person-confirmed rooms yet)
    uniform      0.5 everywhere (Λ_f = prior: never positive)
    handwritten  tools/handwritten_function_table.toml; π_c¬f as for the VLM
    cooc         frequencies on the other half of the corpus (2-fold by scene), Laplace-smoothed; π_c¬f measured too
Only functions present in the corpus are scored. Metrics per function: precision and recall of Λ_f > 0; macro F1
and top-1 accuracy (argmax Λ_f); 95% paired bootstrap over rooms of the macro-F1 difference vlm − other source.

    python3 tools/function_evidence_e2.py [--rooms tools/corpora/scannet_rooms.csv]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALPHA, KAPPA, BETA = 1.0, 4.0, 0.5


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def fold_of(house: str) -> int:
    return int(hashlib.sha256(house.encode()).hexdigest(), 16) % 2


def lam(z: dict[str, int], classes: list[str], pf: float, pi: dict, pin: dict, f: str) -> float:
    s = logit(pf)
    for c in classes:
        a, b = pi[(c, f)], pin[(c, f)]
        s += math.log(a / b) if z[c] else math.log((1.0 - a) / (1.0 - b))
    return s


def derived_notf(pi: dict, classes, all_funcs) -> dict:
    return {(c, f): sum(pi[(c, g)] for g in all_funcs if g != f) / (len(all_funcs) - 1)
            for c in classes for f in all_funcs}


def isotonic(xs: list[float], ks: list[int], ns: list[int]) -> dict[float, float]:
    """Weighted pool-adjacent-violators on k/n over sorted levels xs; returns level -> (k+1)/(n+2) of its block."""
    blocks = [[k, n, [x]] for x, k, n in zip(xs, ks, ns)]
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        if (a[0] + 1) / (a[1] + 2) > (b[0] + 1) / (b[1] + 2):
            blocks[i:i + 2] = [[a[0] + b[0], a[1] + b[1], a[2] + b[2]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return {x: (k + 1) / (n + 2) for k, n, lv in blocks for x in lv}


def fit_levels(vlm: dict, calib_csv: Path) -> tuple[dict[float, float], dict[float, tuple[int, int]]]:
    """Map each elicited level to the frequency, on the calibration corpus, of a class being present in a room of
    function f, pooled over the (c, f) pairs the VLM gave that level."""
    rows = list(csv.DictReader(calib_csv.open()))
    classes = [c for c in list(rows[0])[4:]]
    n_f, k_cf = defaultdict(int), defaultdict(int)
    for r in rows:
        n_f[r["function"]] += 1
        for c in classes:
            k_cf[(c, r["function"])] += int(r[c])
    raw = defaultdict(lambda: [0, 0])
    for (c, f), lv in vlm.items():
        if c in classes and f in n_f:
            raw[lv][0] += k_cf[(c, f)]
            raw[lv][1] += n_f[f]
    xs = sorted(raw)
    return isotonic(xs, [raw[x][0] for x in xs], [raw[x][1] for x in xs]), {x: tuple(raw[x]) for x in xs}


def scores(pred: dict[str, set[str]], truth: list[str], funcs: list[str], idx: list[int]):
    """Macro F1 and per-function (precision, recall, F1) over the rooms in idx."""
    per = {}
    for f in funcs:
        tp = sum(1 for i in idx if f in pred[i] and truth[i] == f)
        fp = sum(1 for i in idx if f in pred[i] and truth[i] != f)
        fn = sum(1 for i in idx if f not in pred[i] and truth[i] == f)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[f] = (p, r, 2 * p * r / (p + r) if p + r else 0.0, tp + fp)
    return sum(v[2] for v in per.values()) / len(funcs), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rooms", type=Path, default=HERE / "corpora/scannet_rooms.csv")
    ap.add_argument("--priors", type=Path, default=HERE / "e1_out/level4_priors.csv")
    ap.add_argument("--fprior", type=Path, default=HERE / "e1_out/function_prior.csv")
    ap.add_argument("--table", type=Path, default=HERE / "handwritten_function_table.toml")
    ap.add_argument("--calib", type=Path, default=HERE / "corpora/procthor_rooms.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rooms = list(csv.DictReader(args.rooms.open()))
    classes = list(rooms[0])[4:]
    funcs = sorted({r["function"] for r in rooms})
    truth = [r["function"] for r in rooms]
    Z = [{c: int(r[c]) for c in classes} for r in rooms]
    pf = {r["func"]: float(r["p_f"]) for r in csv.DictReader(args.fprior.open())}

    vrows = list(csv.DictReader(args.priors.open()))
    all_funcs = sorted({r["func"] for r in vrows})
    vlm = {(r["cls"], r["func"]): float(r["pi_cf"]) for r in vrows}
    vlm_n = {(r["cls"], r["func"]): float(r["pi_c_notf"]) for r in vrows}
    shrink = lambda p: (ALPHA + KAPPA * BETA * p) / (2 * ALPHA + KAPPA * BETA)
    sources: dict[str, tuple[dict, dict]] = {
        "vlm": (vlm, vlm_n),
        "vlm_beta": ({k: shrink(v) for k, v in vlm.items()}, {k: shrink(v) for k, v in vlm_n.items()}),
        "uniform": ({k: 0.5 for k in vlm}, {k: 0.5 for k in vlm}),
    }
    cal, raw = fit_levels(vlm, args.calib)
    in_sample = args.calib.resolve() == args.rooms.resolve()
    print(f"level calibration on {args.calib.name}{' (IN-SAMPLE)' if in_sample else ''}: " +
          ", ".join(f"{x:.2f}→{cal[x]:.3f} ({k}/{n})" for x, (k, n) in raw.items()))
    vc = {k: cal[v] for k, v in vlm.items()}
    sources["vlm_cal"] = (vc, derived_notf(vc, {c for c, _ in vlm}, all_funcs))
    t = tomllib.loads(args.table.read_text())
    hw = {(c, f): (t["signature"] if c in t["functions"].get(f, []) else t["other"]) for c in classes for f in all_funcs}
    sources["handwritten"] = (hw, derived_notf(hw, classes, all_funcs))

    # Co-occurrence: per fold, frequencies from the OTHER fold.
    folds = [fold_of(r["house"]) for r in rooms]
    cooc = {}
    for k in (0, 1):
        n_f, k_cf = defaultdict(int), defaultdict(int)
        for r, z, fo in zip(rooms, Z, folds):
            if fo == k:
                continue
            n_f[r["function"]] += 1
            for c in classes:
                k_cf[(c, r["function"])] += z[c]
        n_all = sum(n_f.values())
        k_c = {c: sum(k_cf[(c, f)] for f in funcs) for c in classes}
        pi = {(c, f): (k_cf[(c, f)] + 1) / (n_f[f] + 2) for c in classes for f in funcs}
        pin = {(c, f): (k_c[c] - k_cf[(c, f)] + 1) / (n_all - n_f[f] + 2) for c in classes for f in funcs}
        cooc[k] = (pi, pin)

    preds, top1 = {}, {}
    lambdas = {}
    for name in list(sources) + ["cooc"]:
        P, T, L = [], [], []
        for i, z in enumerate(Z):
            pi, pin = cooc[folds[i]] if name == "cooc" else sources[name]
            ls = {f: lam(z, classes, pf[f], pi, pin, f) for f in funcs}
            P.append({f for f, v in ls.items() if v > 0})
            T.append(max(ls, key=ls.get) == truth[i])
            L.append(ls)
        preds[name], top1[name], lambdas[name] = P, T, L

    idx = list(range(len(rooms)))
    print(f"corpus {args.rooms.name}: {len(rooms)} rooms, functions {funcs}, {len(classes)} class units; "
          f"P(f) = {', '.join(f'{f} {pf[f]:.2f}' for f in funcs)}")
    print(f"\n{'source':<12} {'macro F1':>8} {'top-1':>6}   per function: precision / recall (claims)")
    summary = {}
    for name in preds:
        mf1, per = scores(preds[name], truth, funcs, idx)
        acc = sum(top1[name]) / len(idx)
        summary[name] = (mf1, acc, per)
        print(f"{name:<12} {mf1:8.3f} {acc:6.3f}   " +
              "  ".join(f"{f[:8]} {p:.2f}/{r:.2f} ({n})" for f, (p, r, _, n) in per.items()))

    rng = random.Random(0)
    print("\nsource − source, macro F1, 95% paired bootstrap over rooms")
    pairs = [("vlm", o) for o in ("vlm_cal", "vlm_beta", "uniform", "handwritten", "cooc")] + \
            [("vlm_cal", o) for o in ("handwritten", "cooc")]
    for base, other in pairs:
        d, a = [], []
        for _ in range(args.boot):
            s = rng.choices(idx, k=len(idx))
            d.append(scores(preds[base], truth, funcs, s)[0] - scores(preds[other], truth, funcs, s)[0])
            a.append(sum(top1[base][i] - top1[other][i] for i in s) / len(s))
        d.sort(); a.sort()
        lo, hi = int(0.025 * len(d)), int(0.975 * len(d)) - 1
        print(f"  {base} vs {other:<12} F1 {summary[base][0] - summary[other][0]:+.3f} [{d[lo]:+.3f}, {d[hi]:+.3f}]"
              f"   top-1 {summary[base][1] - summary[other][1]:+.3f} [{a[lo]:+.3f}, {a[hi]:+.3f}]")

    out = args.out or HERE / "e2_out" / f"e2_{args.rooms.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "function", "precision", "recall", "f1", "claims", "macro_f1", "top1"])
        for name, (mf1, acc, per) in summary.items():
            for f, (p, r, f1, n) in per.items():
                w.writerow([name, f, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", n, f"{mf1:.4f}", f"{acc:.4f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
