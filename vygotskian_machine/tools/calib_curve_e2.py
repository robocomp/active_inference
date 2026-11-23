#!/usr/bin/env python3
"""E2 learning curve: how many person-confirmed rooms the VLM priors need to reach the hand-written table.

The robot does not have 45k labelled rooms; it has the rooms a person has confirmed. We simulate that on ScanNet:
n rooms are drawn from one half (by scene hash, as in function_evidence_e2.py) as "confirmed", and every source is
scored on the other half. Sources fitted from the n confirmed rooms:
    level_map    the VLM's answer levels mapped to frequencies counted in the n rooms, pooled over the (class, function)
                 pairs given each level, monotone by isotonic regression (one shared parameter per level)
    eq5          Eq. (lh_beta_prior) per (class, function) cell, VLM as pseudo-counts (α=1, κ=4, β=0.5) + the n rooms
    cooc         frequencies of the n rooms alone, Laplace-smoothed (no VLM)
Fixed references on the same test half: VLM as elicited, hand-written table, VLM calibrated on ProcTHOR.
For each n, R random draws (seeded); mean and 2.5/97.5 percentiles of macro F1 and top-1. Both halves serve as the
test half in turn. Writes tools/e2_out/calib_curve_scannet.csv.

    python3 tools/calib_curve_e2.py [--reps 20]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from function_evidence_e2 import (ALPHA, BETA, KAPPA, derived_notf, fit_levels, fold_of, isotonic,  # noqa: E402
                                  lam, scores)

NS = [0, 5, 10, 20, 50, 100, 200]


def counts(rooms, Z, idx, classes):
    n_f, k_cf = defaultdict(int), defaultdict(int)
    for i in idx:
        f = rooms[i]["function"]
        n_f[f] += 1
        for c in classes:
            k_cf[(c, f)] += Z[i][c]
    return n_f, k_cf


def evaluate(pi, pin, Z, truth, funcs, classes, pf, test):
    pred, top = {}, 0
    for i in test:
        ls = {f: lam(Z[i], classes, pf[f], pi, pin, f) for f in funcs}
        pred[i] = {f for f, v in ls.items() if v > 0}
        top += max(ls, key=ls.get) == truth[i]
    return scores(pred, truth, funcs, test)[0], top / len(test)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rooms", type=Path, default=HERE / "corpora/scannet_rooms.csv")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    rooms = list(csv.DictReader(args.rooms.open()))
    classes = list(rooms[0])[4:]
    funcs = sorted({r["function"] for r in rooms})
    truth = [r["function"] for r in rooms]
    Z = [{c: int(r[c]) for c in classes} for r in rooms]
    pf = {r["func"]: float(r["p_f"]) for r in csv.DictReader((HERE / "e1_out/function_prior.csv").open())}
    vrows = list(csv.DictReader((HERE / "e1_out/level4_priors.csv").open()))
    all_funcs = sorted({r["func"] for r in vrows})
    vlm = {(r["cls"], r["func"]): float(r["pi_cf"]) for r in vrows}
    vlm_n = {(r["cls"], r["func"]): float(r["pi_c_notf"]) for r in vrows}
    t = tomllib.loads((HERE / "handwritten_function_table.toml").read_text())
    hw = {(c, f): (t["signature"] if c in t["functions"].get(f, []) else t["other"]) for c in classes for f in all_funcs}
    cal, _ = fit_levels(vlm, HERE / "corpora/procthor_rooms.csv")
    vc = {k: cal[v] for k, v in vlm.items()}
    fixed = {"vlm": (vlm, vlm_n), "handwritten": (hw, derived_notf(hw, classes, all_funcs)),
             "vlm_cal_procthor": (vc, derived_notf(vc, classes, all_funcs))}

    folds = [fold_of(r["house"]) for r in rooms]
    rows = []
    for test_fold in (0, 1):
        test = [i for i in range(len(rooms)) if folds[i] == test_fold]
        pool = [i for i in range(len(rooms)) if folds[i] != test_fold]
        for name, (pi, pin) in fixed.items():
            rows.append((test_fold, name, -1, 0, *evaluate(pi, pin, Z, truth, funcs, classes, pf, test)))
        for n in NS:
            if n > len(pool):
                continue
            for rep in range(args.reps if n else 1):
                conf = random.Random(1000 * n + rep + 7 * test_fold).sample(pool, n)
                n_f, k_cf = counts(rooms, Z, conf, classes)
                # level map: pooled over cells with the same elicited level, only functions seen in the n rooms
                raw = defaultdict(lambda: [0, 0])
                for (c, f), lv in vlm.items():
                    if c in classes and n_f[f]:
                        raw[lv][0] += k_cf[(c, f)]
                        raw[lv][1] += n_f[f]
                xs = sorted(set(vlm.values()))
                lm = isotonic(xs, [raw[x][0] for x in xs], [raw[x][1] for x in xs])
                pl = {k: lm[v] for k, v in vlm.items()}
                e5 = {k: (ALPHA + KAPPA * BETA * vlm[k] + k_cf[k]) / (2 * ALPHA + KAPPA * BETA + n_f[k[1]])
                      for k in vlm}
                co = {(c, f): (k_cf[(c, f)] + 1) / (n_f[f] + 2) for c in classes for f in all_funcs}
                for name, pi in (("level_map", pl), ("eq5", e5), ("cooc", co)):
                    rows.append((test_fold, name, n, rep,
                                 *evaluate(pi, derived_notf(pi, classes, all_funcs), Z, truth, funcs, classes, pf,
                                           test)))

    out = HERE / "e2_out/calib_curve_scannet.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_fold", "source", "n_confirmed", "rep", "macro_f1", "top1"])
        w.writerows(rows)

    def summ(vals):
        v = sorted(vals)
        return sum(v) / len(v), v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]

    print("macro F1 / top-1 on the held-out half (mean over both folds and draws; [2.5, 97.5] %)")
    for name in fixed:
        f1 = summ([r[4] for r in rows if r[1] == name])
        t1 = summ([r[5] for r in rows if r[1] == name])
        print(f"  {name:<18} F1 {f1[0]:.3f}   top-1 {t1[0]:.3f}")
    for name in ("level_map", "eq5", "cooc"):
        print(f"  {name}")
        for n in NS:
            sel = [r for r in rows if r[1] == name and r[2] == n]
            if not sel:
                continue
            f1, t1 = summ([r[4] for r in sel]), summ([r[5] for r in sel])
            print(f"    n={n:<4} F1 {f1[0]:.3f} [{f1[1]:.3f}, {f1[2]:.3f}]   top-1 {t1[0]:.3f} [{t1[1]:.3f}, {t1[2]:.3f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
