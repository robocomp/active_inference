#!/usr/bin/env python3
"""Replay REAL selections from epistemic_select.jsonl and answer: why did that cell win?

Every scoring hypothesis of 2026-08-19 died because it was seeded with guesses about the planner's
state. Asserting that travel cost dominated the argmax was refuted by simulation over both a sparse
field and a 348-cell dense grid. The RULE was never the unknown — the per-cell ages and suppressor
values were. This reads what the planner actually held.

  score = fim · staleness^w_ior + w_drive · neglect − w_travel · (d/diag)      [× attempt_supp]

usage: replay_selection.py [epistemic_select.jsonl]
"""
import sys, json, math
from collections import Counter

def main(path):
    recs = []
    bad = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    print(f"[load] {len(recs)} selections, {bad} unparseable")
    if not recs:
        sys.exit("nothing to replay")

    w = recs[-1]
    print(f"[weights] w_travel={w['w_travel']} w_drive={w['w_drive']} w_ior={w['w_ior']} "
          f"tau={w['tau']} min_distance={w['min_distance']}")

    # 1. WHAT IS THE WINNER LIKE? The live median offer was 1.03 m, 3 cm above the MinDistance floor.
    d_win, fim_win, age_win, supp_win = [], [], [], []
    for r in recs:
        if not r["cands"]:
            continue
        c = r["cands"][0]                      # already sorted descending by score
        d_win.append(c["d"]); fim_win.append(c["fim"])
        age_win.append(c["age_s"]); supp_win.append(c["attempt_supp"])
    def q(v, p):
        v = sorted(v); return v[min(len(v) - 1, int(len(v) * p))]
    print(f"\n[winner] d      p10={q(d_win,.1):.2f} p50={q(d_win,.5):.2f} p90={q(d_win,.9):.2f} m")
    print(f"[winner] fim    p10={q(fim_win,.1):.3f} p50={q(fim_win,.5):.3f} p90={q(fim_win,.9):.3f}")
    print(f"[winner] age    p10={q(age_win,.1):.0f} p50={q(age_win,.5):.0f} p90={q(age_win,.9):.0f} s")
    print(f"[winner] supp   p10={q(supp_win,.1):.3f} p50={q(supp_win,.5):.3f} p90={q(supp_win,.9):.3f}")
    print(f"[winner] at the MinDistance floor (<{w['min_distance']+0.1:.1f} m): "
          f"{100*sum(1 for d in d_win if d < w['min_distance']+0.1)/len(d_win):.0f}%")

    # 2. WHICH TERM DECIDES? Compare the winner against the best rival, term by term. Whichever term
    #    accounts for the score gap is the one steering the robot — measured, not assumed.
    wins = Counter()
    gaps = {"fim": [], "neg": [], "travel": [], "supp": []}
    diag = 10.0
    for r in recs:
        if len(r["cands"]) < 2:
            continue
        a, b = r["cands"][0], r["cands"][1]
        fa = a["fim"] * (a["stale"] ** w["w_ior"]);  fb = b["fim"] * (b["stale"] ** w["w_ior"])
        na = w["w_drive"] * a["neg"];                nb = w["w_drive"] * b["neg"]
        ta = w["w_travel"] * (a["d"] / diag);        tb = w["w_travel"] * (b["d"] / diag)
        d_fim, d_neg, d_tr = fa - fb, na - nb, -(ta - tb)
        d_supp = (a["attempt_supp"] - b["attempt_supp"]) * (fa + na)
        terms = {"fim": d_fim, "neg": d_neg, "travel": d_tr, "supp": d_supp}
        for k, v in terms.items():
            gaps[k].append(v)
        wins[max(terms, key=lambda k: terms[k])] += 1
    if wins:
        tot = sum(wins.values())
        print(f"\n[decider] which term most favours the winner over the runner-up, over {tot} selections:")
        for k, n in wins.most_common():
            mean = sum(gaps[k]) / len(gaps[k])
            print(f"    {k:8s} decides {100*n/tot:5.1f}%   mean advantage {mean:+.4f} nats")
        print("\n[spread] a term can only steer if its SPREAD exceeds the others'.")
        for k in ("fim", "neg", "travel", "supp"):
            v = sorted(abs(x) for x in gaps[k])
            print(f"    {k:8s} |Δ| p50={v[len(v)//2]:.4f}  p90={v[int(len(v)*.9)]:.4f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "epistemic_select.jsonl")
