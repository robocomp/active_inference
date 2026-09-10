#!/usr/bin/env python3
"""WAF figures from the experiment CSVs.

perseg(): per-segment confidence over the three cold-start rounds. insert and
retreat are clean binary successes, so their c_k curves coincide exactly; we draw
retreat dashed over a solid insert so the orange insert line stays visible.

noise_ab(): precision vs. success-rate schedule under perception noise. Same
knobs and per-segment partition; only the update rule differs (multiplicative
precision deflation vs. a Beta success-frequency). Three seeds, ten episodes each.
"""
import os, csv, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "experiments")
OUT  = "/home/pbustos/drive/Papers/Papers Ongoing/2026/WAF-Kinova/figures"

ROUNDS = [os.path.join(ROOT, f"perseg_round{i}.csv") for i in (1, 2, 3)]
SEGS = ["c_approach", "c_insert", "c_lift", "c_place", "c_retreat"]
# explicit, distinct colours; retreat dashed so it does not hide insert
STYLE = {
    "c_approach": dict(color="tab:blue",   ls="-",  lw=1.8),
    "c_insert":   dict(color="tab:orange", ls="-",  lw=2.2),
    "c_lift":     dict(color="tab:green",  ls="-",  lw=1.8),
    "c_place":    dict(color="tab:red",    ls="-",  lw=1.8),
    "c_retreat":  dict(color="tab:purple", ls="--", lw=1.8),
}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def perseg():
    rounds = [read(p) for p in ROUNDS]
    n = min(len(r) for r in rounds)
    episodes = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for s in SEGS:
        cols = [[float(r[i][s]) for r in rounds] for i in range(n)]   # per-episode across rounds
        mean = [sum(c) / len(c) for c in cols]
        lo = [min(c) for c in cols]
        hi = [max(c) for c in cols]
        s_ = STYLE[s]
        ax.fill_between(episodes, lo, hi, color=s_["color"], alpha=0.15, linewidth=0)
        ax.plot(episodes, mean, label=s.replace("c_", ""), **s_)

    ax.set_xlabel("episode")
    ax.set_ylabel(r"segment confidence $c_k$")
    ax.set_xlim(1, n)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "fig_perseg_c.pdf")
    fig.savefig(dst)
    print("->", dst)


def noise_ab():
    sig = {"n00": 0.0, "n02": 0.02, "n03": 0.03}
    order = sorted(sig, key=lambda k: sig[k])
    laws = [("prec", "tab:blue", "o", "precision"),
            ("succ", "tab:red", "s", "success-rate")]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for law, color, mk, lab in laws:
        sx, sr_m, sr_e = [], [], []          # success rate vs sigma
        ex, et_m, et_e = [], [], []          # episode time (successes only) vs sigma
        for nt in order:
            srs, ets = [], []
            for f in sorted(glob.glob(os.path.join(ROOT, "stats", f"{law}_{nt}_s*.csv"))):
                r = read(f)
                succ = [int(x["success"]) for x in r]
                srs.append(sum(succ) / len(succ))
                es = [float(x["episode_s"]) for x in r if x["success"] == "1"]
                if es:
                    ets.append(st.mean(es))
            if srs:
                sx.append(sig[nt]); sr_m.append(st.mean(srs)); sr_e.append(st.pstdev(srs))
            if ets:
                ex.append(sig[nt]); et_m.append(st.mean(ets)); et_e.append(st.pstdev(ets))
        a1.errorbar(sx, sr_m, yerr=sr_e, marker=mk, color=color, label=lab, capsize=3)
        a2.errorbar(ex, et_m, yerr=et_e, marker=mk, color=color, label=lab, capsize=3)

    a1.set_xlabel(r"perception noise $\sigma$ (m)")
    a1.set_ylabel("success rate")
    a1.set_ylim(-0.05, 1.08)
    a1.set_xticks(sorted(sig.values()))
    a1.legend(fontsize=8, loc="lower left")
    a2.set_xlabel(r"perception noise $\sigma$ (m)")
    a2.set_ylabel("episode time (s), successes only")
    a2.set_xticks(sorted(sig.values()))
    a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "fig_noise_ab.pdf")
    fig.savefig(dst)
    print("->", dst)


if __name__ == "__main__":
    perseg()
    noise_ab()
