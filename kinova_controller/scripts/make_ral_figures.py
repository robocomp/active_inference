#!/usr/bin/env python3
"""RA-L figures + results table from the experiment CSVs. Robust to missing cells."""
import glob, os, csv, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "experiments")
OUT  = os.path.join(ROOT, "figs"); os.makedirs(OUT, exist_ok=True)

def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# ---------- 1. Noise A/B: success + episode_s vs sigma, prec vs succ, mean±std over seeds ----------
def noise_ab():
    sig = {"00": 0.0, "02": 0.02, "03": 0.03}
    cells = {}  # (law,sig) -> list of (succ_frac, mean_episode_s) per seed
    for f in glob.glob(os.path.join(ROOT, "stats", "*.csv")):
        b = os.path.basename(f)[:-4]                      # e.g. prec_n02_s1
        parts = b.split("_")
        if len(parts) != 3: continue
        law, ntag, _ = parts
        rows = read(f)
        if not rows: continue
        succ = [int(r["success"]) for r in rows]
        sf = sum(succ)/len(succ)
        es = [float(r["episode_s"]) for r in rows if r["success"] == "1"]
        me = st.mean(es) if es else float("nan")
        cells.setdefault((law, ntag[1:]), []).append((sf, me))
    if not cells:
        print("[noise_ab] no stats cells yet"); return
    print("\n=== NOISE A/B (mean±std over seeds) ===")
    print(f"{'cell':<14}{'success':<16}{'episode_s':<16}{'nseeds'}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    xs = sorted(sig.values())
    for law, color in (("prec", "tab:blue"), ("succ", "tab:red")):
        sr_m, sr_s, et_m, et_s = [], [], [], []
        for s in ("00", "02", "03"):
            v = cells.get((law, s), [])
            sf = [x[0] for x in v]; ee = [x[1] for x in v if x[1] == x[1]]
            sr_m.append(st.mean(sf) if sf else float("nan"))
            sr_s.append(st.pstdev(sf) if len(sf) > 1 else 0.0)
            et_m.append(st.mean(ee) if ee else float("nan"))
            et_s.append(st.pstdev(ee) if len(ee) > 1 else 0.0)
            print(f"{law}_n{s:<10}{st.mean(sf) if sf else float('nan'):.2f}±{(st.pstdev(sf) if len(sf)>1 else 0):.2f}      "
                  f"{(st.mean(ee) if ee else float('nan')):.2f}±{(st.pstdev(ee) if len(ee)>1 else 0):.2f}        {len(sf)}")
        lab = "precision" if law == "prec" else "success-rate"
        a1.errorbar(xs, sr_m, yerr=sr_s, marker="o", color=color, label=lab, capsize=3)
        a2.errorbar(xs, et_m, yerr=et_s, marker="o", color=color, label=lab, capsize=3)
    a1.set_xlabel(r"$\sigma_{obs}$ (m)"); a1.set_ylabel("success rate"); a1.set_ylim(-0.05, 1.05); a1.legend(); a1.grid(alpha=.3)
    a2.set_xlabel(r"$\sigma_{obs}$ (m)"); a2.set_ylabel("episode time (s, successes)"); a2.legend(); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_noise_ab.pdf")); print("-> figs/fig_noise_ab.pdf")

# ---------- 2. Latency: effective rate vs skill, per latency ----------
def latency():
    fs = sorted(glob.glob(os.path.join(ROOT, "latency_*ms.csv")))
    if not fs: print("[latency] no latency_*ms.csv"); return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for f in fs:
        L = os.path.basename(f).replace("latency_", "").replace("ms.csv", "")
        rows = read(f)
        try:
            x = [float(r["c_approach"]) for r in rows]; y = [float(r["eff_rate_hz"]) for r in rows]
        except Exception: continue
        order = sorted(range(len(x)), key=lambda i: x[i])
        ax.plot([x[i] for i in order], [y[i] for i in order], marker="o", label=f"{L} ms")
    ax.set_xlabel(r"approach skill $c$"); ax.set_ylabel("effective rate (Hz)"); ax.legend(title="look-up latency"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_latency.pdf")); print("-> figs/fig_latency.pdf")

# ---------- 3. Localization: per-rep c_k under noise (approach pinned vs others) ----------
def localization():
    cand = glob.glob(os.path.join(ROOT, "stats", "prec_n02_s*.csv"))
    if not cand: print("[localization] no prec_n02 cell"); return
    rows = read(sorted(cand)[0])
    segs = ["c_approach", "c_insert", "c_lift", "c_place", "c_retreat"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    reps = [int(r["rep"]) for r in rows]
    for s in segs:
        ax.plot(reps, [float(r[s]) for r in rows], marker=".", label=s.replace("c_", ""))
    ax.set_xlabel("episode"); ax.set_ylabel(r"segment confidence $c_k$"); ax.set_title(r"precision, $\sigma_{obs}{=}0.02$")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_localization.pdf")); print("-> figs/fig_localization.pdf")

if __name__ == "__main__":
    noise_ab(); latency(); localization()
