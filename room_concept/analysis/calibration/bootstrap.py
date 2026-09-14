"""Coverage of the declared sigma, with confidence intervals bootstrapped BY ROOM.

The 4344 matched corners are not independent: four corners of one room share its walls, its
trajectory and its annotation. Resampling corners would therefore report an interval far too tight.
Resampling ROOMS with replacement respects the correlation that actually exists.
"""
import sys, csv, numpy as np
OUT = sys.argv[1]
BAR = 0.06     # the publication gate, metres

rows = list(csv.DictReader(open(f"{OUT}/corners.csv")))
room = np.array([int(r["room"]) for r in rows])
sig  = np.array([float(r["sigma"]) for r in rows])
err  = np.array([float(r["error"]) for r in rows])
rooms = np.unique(room)
print(f"{len(rows)} corners, {len(rooms)} rooms")

def stats(mask_r):
    idx = np.concatenate([np.where(room == r)[0] for r in mask_r])
    s, e = sig[idx], err[idx]
    pub = s <= BAR
    return dict(
        c1=float(np.mean(e <= s)), c2=float(np.mean(e <= 2*s)), c3=float(np.mean(e <= 3*s)),
        pub_rate=float(np.mean(pub)),
        pub_holds=float(np.mean(e[pub] <= BAR)) if pub.any() else float("nan"),
        pub_med=float(np.median(e[pub])) if pub.any() else float("nan"),
        wit_med=float(np.median(e[~pub])) if (~pub).any() else float("nan"),
        wit_holds=float(np.mean(e[~pub] <= BAR)) if (~pub).any() else float("nan"),
        spearman=float(np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(e)))[0, 1]),
    )

point = stats(rooms)
rng = np.random.default_rng(4242)
B = 2000
boot = {k: [] for k in point}
for _ in range(B):
    draw = rng.choice(rooms, size=len(rooms), replace=True)
    st = stats(draw)
    for k, v in st.items(): boot[k].append(v)

print("\n" + "quantity".ljust(22) + "point".rjust(9) + ("  95% CI by room, B=" + str(B)).rjust(28))
for k in ["c1", "c2", "c3", "pub_rate", "pub_holds", "pub_med", "wit_med", "wit_holds", "spearman"]:
    a = np.array(boot[k]); a = a[np.isfinite(a)]
    lo, hi = np.percentile(a, [2.5, 97.5])
    unit = " m" if k.endswith("med") else ""
    print(f"{k:<22}{point[k]:>9.4f}{unit}   [{lo:.4f}, {hi:.4f}]")

# ── BASELINE: one global scalar precision, calibrated on this very corpus ───────────────────────
# Chosen so it achieves the SAME 1-sigma coverage as ours — i.e. calibrated as favourably as the
# corpus allows — and then asked to do the job the per-corner sigma does: rank corners by how wrong
# they are, and decide which to publish.
q = float(np.quantile(err, point["c1"]))
print(f"\nglobal constant sigma calibrated to match our 1-sigma coverage: {q:.4f} m")
print(f"  its Spearman rho against realised error: 0.0000 (a constant cannot rank)")
print(f"  publishable fraction under the {BAR} m gate: "
      f"{'100%' if q <= BAR else '0%'} (all corners share one sigma, so the gate cannot select)")
