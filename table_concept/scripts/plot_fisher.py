#!/usr/bin/env python3
"""
plot_fisher.py — visualise the evolution of the table_concept Fisher information filter.

Reads the per-cycle CSV written by TableFitter::log_fisher_csv (WarmStart.FisherCsvPath,
default etc/fisher_filter_log.csv) and plots, per table node, how the belief hardens as the
robot orbits the table and gathers evidence:

  1. State (w,h) settling                — the dimensions stop reshaping
  2. Accumulated normalised info (acc_*) — the stiffness driver, vs the InfoHalf knee
  3. Posterior std (std_*, log mm)       — data-only uncertainty shrinking with evidence
  4. Per-frame observation Fisher (obs_*) on fresh masks — the measurements being folded in

Usage:
    python3 scripts/plot_fisher.py [csv_path] [--node table_1] [--info-half 20] [--save out.png]
"""
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

DOFS = ["cx", "cy", "w", "h", "H", "leg", "yaw", "inset"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="etc/fisher_filter_log.csv")
    ap.add_argument("--node", default=None, help="table node to plot (default: all nodes)")
    ap.add_argument("--info-half", type=float, default=20.0,
                    help="WarmStart.InfoHalf (draws the gain-halving knee on panel 2)")
    ap.add_argument("--save", default=None, help="write the figure to this path instead of showing")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print(f"{args.csv}: no rows", file=sys.stderr)
        return 1

    nodes = [args.node] if args.node else sorted(df["node"].unique())

    for node in nodes:
        d = df[df["node"] == node].sort_values("cycle")
        if d.empty:
            print(f"no rows for node '{node}'", file=sys.stderr)
            continue
        fresh = d[d["fresh"] == 1]

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        fig.suptitle(f"Fisher information filter — {node}", fontsize=13)

        # 1) State (the plastic dimensions) settling
        a = ax[0, 0]
        a.plot(d["cycle"], d["state_w"], label="w", color="C0")
        a.plot(d["cycle"], d["state_h"], label="h", color="C1")
        a.set_ylabel("state (m)")
        a.set_title("1. Dimensions settling")
        a.legend(loc="best"); a.grid(alpha=0.3)

        # 2) The stiffness controller: per-DOF Kalman gain if present (calibrated), else the
        #    legacy normalised accumulator vs the InfoHalf knee.
        a = ax[0, 1]
        has_gain = "gain_w" in d.columns and (d["fresh"] == 1).any()
        if has_gain:
            g = d[d["fresh"] == 1]
            for j, dof in enumerate(DOFS):
                faint = dof not in ("w", "h")
                a.plot(g["cycle"], g[f"gain_{dof}"], (".-" if not faint else "."),
                       ms=3, label=dof, color="0.75" if faint else f"C{j}",
                       lw=1.5 if not faint else 0, zorder=3 if not faint else 1)
            a.set_ylabel("acceptance gain  K = obs/(Y_pred+obs)")
            a.set_title("2. Calibrated stiffness — per-DOF Kalman gain (fresh frames)")
            a.set_ylim(-0.05, 1.05)
        else:
            for j, dof in enumerate(DOFS):
                faint = dof not in ("w", "h")
                a.plot(d["cycle"], d[f"acc_{dof}"], label=dof,
                       color="0.75" if faint else f"C{j}",
                       lw=1 if faint else 2, zorder=1 if faint else 3)
            a.axhline(args.info_half, color="k", ls="--", lw=1, label=f"InfoHalf={args.info_half:g}")
            a.set_ylabel("accumulated info  (equiv. views)")
            a.set_title("2. Accumulated info → stiffness  (stiff = IH/(IH+acc))")
        a.legend(loc="best", ncol=2, fontsize=8); a.grid(alpha=0.3)

        # 3) Posterior std (data-only), log scale; -1 = unobserved → masked
        a = ax[1, 0]
        for dof, col in (("w", "C0"), ("h", "C1"), ("cx", "C2"), ("cy", "C3"), ("yaw", "C4")):
            s = d[f"std_{dof}"].where(d[f"std_{dof}"] > 0)
            unit = "mrad" if dof == "yaw" else "mm"
            a.plot(d["cycle"], s, label=f"{dof} ({unit})", color=col)
        a.set_yscale("log")
        a.set_ylabel("posterior std  (mm / mrad, log)")
        a.set_xlabel("cycle")
        a.set_title("3. Posterior uncertainty shrinking")
        a.legend(loc="best", fontsize=8); a.grid(alpha=0.3, which="both")

        # 4) Per-frame observation Fisher on fresh masks (the measurements)
        a = ax[1, 1]
        a.plot(fresh["cycle"], fresh["obs_w"], ".", ms=4, label="obs_w", color="C0")
        a.plot(fresh["cycle"], fresh["obs_h"], ".", ms=4, label="obs_h", color="C1")
        a.set_yscale("log")
        a.set_ylabel("per-frame Fisher info (fresh masks)")
        a.set_xlabel("cycle")
        a.set_title(f"4. Measurements folded in  ({len(fresh)} fresh / {len(d)} cycles)")
        a.legend(loc="best", fontsize=8); a.grid(alpha=0.3, which="both")

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if args.save:
            out = args.save if len(nodes) == 1 else args.save.replace(".png", f"_{node}.png")
            fig.savefig(out, dpi=120)
            print(f"wrote {out}")

    if not args.save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
