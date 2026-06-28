#!/usr/bin/env python3
"""
plot_fisher.py — visualise the evolution of the chair_concept Fisher information filter.

Reads the per-cycle CSV written by ChairFitter::log_fisher_csv (WarmStart.FisherCsvPath,
default etc/fisher_filter_log.csv) and plots, per chair node, how the belief hardens as the
robot orbits the chair and gathers evidence:

  1. State (w,h) settling                — the dimensions stop reshaping
  2. Accumulated normalised info (acc_*) — the stiffness driver, vs the InfoHalf knee
  3. Posterior std (std_*, log mm)       — data-only uncertainty shrinking with evidence
  4. Per-frame observation Fisher (obs_*) on fresh masks — the measurements being folded in

Usage:
    python3 scripts/plot_fisher.py [csv_path] [--node chair_1] [--info-half 20] [--save out.png]
"""
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

DOFS = ["cx", "cy", "w", "h", "H", "leg", "yaw", "inset"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="etc/fisher_filter_log.csv")
    ap.add_argument("--node", default=None, help="chair node to plot (default: all nodes)")
    ap.add_argument("--info-half", type=float, default=20.0,
                    help="WarmStart.InfoHalf (draws the gain-halving knee on panel 2)")
    ap.add_argument("--gt-w", type=float, default=1.5, help="ground-truth width (m) for panel 1")
    ap.add_argument("--gt-h", type=float, default=1.4, help="ground-truth depth (m) for panel 1")
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

        # Size-bias diagnostic (printed if the extent columns are present): is the w/h over-estimate
        # the model overshooting the data, or off-chair margin points inflating the observed span?
        if "ext_hw02" in d.columns and not fresh.empty:
            late = fresh.tail(60)
            fw, fh = late["state_w"] / 2, late["state_h"] / 2          # fitted half-extents
            legfrac = late["n_leg"] / late["n_pts"].clip(lower=1)
            print(f"\n[{node}] extent diagnostic (late {len(late)} fresh frames, half-extents in m):")
            print(f"  fitted   half_w={fw.mean():.3f}  half_h={fh.mean():.3f}   (state_w={2*fw.mean():.3f} state_h={2*fh.mean():.3f})")
            for tag, lo in (("2-98", "02"), ("5-95", "05"), ("10-90", "10")):
                ex, ey = late[f"ext_hw{lo}"].mean(), late[f"ext_hh{lo}"].mean()
                print(f"  observed {tag:>5} span: half_x={ex:.3f}  half_y={ey:.3f}   "
                      f"(Δ vs fit: x={2*(fw.mean()-ex):+.3f} y={2*(fh.mean()-ey):+.3f})")
            print(f"  2-98 span local centre offset: x={late['ext_offx'].mean():+.3f} y={late['ext_offy'].mean():+.3f}")
            print(f"  point split: top={1-legfrac.mean():.1%}  leg={legfrac.mean():.1%}  "
                  f"(n≈{int(late['n_pts'].mean())}/frame)")

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        fig.suptitle(f"Fisher information filter — {node}", fontsize=13)

        # 1) State (the plastic dimensions) settling
        a = ax[0, 0]
        a.plot(d["cycle"], d["state_w"], label="w", color="C0")
        a.plot(d["cycle"], d["state_h"], label="h", color="C1")
        a.axhline(args.gt_w, color="C0", ls="--", lw=1, alpha=0.7, label=f"GT w={args.gt_w:g}")
        a.axhline(args.gt_h, color="C1", ls="--", lw=1, alpha=0.7, label=f"GT h={args.gt_h:g}")
        a.set_ylabel("state (m)")
        a.set_title("1. Dimensions settling")
        a.legend(loc="best", fontsize=8); a.grid(alpha=0.3)

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
