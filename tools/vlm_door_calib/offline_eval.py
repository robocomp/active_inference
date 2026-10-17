#!/usr/bin/env python3
"""offline_eval.py — how reliable is qwen3-vl:8b-instruct at telling a door's state, on SAVED frames.

    python3 tools/vlm_door_calib/offline_eval.py run    [--variants plain,json,json_box,json_ovl,passable] [--out results.csv]
    python3 tools/vlm_door_calib/offline_eval.py report [--out results.csv]

Data (all under data/, hand-labelled 2026-10-12):
  zeddump_labels.csv  148 crops of retina's ZedDump frames (retina/etc/zed_frames, raw ZED images, NO overlays) from the
                      door-crossing runs try1..try33. Each row carries the Webots ground-truth robot pose of the frame;
                      the crop is the projection of the doorway of DOOR_0 / DOOR_1 (door_vlm.zed_door_box) through
                      that pose, so the crop is centred on where the door IS, whether or not a wall hides it.
                      Labels (by eye, from the crop with the projected box drawn):
                        target_visible  1 = the projected doorway is visible, 0 = hidden (wall), -1 = ambiguous
                        state           open | closed | none (target hidden) | amb   — Webots doors here are either
                                        shut (0 deg) or opened by DoorControl without a hint (90 deg); no crop showed a
                                        door mid-swing clearly, so the ajar/half bins are EMPTY offline (live_calib.py
                                        fills them).
                        main_state      the most prominent door in the UN-boxed crop (another door can be in view when
                                        the target is hidden) — the truth for the un-boxed prompts' "visible" flag.
  overlay_labels.csv  7 room_concept snapshot frames WITH estimation overlays (all closed doors), hand-cropped.

Results are appended per (variant, id) so an interrupted run resumes.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import door_vlm as dv  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                  # active_inference/
FRAMES = ROOT / "retina/etc/zed_frames"


def zeddump_items():
    lab = pd.read_csv(HERE / "data/zeddump_labels.csv")
    for r in lab.itertuples():
        bgr = cv2.imread(str(FRAMES / r.file))
        if bgr is None:
            print(f"missing frame {r.file}", file=sys.stderr)
            continue
        box = dict(u0=r.u0, u1=r.u1, v0=r.v0, v1=r.v1)
        (x0, y0, _), crop = dv.square_crop(bgr, box)
        yield f"z{r.id}", crop, dv.draw_box(crop, box, (x0, y0)), r


def overlay_items():
    lab = pd.read_csv(HERE / "data/overlay_labels.csv")
    for r in lab.itertuples():
        bgr = cv2.imread(str(ROOT / r.file))
        H, W = bgr.shape[:2]
        s = int(r.side)
        x0 = int(np.clip(r.cx - s // 2, 0, W - s)); y0 = int(np.clip(r.cy - s // 2, 0, H - s))
        yield r.id, bgr[y0:y0 + s, x0:x0 + s].copy(), None, r


def run(args):
    out = Path(args.out)
    done = set()
    if out.exists():
        for row in csv.DictReader(open(out)):
            done.add((row["variant"], row["id"]))
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["set", "variant", "id", "state_pred", "visible_pred", "latency_s", "prompt_tokens", "eval_tokens",
                    "raw"])
    variants = args.variants.split(",")
    for setname, items in (("zeddump", zeddump_items), ("overlay", overlay_items)):
        for iid, crop, boxed, _ in items():
            for v in variants:
                if setname == "overlay" and v == "json_box":
                    continue                                    # no projected box for the hand-cropped set
                if (v, iid) in done:
                    continue
                img = boxed if dv.VARIANTS[v]["box"] else crop
                a = dv.ask(img, v)
                w.writerow([setname, v, iid, a["state"], a["visible"], a["latency_s"], a["prompt_tokens"],
                            a["eval_tokens"], a["raw"]])
                f.flush()
                print(f"{setname:8s} {v:9s} {iid:6s} -> {a['state']:7s} {a['latency_s']:.2f}s  {a['raw'][:60]}")
    f.close()


def confusion(df, truth, pred, rows, cols):
    m = pd.crosstab(df[truth], df[pred]).reindex(index=rows, columns=cols, fill_value=0)
    m["n"] = m.sum(axis=1)
    return m


def report(args):
    res = pd.read_csv(args.out, dtype={"id": str})
    lab = pd.read_csv(HERE / "data/zeddump_labels.csv")
    lab["id"] = "z" + lab.id.astype(str)
    z = res[res.set == "zeddump"].merge(lab, on="id")
    z["truth_target"] = z.state.where(z.target_visible == 1, "none").where(z.target_visible != -1, "amb")
    z["dbin"] = pd.cut(z.dist, [1, 2.5, 4, 6], right=False).astype(str)
    cols = ["closed", "ajar", "open", "none", "?"]
    print("=" * 100)
    for v, g in z.groupby("variant"):
        g = g[g.truth_target != "amb"]
        print(f"\n### variant {v}: n={len(g)}  latency median {g.latency_s.median():.2f}s  p90 "
              f"{g.latency_s.quantile(.9):.2f}s  prompt tokens ~{g.prompt_tokens.median():.0f}")
        print("Target semantics (truth = state of the PROJECTED door; none = hidden by a wall):")
        print(confusion(g, "truth_target", "state_pred", ["closed", "open", "none"], cols).to_string())
        vis = g[g.truth_target != "none"]
        acc = (vis.state_pred == vis.truth_target).mean()
        print(f"  state accuracy on visible targets: {acc:.3f} ({(vis.state_pred == vis.truth_target).sum()}/{len(vis)})")
        # operational: passable = open(90 deg); not passable = closed. P(say open | passable) etc.
        pp = vis[vis.truth_target == "open"]; pn = vis[vis.truth_target == "closed"]
        print(f"  P(open|passable)={np.mean(pp.state_pred == 'open'):.3f}  P(open|shut)={np.mean(pn.state_pred == 'open'):.3f}"
              f"  P(closed|shut)={np.mean(pn.state_pred == 'closed'):.3f}  P(ajar|shut)={np.mean(pn.state_pred == 'ajar'):.3f}"
              f"  P(ajar|passable)={np.mean(pp.state_pred == 'ajar'):.3f}")
        hid = g[g.truth_target == "none"]
        print(f"  hidden target: P(visible=false)={np.mean(hid.state_pred == 'none'):.3f} (n={len(hid)})")
        if not dv.VARIANTS[v]["box"]:
            gm = g[g.main_state != "amb"].copy()
            print("  Main-door semantics (truth = most prominent door in the crop):")
            print(confusion(gm, "main_state", "state_pred", ["closed", "open", "none"], cols).to_string())
        vis = vis.assign(ok=(vis.state_pred == vis.truth_target))
        print("  by distance (visible targets, accuracy / n):")
        print(vis.groupby(["dbin", "truth_target"]).ok.agg(["mean", "size"]).round(2).to_string())
        print("  by side (+1 room 1 / -1 room 2), visible targets:")
        print(vis.groupby(["side", "truth_target"]).ok.agg(["mean", "size"]).round(2).to_string())
        wrong = vis[vis.state_pred != vis.truth_target]
        if len(wrong):
            print("  errors:", ", ".join(f"{r.id}:{r.truth_target}->{r.state_pred}(d{r.dist:.1f},s{r.side})"
                                         for r in wrong.itertuples()))
    o = res[res.set == "overlay"]
    if len(o):
        print("\n### overlay set (room_concept snapshots, all closed, n=7 per variant)")
        print(o.groupby("variant").state_pred.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--variants", default="plain,json,json_box,json_ovl,passable")
    ap.add_argument("--out", default=str(HERE / "data/offline_results.csv"))
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
