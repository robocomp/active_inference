#!/usr/bin/env python3
"""Per-room E2 answers on ScanNet: the top function under each source of π_cf and the one-image answer.

Rebuilds the same sources as tools/function_evidence_e2.py (vlm, vlm_cal, handwritten) and writes
tools/e2_out/e2_per_room_scannet.csv (room, truth, objects present, top function per source, image answer).

    python3 tools/e2_per_room.py
"""
from __future__ import annotations

import csv
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from function_evidence_e2 import derived_notf, fit_levels, lam  # noqa: E402


def main() -> None:
    rooms = list(csv.DictReader((HERE / "corpora/scannet_rooms.csv").open()))
    classes = list(rooms[0])[4:]
    funcs = sorted({r["function"] for r in rooms})
    pf = {r["func"]: float(r["p_f"]) for r in csv.DictReader((HERE / "e1_out/function_prior.csv").open())}
    vrows = list(csv.DictReader((HERE / "e1_out/level4_priors.csv").open()))
    all_funcs = sorted({r["func"] for r in vrows})
    vlm = {(r["cls"], r["func"]): float(r["pi_cf"]) for r in vrows}
    vlm_n = {(r["cls"], r["func"]): float(r["pi_c_notf"]) for r in vrows}
    cal, _ = fit_levels(vlm, HERE / "corpora/procthor_rooms.csv")
    vc = {k: cal[v] for k, v in vlm.items()}
    t = tomllib.loads((HERE / "handwritten_function_table.toml").read_text())
    hw = {(c, f): (t["signature"] if c in t["functions"].get(f, []) else t["other"]) for c in classes for f in all_funcs}
    sources = {"vlm": (vlm, vlm_n), "vlm_cal": (vc, derived_notf(vc, classes, all_funcs)),
               "handwritten": (hw, derived_notf(hw, classes, all_funcs))}
    img = {r["scan"]: r["answer"] for r in map(json.loads, (HERE / "e2_out/image_answers.jsonl").open())
           if r["k"] == 1 and r.get("num_ctx") is None}
    out = HERE / "e2_out/e2_per_room_scannet.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["room", "truth", "objects"] + list(sources) + ["claims_vlm_cal", "image"])
        for r in rooms:
            z = {c: int(r[c]) for c in classes}
            row = [r["room"], r["function"], ";".join(c for c in classes if z[c])]
            claims = ""
            for name, (pi, pin) in sources.items():
                ls = {f: lam(z, classes, pf[f], pi, pin, f) for f in funcs}
                row.append(max(ls, key=ls.get))
                if name == "vlm_cal":
                    claims = ";".join(f for f, v in ls.items() if v > 0)
            w.writerow(row + [claims, img[r["room"]]])
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
