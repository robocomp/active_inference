#!/usr/bin/env python3
"""E1 corpus step: per-room class presence from ScanNet v2, in the E1 vocabulary.

One row per room (first scan of each scene): house(=scene), split, room(=scan), function, <class>...
The output has the same layout as corpus_procthor.py, so tools/brier_e1.py reads either.

    python3 tools/corpus_scannet.py [--map tools/corpora/scannet_map.toml] [--out tools/corpora/scannet_rooms.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def scene_type(txt: Path) -> str | None:
    for line in txt.read_text().splitlines():
        if line.startswith("sceneType"):
            return line.split("=", 1)[1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, default=HERE / "corpora/scannet_map.toml")
    ap.add_argument("--out", type=Path, default=HERE / "corpora/scannet_rooms.csv")
    args = ap.parse_args()

    m = tomllib.loads(args.map.read_text())
    classes = list(m["classes"])
    label_to_classes: dict[str, list[str]] = {}
    for c, labels in m["classes"].items():
        for l in labels:
            label_to_classes.setdefault(l.lower(), []).append(c)
    scans_dir = Path(m["data_dir"]) / "scans"

    per_func = Counter()
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["house", "split", "room", "function"] + classes)
        for d in sorted(scans_dir.iterdir()):
            sid = d.name
            if m.get("first_scan_only", True) and not sid.endswith("_00"):
                continue
            func = m["functions"].get(scene_type(d / f"{sid}.txt") or "")
            if func is None:
                continue
            present = set()
            for g in json.loads((d / f"{sid}.aggregation.json").read_text())["segGroups"]:
                present.update(label_to_classes.get(g["label"].lower(), []))
            w.writerow([sid[:9], "scannet", sid, func] + [int(c in present) for c in classes])
            per_func[func] += 1
    print(f"wrote {args.out}: {sum(per_func.values())} rooms {dict(per_func)}, {len(classes)} classes")


if __name__ == "__main__":
    sys.exit(main())
