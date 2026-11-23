#!/usr/bin/env python3
"""E1 corpus step: per-room class presence from ProcTHOR-10k, in the E1 vocabulary.

Reads the houses (jsonl.gz, one house per line) and writes one row per room:
    house, split, room, function, <class>... (1 = at least one instance in the room)
An object's room is the number in its id ("Bed|4|0|0" -> room|4; children "Mug|surface|4|0" -> room|4);
children inherit their parent's room. Rooms whose type is not mapped are skipped.

    python3 tools/corpus_procthor.py [--map tools/corpora/procthor_map.toml] [--out tools/corpora/procthor_rooms.csv]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def room_of(obj_id: str) -> str | None:
    parts = obj_id.split("|")
    for p in parts[1:]:
        if p.isdigit():
            return p
    return None


def rooms_of_house(house: dict, type_to_classes: dict[str, list[str]]) -> dict[str, set[str]]:
    """room number -> set of E1 classes present."""
    present: dict[str, set[str]] = {}

    def walk(o: dict, room: str | None) -> None:
        r = room_of(o["id"]) if room is None else room
        t = o["id"].split("|")[0]
        if r is not None:
            for c in type_to_classes.get(t, []):
                present.setdefault(r, set()).add(c)
        for ch in o.get("children") or []:
            walk(ch, r)

    for o in house["objects"]:
        walk(o, None)
    return present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, default=HERE / "corpora/procthor_map.toml")
    ap.add_argument("--out", type=Path, default=HERE / "corpora/procthor_rooms.csv")
    args = ap.parse_args()

    m = tomllib.loads(args.map.read_text())
    classes = list(m["classes"])
    type_to_classes: dict[str, list[str]] = {}
    for c, types in m["classes"].items():
        for t in types:
            type_to_classes.setdefault(t, []).append(c)
    data = Path(m["data_dir"])

    n_rooms = n_unassigned = 0
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["house", "split", "room", "function"] + classes)
        for split in ("train", "val", "test"):
            with gzip.open(data / f"{split}.jsonl.gz", "rt") as f:
                for i, line in enumerate(f):
                    house = json.loads(line)
                    present = rooms_of_house(house, type_to_classes)
                    room_ids = {r["id"].split("|")[1] for r in house["rooms"]}
                    n_unassigned += sum(1 for r in present if r not in room_ids)
                    for r in house["rooms"]:
                        func = m["functions"].get(r["roomType"])
                        if func is None:
                            continue
                        num = r["id"].split("|")[1]
                        p = present.get(num, set())
                        w.writerow([f"{split}_{i}", split, num, func] + [int(c in p) for c in classes])
                        n_rooms += 1
    print(f"wrote {args.out}: {n_rooms} rooms, {len(classes)} classes"
          + (f"; WARNING {n_unassigned} object rooms not in the house's room list" if n_unassigned else ""))


if __name__ == "__main__":
    sys.exit(main())
