#!/usr/bin/env python3 -I
"""One-off converter: the REMOVED rows of etc/door_identities.csv -> etc/door_phantom_memory.csv (common/phantom_memory).

  python3 -I tools/phantoms_from_identities.py --map room_1=1,room_2=2,room_3=3 [--in etc/door_identities.csv] [--out etc/door_phantom_memory.csv]

--map pairs each room NAME with its LTSM room_id (the `room_id` of the live room node). A removed row of a room that is not in the map
cannot be keyed and is DROPPED and listed. height_sd is unknown in the identity CSV: written as 0 (the agent then uses the prior's sd).
Numbers are read with float() and written with '.' decimals regardless of the machine's locale.
"""
import argparse
import os
import sys

HEADER = "room_id,x,y,yaw,width,height,height_sd,removed_ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="room_1=1,room_2=2,...")
    ap.add_argument("--in", dest="src", default="etc/door_identities.csv")
    ap.add_argument("--out", default="etc/door_phantom_memory.csv")
    a = ap.parse_args()
    ids = {}
    for kv in a.map.split(","):
        name, _, k = kv.partition("=")
        ids[name.strip()] = int(k)
    rows, dropped = [], []
    with open(a.src, encoding="utf-8") as f:
        cols = None
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            c = line.split(",")
            if c[0] == "room":
                cols = {n: i for i, n in enumerate(c)}
                continue
            if cols is None or "removed_ms" not in cols or len(c) <= cols["removed_ms"] or not c[cols["removed_ms"]].strip():
                continue   # not a removed row
            try:
                room, name = c[cols["room"]], c[cols["name"]]
                if room not in ids:
                    dropped.append(f"{room}:{name} (room not in --map)")
                    continue
                g = lambda n: float(c[cols[n]]) if len(c) > cols[n] and c[cols[n]].strip() else 0.0
                rows.append((ids[room], g("ap_x"), g("ap_y"), g("yaw"), g("width"), g("height"), 0.0, int(float(c[cols["removed_ms"]]))))
            except (ValueError, KeyError) as e:
                dropped.append(f"{c[0]}:{c[1] if len(c) > 1 else '?'} ({e})")
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(",".join([str(r[0])] + [repr(float(v)) for v in r[1:7]] + [str(r[7])]) + "\n")
    os.replace(tmp, a.out)
    print(f"{len(rows)} removed row(s) written to {a.out}")
    for d in dropped:
        print("DROPPED:", d, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
