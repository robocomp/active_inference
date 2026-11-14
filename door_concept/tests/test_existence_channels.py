#!/usr/bin/env -S python3 -I
"""Task 8b: tools/existence_channels.py splits one door's change of L by channel. A synthetic door_existence_log.csv with a known
split (ZED -0.5, Ricoh -0.2, support +0.1, size prior -0.3, base prior -0.1, and a +0.05 that no channel owns) must be read back exactly,
a second door in the same file must not leak in, and the views table must average q and hit per camera.
Run from door_concept/:  python3 -I tests/test_existence_channels.py"""
import csv, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import existence_channels as ec

COLS = ["cycle", "node", "L", "prior_band", "prior_base", "prior_size", "prior_extent", "zed_dl", "ric_dl", "geo_dl", "act_dl", "sup_dl"]
d = tempfile.mkdtemp()
p = os.path.join(d, "ex.csv")
with open(p, "w", newline="") as f:
    w = csv.writer(f); w.writerow(COLS)
    # first row: L 2.0; second: the whole span's change arrives in one update
    w.writerow([10, "door_a", 2.0, 0, -0.4, -0.8, 0, "nan", "nan", "nan", "nan", "nan"])
    w.writerow([11, "door_a", 2.0 - 0.5 - 0.2 + 0.1 - 0.3 - 0.1 + 0.05, 0, -0.5, -1.1, 0, -0.5, -0.2, 0, "nan", 0.1])
    w.writerow([10, "door_b", 9.0, 0, -0.4, -0.8, 0, 0.0, 0.0, 0.0, "nan", 0.0])
    w.writerow([11, "door_b", 9.0, 0, -0.4, -0.8, 0, 7.0, 0.0, 0.0, "nan", 0.0])
v = os.path.join(d, "views.csv")
with open(v, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cycle", "door", "camera", "q", "hit"])
    w.writerow([11, "door_a", "zed", 0.6, 0.0]); w.writerow([11, "door_a", "zed", 0.4, 0.5]); w.writerow([11, "door_b", "zed", 0.9, 0.9])

fails = 0
def check(name, got, want, tol=1e-5):
    global fails
    if abs(got - want) > tol:
        print(f"FAIL {name}: {got} != {want}"); fails += 1

s = ec.split(list(csv.DictReader(open(p))), "door_a", 10, 11)
check("dL", s["dL"], -0.95)
check("zed", s["zed_dl"], -0.5); check("ric", s["ric_dl"], -0.2); check("sup", s["sup_dl"], 0.1)
check("size", s["prior_size"], -0.3); check("base", s["prior_base"], -0.1)
check("rest", s["rest"], 0.05 - 0.0)   # what no channel and no prior term owns
vw = ec.views(list(csv.DictReader(open(v))), "door_a", 10, 11)["zed"]
check("view rows", vw[0], 2); check("view q", vw[1], 0.5); check("view hit", vw[2], 0.25)
out = subprocess.run([sys.executable, "-I", os.path.join(HERE, "..", "tools", "existence_channels.py"), p, "door_a", "10", "11", "--views", v],
                     capture_output=True, text=True)
if out.returncode != 0 or "zed_dl" not in out.stdout or "-0.500" not in out.stdout:
    print("FAIL command line:", out.stdout, out.stderr); fails += 1
print("PASS" if fails == 0 else f"{fails} FAIL")
sys.exit(1 if fails else 0)
