"""tools/existence_swing.py reports alternating geo_dl as sign changes (run: python3 -I tests/test_existence_swing.py)."""
import csv, os, subprocess, sys, tempfile

HDR = ("cycle,node,L,p_exists,cx,cy,role,delete_line,view_id,ric_view_id,prior_band,prior_base,prior_size,carried,crossing,"
       "ledger,n_views,zed_dl,ric_dl,geo_dl,act_dl,zed_q,ric_q,zed_hit,ric_hit,kappa,geo_x,geo_mu0,geo_mu1,geo_s,sup_x,sup_dl,"
       "Sc,Sp,sat_term,sup_c,sup_q,prior_extent").split(",")
assert len(HDR) == 38
here = os.path.dirname(os.path.abspath(__file__))
tool = os.path.join(here, "..", "tools", "existence_swing.py")

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "log.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, HDR, restval="0")
        w.writeheader()
        for i, g in enumerate([3, -3, 3, -3, 3, -3]):
            w.writerow({"cycle": i, "node": "door_2", "geo_dl": g, "geo_x": 1.0, "geo_mu0": 0.5, "geo_mu1": 0.9, "geo_s": 0.2, "sup_x": 0.1, "cx": 1, "cy": 2})
    r = subprocess.run([sys.executable, "-I", tool, p, "--node", "door_2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    line = [l for l in r.stdout.splitlines() if "door_2" in l and "sign_changes" in l]
    assert line and "sign_changes=5" in line[0], r.stdout
print("PASS")
