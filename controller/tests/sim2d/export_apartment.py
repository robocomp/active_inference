"""Export threerooms-piso.wbt as 2-D segments at the LiDAR band, plus its doors, for the controller's sim2d mock.
   Output (classic locale, one record per line): 'seg x0 y0 x1 y1' and 'door name cx cy nx ny tx ty width'."""
import sys
sys.path.insert(0, "/home/pbustos/robocomp/components/active_inference.worktrees/rt-build/room_concept/estimation_scripts/scripts")
import adoption_mock as am
W = am.parse_world(am.WORLD, lidar_z=0.5)
out = sys.argv[1]
with open(out, "w") as f:
    for p, q in W.segments:
        f.write(f"seg {p[0]:.4f} {p[1]:.4f} {q[0]:.4f} {q[1]:.4f}\n")
    for k, d in W.doors.items():
        c, n, t = d["centre"], d["normal"], d["tangent"]
        f.write(f"door {k} {c[0]:.4f} {c[1]:.4f} {n[0]:.4f} {n[1]:.4f} {t[0]:.4f} {t[1]:.4f} {d['width']:.4f}\n")
xs = [v for p, q in W.segments for v in (p[0], q[0])]; ys = [v for p, q in W.segments for v in (p[1], q[1])]
print(len(W.segments), "segments;", len(W.doors), "doors:", list(W.doors), f"bbox x {min(xs):.2f}..{max(xs):.2f} y {min(ys):.2f}..{max(ys):.2f}", W.meshes)
