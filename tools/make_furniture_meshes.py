#!/usr/bin/env python3
"""make_furniture_meshes.py — typical display meshes for bed_concept and toilet_concept, generated procedurally.

WHY PROCEDURAL. Each concept agent publishes `mesh_path` on its node so any viewer can draw it, and the mesh must live
in the agent's own folder. The Webots Bed/Toilet protos carry real meshes, but they are "Licensed for use only with
Webots" (cyberbotics.com/webots_assets_license), so they are not copied here. (nightstand_concept's mesh is the Poly
Haven "Painted Wooden Nightstand", CC0.) Dimensions follow the Webots models in threerooms-piso.

THE CONTRACT (common/graph3d/graph3d_scene_builder.cpp): a display mesh is a UNIT BOX — x,y ∈ [-0.5,0.5], z ∈ [0,1] —
stretched by the node's width_m / depth_m / height_m. +Y is the BACK (the wall side; the belief's front is −Y).
  · toilet: height_m is the whole toilet (tank top), so the mesh fills the unit box.
  · bed: height_m is the fitted MATTRESS box (the headboard is a fixed template, not a DOF, and the fitter reads
    height_m back as the mattress on re-attach), so z = 1 is the mattress top and the headboard rises ABOVE it to
    headboard/mattress ≈ 1.9. That one overshoot is deliberate; everything else is inside the unit box.
usage: python3 tools/make_furniture_meshes.py      (from the repo root; rewrites the two meshes + .mtl files)
"""
import pathlib
import numpy as np
import trimesh
from trimesh.creation import box, cylinder, capsule


def bx(x0, x1, y0, y1, z0, z1):
    m = box(extents=[x1 - x0, y1 - y0, z1 - z0])
    m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return m


def rounded(x0, x1, y0, y1, z0, z1, r):
    """A box with rounded vertical edges and a soft top: a slab of 4 corner cylinders + 2 cross boxes."""
    parts = [bx(x0 + r, x1 - r, y0, y1, z0, z1), bx(x0, x1, y0 + r, y1 - r, z0, z1)]
    for cx in (x0 + r, x1 - r):
        for cy in (y0 + r, y1 - r):
            c = cylinder(radius=r, height=z1 - z0, sections=16); c.apply_translation([cx, cy, (z0 + z1) / 2])
            parts.append(c)
    return trimesh.util.concatenate(parts)


def ellipse_loft(rings, sections=40):
    """A closed surface through elliptical rings [(z, rx, ry, cy), ...] (bottom→top), capped at both ends."""
    th = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    V = []; F = []
    for z, rx, ry, cy in rings:
        V += [[rx * np.cos(t), cy + ry * np.sin(t), z] for t in th]
    n = len(rings)
    for k in range(n - 1):
        for i in range(sections):
            a, b = k * sections + i, k * sections + (i + 1) % sections
            F += [[a, b, b + sections], [a, b + sections, a + sections]]
    for k, flip in ((0, True), (n - 1, False)):
        z, _, _, cy = rings[k]; c = len(V); V.append([0, cy, z])
        for i in range(sections):
            a, b = k * sections + i, k * sections + (i + 1) % sections
            F.append([c, b, a] if flip else [c, a, b])
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def write(mesh_parts, name, header):
    """Each (mesh, material) part becomes its own OBJ group with a usemtl, so a viewer can colour the parts."""
    d = pathlib.Path(f"{name}_concept/meshes"); d.mkdir(parents=True, exist_ok=True)
    lines = [f"# {name}.obj — {l}" if i == 0 else f"# {l}" for i, l in enumerate(header)] + [f"mtllib {name}.mtl"]
    off = 0; mats = {}
    for part, (mat, kd) in mesh_parts:
        mats[mat] = kd
        lines.append(f"o {mat}"); lines.append(f"usemtl {mat}")
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in part.vertices]
        lines += [f"f {a + off + 1} {b + off + 1} {c + off + 1}" for a, b, c in part.faces]
        off += len(part.vertices)
    (d / f"{name}.obj").write_text("\n".join(lines) + "\n")
    (d / f"{name}.mtl").write_text("".join(f"newmtl {m}\nKa 0.05 0.05 0.05\nKd {kd}\nd 1.0\nillum 1\n\n"
                                          for m, kd in mats.items()))
    print(f"{d / name}.obj: {sum(len(p.faces) for p, _ in mesh_parts)} faces")


def bed():
    W, L, H, HB = 1.75, 2.10, 0.65, 1.25      # metres: width, length (front→back = local Y), mattress top, headboard
    s = lambda x, y, z: (x / W, y / L, z / H)  # metres → unit box (z: mattress top = 1)
    wood = ("frame", "0.30 0.20 0.13"); linen = ("mattress", "0.92 0.92 0.90")
    blanket = ("blanket", "0.36 0.45 0.62"); pillow = ("pillow", "0.96 0.96 0.97")
    P = []
    def add(m, mat):
        m = m.copy(); m.vertices = np.array([s(*v) for v in m.vertices]); P.append((m, mat))
    hw, hl = W / 2, L / 2
    for x in (-hw + 0.05, hw - 0.05):                                 # legs
        for y in (-hl + 0.05, hl - 0.05):
            add(bx(x - 0.04, x + 0.04, y - 0.04, y + 0.04, 0.0, 0.20), wood)
    add(bx(-hw, hw, -hl, hl, 0.20, 0.36), wood)                       # side rails / base
    add(rounded(-hw + 0.04, hw - 0.04, -hl + 0.06, hl - 0.12, 0.36, 0.61, 0.06), linen)          # mattress
    add(rounded(-hw + 0.01, hw - 0.01, -hl + 0.02, hl - 0.75, 0.58, 0.65, 0.05), blanket)        # blanket (foot 2/3)
    for x in (-0.42, 0.42):                                           # pillows at the head (+Y)
        p = capsule(radius=0.07, height=0.55, count=[16, 16])
        p.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        p.apply_scale([1.0, 0.55 / 0.14 * 0.12 / 0.55 * 2.4, 0.6]); p.apply_translation([x, hl - 0.33, 0.66])
        add(p, pillow)
    add(bx(-hw, hw, hl - 0.10, hl, 0.0, HB), wood)                    # headboard (+Y, the wall)
    add(bx(-hw, hw, -hl, -hl + 0.08, 0.0, 0.83), wood)                # footboard
    write(P, "bed", ["typical bed, procedural (tools/make_furniture_meshes.py), 1.75 x 2.10 m like Webots' Bed.",
                     "Unit box per common/graph3d: x,y in [-0.5,0.5]; z = 1 is the MATTRESS top (height_m is the",
                     "fitted mattress box), the headboard rises above it to 1.25/0.65. +Y = headboard = wall side."])


def toilet():
    W, D, H = 0.58, 0.73, 1.02              # width, depth (front→back = local Y), tank top
    s = lambda x, y, z: (x / W, y / D, z / H)
    china = ("porcelain", "0.95 0.95 0.96"); seat = ("seat", "0.88 0.88 0.90"); metal = ("button", "0.75 0.75 0.78")
    P = []
    def add(m, mat):
        m = m.copy(); m.vertices = np.array([s(*v) for v in m.vertices]); P.append((m, mat))
    yb = -0.08                              # bowl centre (in front of the tank); back of the tank at +D/2
    add(ellipse_loft([(0.00, 0.13, 0.17, yb + 0.06), (0.22, 0.15, 0.20, yb + 0.03),
                      (0.34, 0.19, 0.25, yb), (0.40, 0.20, 0.27, yb)]), china)                   # pedestal + bowl
    add(ellipse_loft([(0.40, 0.205, 0.275, yb), (0.43, 0.205, 0.275, yb)]), seat)               # seat
    add(ellipse_loft([(0.43, 0.20, 0.27, yb), (0.45, 0.195, 0.265, yb)]), seat)                 # lid
    add(rounded(-0.23, 0.23, D / 2 - 0.22, D / 2, 0.42, 0.98, 0.04), china)                     # tank (+Y, wall)
    add(rounded(-0.25, 0.25, D / 2 - 0.235, D / 2, 0.98, H, 0.04), china)                       # tank lid
    b = cylinder(radius=0.03, height=0.012, sections=16); b.apply_translation([0, D / 2 - 0.11, H + 0.006 - 0.012])
    add(b, metal)
    add(bx(-0.10, 0.10, yb + 0.15, D / 2 - 0.20, 0.30, 0.45), china)                            # bowl→tank neck
    add(rounded(-0.12, 0.12, yb + 0.05, D / 2 - 0.06, 0.0, 0.43, 0.04), china)                 # rear plinth under the tank
    write(P, "toilet", ["typical toilet, procedural (tools/make_furniture_meshes.py), 0.58 x 0.73 x 1.02 m like",
                        "Webots' Toilet. Unit box per common/graph3d: x,y in [-0.5,0.5], z in [0,1]. +Y = tank = wall side."])


if __name__ == "__main__":
    bed(); toilet()
