#!/usr/bin/env python3
"""
Generate an authoritative parity reference for the C++ port.

Reuses the ORIGINAL Python prototype (human_kinematic_model.py + vfe_inference.py,
autograd-based) so the reference is the ground truth the C++ finite-difference port is
validated against.

Outputs two CSVs (default into this directory):
  inputs.csv     frame, kp0_x..kp17_z (54), c0..c17 (18)   -- fed to both impls
  reference.csv  frame, mu0..mu10 (11), mean_l2, rmse, uncertainty_trace

A synthetic, deterministic sequence is generated: a smooth joint-angle trajectory pushed
through the forward model, placed in the world by a random rigid transform, corrupted
with Gaussian noise + occasional missing joints, with per-joint confidences.
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.normpath(os.path.join(
    HERE, "..", "..", "python", "human_model", "human_model", "body tracking", "python"))
sys.path.insert(0, PROTO)

from human_kinematic_model import lengths_from_standard_np, HumanKinematicModel  # noqa: E402
from vfe_inference import AInfLaplacePoseEstimator, InferenceConfig              # noqa: E402


def standard_template_np() -> np.ndarray:
    """Identical to main.py: used only to derive segment lengths + face offsets."""
    kp = np.zeros((18, 3), dtype=np.float32)
    kp[0]  = [0.0, 1.65, 0.0]
    kp[14] = [0.08, 1.68, 0.05]
    kp[15] = [-0.08, 1.68, 0.05]
    kp[16] = [0.12, 1.65, 0.0]
    kp[17] = [-0.12, 1.65, 0.0]
    kp[1]  = [0.0, 1.50, 0.0]
    kp[2]  = [0.20, 1.50, 0.0]
    kp[5]  = [-0.20, 1.50, 0.0]
    kp[3]  = [0.35, 1.20, 0.0]
    kp[4]  = [0.50, 0.95, 0.0]
    kp[6]  = [-0.35, 1.20, 0.0]
    kp[7]  = [-0.50, 0.95, 0.0]
    kp[8]  = [0.15, 1.00, 0.0]
    kp[11] = [-0.15, 1.00, 0.0]
    kp[9]  = [0.15, 0.60, 0.0]
    kp[10] = [0.15, 0.10, 0.0]
    kp[12] = [-0.15, 0.60, 0.0]
    kp[13] = [-0.15, 0.10, 0.0]
    return kp


def angles_to_dict(x: np.ndarray):
    import torch
    t = torch.tensor(x, dtype=torch.float32)
    return {
        "sh_L": t[0:3], "sh_R": t[3:6],
        "el_L": t[6:7], "el_R": t[7:8],
        "lb_x": t[8:9], "lb_z": t[9:10], "lb_roll": t[10:11],
    }


def Rmat(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Ry @ Rx @ Rz).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--outdir", type=str, default=HERE)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    template = standard_template_np()
    lengths = lengths_from_standard_np(template)
    model = HumanKinematicModel(lengths, device="cpu")
    cfg = InferenceConfig(anchors=[1, 2, 5, 8, 11], device="cpu")
    estimator = AInfLaplacePoseEstimator(model, cfg)

    # Fixed world placement so Kabsch has real work to do.
    R_world = Rmat(0.6, 0.1, -0.2)
    t_world = np.array([0.5, 0.0, 2.0], dtype=np.float32)

    droppable = [4, 7, 10, 13, 16, 17]  # never anchors

    in_path = os.path.join(args.outdir, "inputs.csv")
    ref_path = os.path.join(args.outdir, "reference.csv")
    fin = open(in_path, "w")
    fref = open(ref_path, "w")
    fin.write("# frame, kp0_x..kp17_z (54), c0..c17 (18)\n")
    fref.write("# frame, mu0..mu10 (11), mean_l2, rmse, uncertainty_trace\n")

    for f in range(args.frames):
        ph = 2.0 * np.pi * f / max(1, args.frames)
        truth = np.zeros(11, dtype=np.float32)
        truth[0] = 0.4 * np.sin(ph)          # sh_L yaw
        truth[1] = -0.3 + 0.2 * np.cos(ph)   # sh_L pitch
        truth[3] = -0.4 * np.sin(ph)         # sh_R yaw (mirrored)
        truth[4] = -0.3 + 0.2 * np.cos(ph)   # sh_R pitch
        truth[6] = 0.5 + 0.4 * (0.5 + 0.5 * np.sin(ph))   # el_L
        truth[7] = 0.5 + 0.4 * (0.5 + 0.5 * np.sin(ph))   # el_R
        truth[10] = 0.1 * np.sin(0.5 * ph)   # lb_roll

        canon = model(angles_to_dict(truth)).detach().cpu().numpy().astype(np.float32)
        world = (canon @ R_world.T + t_world).astype(np.float32)
        world = world + rng.normal(0.0, args.noise, world.shape).astype(np.float32)

        conf = np.clip(rng.normal(85.0, 8.0, 18), 0.0, 100.0).astype(np.float32)

        # Occasionally drop a non-anchor joint.
        live = world.copy()
        if f % 5 == 2:
            j = droppable[(f // 5) % len(droppable)]
            live[j] = np.nan
            conf[j] = 0.0

        res = estimator.infer(live, kp_conf_np=conf)

        # inputs row
        kp_flat = ",".join(
            ("nan" if not np.isfinite(v) else f"{v:.6f}") for v in live.reshape(-1))
        conf_flat = ",".join(f"{c:.6f}" for c in conf)
        fin.write(f"{f},{kp_flat},{conf_flat}\n")

        # reference row
        mu_flat = ",".join(f"{m:.6f}" for m in np.asarray(res.mu).reshape(-1))
        fref.write(f"{f},{mu_flat},{res.mean_l2:.6f},{res.rmse:.6f},"
                   f"{res.uncertainty_trace:.6f}\n")

    fin.close()
    fref.close()
    print(f"wrote {in_path}")
    print(f"wrote {ref_path}")


if __name__ == "__main__":
    main()
