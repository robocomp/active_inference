#!/usr/bin/env python3
"""
export_dinov2.py — export DINOv2-with-registers ViT-S/14 to ONNX for the retina place-memory stage.

WHAT THIS PRODUCES
    models/dinov2/dinov2_vits14_reg_448x224.onnx
      input   pixel_values  [1, 3, 224, 448]  RGB, ImageNet-normalised, planar CHW
      output  cls           [1, 384]          global token — PREFILTER ONLY, never part of the score
      output  patch         [1, 16, 32, 384]  patch grid, ALREADY reshaped (h, w, d)

WHY THE REGISTER VARIANT.  Plain DINOv2 emits a few extremely high-norm "artifact" patch tokens that
carry global information; the register variant was introduced to remove them.  Our entire
representation is a POOL OVER PATCH TOKENS, so one artifact token would dominate a sector's mean.
Free upgrade at identical cost.

★ WHY THE GRID IS RESHAPED IN THE GRAPH AND NOT IN C++.  A transposed patch grid (h/w swapped) turns
azimuth sectors into ELEVATION BANDS.  The circular shift then means nothing, and every downstream
number still looks plausible — no crash, no warning.  Doing the reshape here makes the axis order a
property of the model file that the C++ side cannot get wrong.

★ WHY 448x224.  The ricoh panorama is 1920x960 equirectangular, i.e. exactly 2:1, and 448x224 is also
2:1 — so there is no aspect distortion.  Patch 14 gives a 16x32 grid: 32 azimuth columns of 11.25
each, which group evenly into 16 sectors of 22.5.

★ POSITIONAL EMBEDDING.  DINOv2 pretrains on a SQUARE patch grid and bicubically interpolates its
pos_embed to h/14 x w/14 at call time.  That is the only reason a non-square 448x224 input works at
all, and it must be requested explicitly (interpolate_pos_encoding=True) on the HF path.

★ PREPROCESSING MUST MATCH retina/src/place_encoder.cpp EXACTLY.  See the table in the plan; the one
that actually bites is the RESIZE FILTER.  1920 -> 448 is a 4.3x downscale: torchvision antialias=True
(PIL bilinear) corresponds to cv::INTER_AREA, NOT cv::INTER_LINEAR.  Using INTER_LINEAR aliases badly
and presents as "the model is bad" rather than "the preprocessing differs".

Usage:
    python3 tools/export_dinov2.py                     # export + self-verify (synthetic)
    python3 tools/export_dinov2.py --pano etc/depth_frames/1786014271455.jpg   # + real roll test
"""

import argparse
import os
import sys

# ★ xformers' fused attention does not trace to ONNX.  This is the single most likely export failure
#   on the torch.hub path, and it must be set BEFORE torch/dinov2 import.
os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import torch
import torch.nn as nn

# ── geometry ────────────────────────────────────────────────────────────────────────────────────
IN_W, IN_H = 448, 224
PATCH = 14
GW, GH = IN_W // PATCH, IN_H // PATCH        # 32 columns x 16 rows
DIM = 384
HF_ID = "facebook/dinov2-with-registers-small"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PanoDinoV2(nn.Module):
    """Backbone + the split/reshape, so the ONNX graph itself carries the token layout."""

    def __init__(self, backbone, n_reg: int):
        super().__init__()
        self.backbone = backbone
        self.n_reg = n_reg

    def forward(self, pixel_values):
        out = self.backbone(pixel_values, interpolate_pos_encoding=True)
        h = out.last_hidden_state                      # [1, 1 + n_reg + GH*GW, DIM]
        cls = h[:, 0]                                  # [1, DIM]
        # ★ Patches start AFTER the cls token AND the register tokens: index 1 + n_reg, not 1.
        #   An off-by-n_reg scrambles the whole grid exactly like a transpose does.
        patch = h[:, 1 + self.n_reg:]                  # [1, GH*GW, DIM]
        patch = patch.reshape(1, GH, GW, DIM)          # [1, 16, 32, 384]  (row-major h, w)
        return cls, patch


def bake_pos_embed(model):
    """★ Pre-interpolate the positional embedding for our FIXED input size, once, and freeze it.

    DINOv2 pretrains on a square 37x37 patch grid and bicubically interpolates pos_embed to the
    actual h/14 x w/14 at every forward.  That interpolation is `antialias=True` bicubic, which
    torch exports as `aten::_upsample_bicubic2d_aa` — an op ONNX opset 17 does not have, and the
    reason a naive export dies.

    But our input size NEVER changes, so the interpolation is a constant.  Computing it once in
    eager mode and replacing the parameter removes the op from the graph entirely: no unsupported
    operator, less work per inference, and a simpler graph for TensorRT to partition.  Verified
    below to be numerically identical to the runtime path.
    """
    emb = model.backbone.embeddings
    n_patches = GH * GW
    with torch.no_grad():
        probe = torch.zeros(1, 1 + n_patches, DIM)
        pe = emb.interpolate_pos_encoding(probe, IN_H, IN_W)        # [1, 1 + GH*GW, DIM]
    if tuple(pe.shape) != (1, 1 + n_patches, DIM):
        raise SystemExit(f"interpolated pos_embed has shape {tuple(pe.shape)}, "
                         f"expected {(1, 1 + n_patches, DIM)}")
    old = emb.position_embeddings.shape
    emb.position_embeddings = nn.Parameter(pe, requires_grad=False)
    # Now short-circuit the method: the stored embedding is already the right size.
    emb.interpolate_pos_encoding = lambda embeddings, height, width: emb.position_embeddings
    print(f"[posembed] baked {tuple(old)} -> {tuple(pe.shape)} for {IN_H}x{IN_W} "
          f"({GH}x{GW} patches); runtime interpolation removed")


def verify_bake(model_baked, n_reg):
    """The bake must be a no-op numerically. Compare against a freshly loaded, un-baked model."""
    from transformers import AutoModel
    ref = PanoDinoV2(AutoModel.from_pretrained(HF_ID).eval(), n_reg).eval()
    x = torch.randn(1, 3, IN_H, IN_W)
    with torch.no_grad():
        r_cls, r_patch = ref(x)
        b_cls, b_patch = model_baked(x)
    ok = True
    for name, r, b in (("cls", r_cls, b_cls), ("patch", r_patch, b_patch)):
        d = float((r - b).abs().max())
        print(f"[posembed] bake-vs-runtime {name:6s} max|delta|={d:.3e}")
        if d > 1e-5:
            ok = False
    print(f"[posembed] {'PASS' if ok else 'FAIL'}")
    return ok


def build_model():
    from transformers import AutoModel, AutoConfig

    cfg = AutoConfig.from_pretrained(HF_ID)
    n_reg = getattr(cfg, "num_register_tokens", None)
    if n_reg is None:
        raise SystemExit(f"{HF_ID} has no num_register_tokens — wrong checkpoint?")
    if cfg.hidden_size != DIM:
        raise SystemExit(f"hidden_size {cfg.hidden_size} != expected {DIM}")
    if cfg.patch_size != PATCH:
        raise SystemExit(f"patch_size {cfg.patch_size} != expected {PATCH}")

    backbone = AutoModel.from_pretrained(HF_ID).eval()
    print(f"[model] {HF_ID}  dim={cfg.hidden_size} patch={cfg.patch_size} "
          f"num_register_tokens={n_reg}")
    return PanoDinoV2(backbone, n_reg).eval(), n_reg


def check_token_layout(model, n_reg):
    """Assert the sequence length is exactly 1 + n_reg + GH*GW before trusting the slice."""
    x = torch.zeros(1, 3, IN_H, IN_W)
    with torch.no_grad():
        raw = model.backbone(x, interpolate_pos_encoding=True).last_hidden_state
    expect = 1 + n_reg + GH * GW
    if raw.shape[1] != expect:
        raise SystemExit(f"token layout mismatch: got {raw.shape[1]} tokens, expected {expect} "
                         f"(= 1 cls + {n_reg} reg + {GH}x{GW} patches)")
    print(f"[layout] {raw.shape[1]} tokens = 1 cls + {n_reg} reg + {GH}x{GW} patches  OK")


def export(model, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dummy = torch.randn(1, 3, IN_H, IN_W)
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["pixel_values"], output_names=["cls", "patch"],
        opset_version=17,            # native LayerNormalization; TRT-friendly
        do_constant_folding=True,
        dynamo=False,                # the dynamo path can emit ops TRT rejects
        dynamic_axes=None,           # fixed shape: one input size exists, and TRT caching likes it
    )
    print(f"[export] wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def verify_parity(model, out_path):
    """torch vs ORT on the same input. fit_envelope-style: print the numbers, do not just assert."""
    import onnxruntime as ort

    x = torch.randn(1, 3, IN_H, IN_W)
    with torch.no_grad():
        t_cls, t_patch = model(x)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    o_cls, o_patch = sess.run(["cls", "patch"], {"pixel_values": x.numpy()})

    if tuple(o_cls.shape) != (1, DIM) or tuple(o_patch.shape) != (1, GH, GW, DIM):
        raise SystemExit(f"output shape mismatch: cls {o_cls.shape} patch {o_patch.shape}")

    ok = True
    for name, t, o in (("cls", t_cls.numpy(), o_cls), ("patch", t_patch.numpy(), o_patch)):
        d = float(np.abs(t - o).max())
        c = float(np.dot(t.ravel(), o.ravel()) /
                  (np.linalg.norm(t.ravel()) * np.linalg.norm(o.ravel()) + 1e-12))
        print(f"[parity] {name:6s} shape={o.shape} max|delta|={d:.3e} cosine={c:.6f}")
        if d > 1e-3 or c < 0.9999:
            ok = False
    print(f"[parity] {'PASS' if ok else 'FAIL'}")
    return ok


# ── pooling (mirrors what place_encoder.cpp must do) ────────────────────────────────────────────
def l2(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def pool_sectors(patch, n_sectors=16, pool_p=3.0, band=(6, 12), center=True, sector_soft=0.0):
    """patch [1,GH,GW,D] -> [n_sectors, D], L2-normalised. Mirrors place_encoder.cpp exactly.

    ★ ALL THREE DEFAULTS ARE MEASURED, NOT CHOSEN.  Sweep of band x centering x pool_p over 5
    panoramas x 15 rolls = 75 trials each (2026-08-28, tools/export_dinov2.py --sweep):

        band     ctr  p |  roll acc   peak margin
        (0,16)   F    1 |    0.57       +0.017     <- the "obvious" defaults
        (6,12)   F    1 |    1.00       +0.066
        (6,12)   T    3 |    1.00       +0.354     <- shipped

    and the same ranking holds for PLACE discrimination (similarity contrast between frames <2 s
    apart and >60 s apart, 90 frames over 2399 s): +0.046 for (0,16)/F/1 vs +0.169 for (6,12)/T/3.
    So the choice is not a trade-off between yaw mechanics and position resolution — one config
    wins both.

    ★ WHY THE ELEVATION BAND.  Measured per-row |mean over azimuth| on a real panorama (1.0 = that
    row is CONSTANT around the circle, i.e. carries no bearing information):
        rows 0-4   0.88-0.93   ceiling
        rows 7-11  0.70-0.77   horizon + near floor   <- where the information is
        rows 14-15 0.92-0.95   THE ROBOT'S OWN BODY
    The ricoh looks down onto the robot, so the bottom of every panorama is the same rigid object
    from every position in the room — pure nuisance, and it was diluting every sector.

    ★ WHY CENTERING.  |mean token| over a whole frame is 0.656: DINOv2 tokens share a large common
    component that carries no positional information at all.  A mean over L2-normalised tokens
    inherits it, so every sector-to-sector cosine starts around 0.9 before any content matters and
    the match peak drowns in it.  Subtracting the per-frame mean removes it.  ★ Per-FRAME (not a
    global dataset mean) is deliberate: rolling a panorama permutes sectors but leaves the frame
    mean unchanged, so centering is exactly rotation-invariant and cannot break the circular shift.

    ★ WHY GeM p=3 AFTER ALL.  The usual objection is that GeM assumes non-negative features and
    DINOv2 tokens are signed.  The sign-preserving form used here sidesteps that, and once the
    common mode is removed the sharpening toward dominant activations is worth +0.12 of margin.
    """
    g = patch[0][band[0]:band[1]]                      # [rows, GW, D]
    g = l2(g)                                          # per-token normalise
    if center:
        g = g - g.reshape(-1, g.shape[-1]).mean(axis=0)
    cols_per = GW // n_sectors
    out = np.zeros((n_sectors, g.shape[-1]), dtype=np.float32)
    # Per-column pooled vectors first, then blend columns by the sector window. ★ Must stay in lockstep
    # with PlaceEncoder::pool in retina/src/place_encoder.cpp — a parity harness checks them.
    cols = np.empty((GW, g.shape[-1]), dtype=np.float32)
    for j in range(GW):
        b = g[:, j, :]
        cols[j] = b.mean(axis=0) if abs(pool_p - 1.0) < 1e-9 else \
                  (np.sign(b) * np.abs(b) ** pool_p).mean(axis=0)
    for s in range(n_sectors):
        centre = (s + 0.5) * cols_per
        d = np.abs(np.arange(GW) + 0.5 - centre)
        d = np.minimum(d, GW - d)                       # wrap: azimuth is circular
        if sector_soft > 0.0:
            hw = sector_soft * cols_per
            w = np.where(d < hw, 0.5 * (1 + np.cos(np.pi * d / hw)), 0.0)
        else:
            w = ((np.arange(GW) >= s * cols_per) & (np.arange(GW) < (s + 1) * cols_per)).astype(float)
        if w.sum() <= 0:
            continue
        v = (w[:, None] / w.sum() * cols).sum(axis=0)
        out[s] = v if abs(pool_p - 1.0) < 1e-9 else np.sign(v) * np.abs(v) ** (1.0 / pool_p)
    return l2(out)


def circular_match(q, k):
    """sim[s] = mean_i <Q[(i+s) mod S], K[i]>. Returns (best_shift, sim array)."""
    S = q.shape[0]
    C = q @ k.T                                        # C[j][i] = <Q[j], K[i]>
    sim = np.array([np.mean([C[(i + s) % S, i] for i in range(S)]) for s in range(S)])
    return int(np.argmax(sim)), sim


def preprocess(bgr):
    """The EXACT steps place_encoder.cpp must reproduce."""
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)   # ★ AREA, not LINEAR
    f = r.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(f.transpose(2, 0, 1)[None])           # NCHW


def roll_test(out_path, pano_path, n_sectors=16):
    """★ THE CENTRAL CLAIM: a yaw change must be exactly a cyclic permutation of the sector array.

    Roll the panorama by W*k/S pixels; the recovered sector shift must be k, for EVERY k.
    If this fails, the circular-shift design is dead and no C++ should be written.

    ★ SIGN CONVENTION, established here rather than assumed: np.roll(img, +px, axis=1) moves content
    toward HIGHER column index, so a base sector i appears at rolled index i+k.  With
    sim[s] = mean_i <Q[(i+s) mod S], K[i]> the peak lands at s = +k.  Getting this backwards yields a
    REFLECTION rather than a rotation -- the same failure signature room_concept documented for
    robot_gt_angle -- so the C++ side must reproduce this convention exactly.
    """
    import cv2, onnxruntime as ort

    bgr = cv2.imread(pano_path)
    if bgr is None:
        raise SystemExit(f"cannot read {pano_path}")
    if bgr.shape[1] != 2 * bgr.shape[0]:
        print(f"[roll] WARNING {pano_path} is {bgr.shape[1]}x{bgr.shape[0]}, not 2:1")
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    def sectors(img):
        _, p = sess.run(["cls", "patch"], {"pixel_values": preprocess(img)})
        return pool_sectors(p, n_sectors=n_sectors)

    base = sectors(bgr)
    W = bgr.shape[1]
    print(f"[roll] {os.path.basename(pano_path)}  S={n_sectors}  sector={360/n_sectors:.2f} deg")
    nok, margins = 0, []
    for k in range(1, n_sectors):
        shift, sim = circular_match(sectors(np.roll(bgr, (W * k) // n_sectors, axis=1)), base)
        ss = np.sort(sim)
        margins.append(ss[-1] - ss[-2])
        good = shift == k
        nok += good
        if not good:
            print(f"[roll]   k={k:2d} -> recovered {shift:2d}  MISMATCH  peak={ss[-1]:.4f}")
    acc = nok / (n_sectors - 1)
    print(f"[roll] {nok}/{n_sectors-1} shifts recovered exactly  acc={acc:.2f}  "
          f"mean margin={np.mean(margins):+.4f}")
    print(f"[roll] {'PASS' if nok == n_sectors - 1 else 'FAIL'}")
    return nok == n_sectors - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/dinov2/dinov2_vits14_reg_448x224.onnx")
    ap.add_argument("--pano", default=None, help="panorama for the roll test")
    ap.add_argument("--sectors", type=int, default=16)
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()

    model, n_reg = build_model()
    check_token_layout(model, n_reg)
    bake_pos_embed(model)
    ok_bake = verify_bake(model, n_reg)
    if not a.skip_export:
        export(model, a.out)
    ok = ok_bake and verify_parity(model, a.out)
    if a.pano:
        ok &= roll_test(a.out, a.pano, a.sectors)
    print(f"\n=== {'ALL CHECKS PASS' if ok else 'CHECKS FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
