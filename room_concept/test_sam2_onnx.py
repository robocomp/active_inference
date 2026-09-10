#!/usr/bin/env python3
"""Tiny SAM2 ONNX test: encode an image once, prompt with a single point, save the mask.

Two-stage pipeline (matches the samexporter export):
    encoder.onnx : image  -> image_embed + 2 high-res feature maps   (run once / frame)
    decoder.onnx : those + point prompt -> low-res mask logits + IoU (run per prompt)

Robustness: encoder outputs are matched by shape (embed=256ch, hi-res=32ch & 64ch),
decoder inputs by name-alias + auto dtype cast, so it survives naming differences
between SAM2 exporters. Both graph signatures are printed on startup.

Usage:
    python test_sam2_onnx.py --image frame.jpg --x 640 --y 400
    # (omit --x/--y to prompt the image centre)
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw

ENC_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# decoder input name -> accepted aliases (matched case-insensitively, both directions)
ALIASES = {
    "image_embed":     ["image_embed", "image_embeddings", "embeddings"],
    "high_res_feats_0": ["high_res_feats_0", "high_res_features_0", "high_res_feat_0"],
    "high_res_feats_1": ["high_res_feats_1", "high_res_features_1", "high_res_feat_1"],
    "point_coords":    ["point_coords", "point_coordinates", "coords"],
    "point_labels":    ["point_labels", "labels"],
    "mask_input":      ["mask_input", "input_mask"],
    "has_mask_input":  ["has_mask_input", "has_mask"],
    "orig_im_size":    ["orig_im_size", "orig_image_size", "original_size"],
}

ORT_TO_NP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": bool,
}


def preprocess(img, size=ENC_SIZE):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)  # (1,3,H,W)


def print_sig(tag, sess):
    print(f"\n[{tag}] inputs:")
    for i in sess.get_inputs():
        print(f"    {i.name:<18} {i.type:<16} {i.shape}")
    print(f"[{tag}] outputs:")
    for o in sess.get_outputs():
        print(f"    {o.name:<18} {o.type:<16} {o.shape}")


def map_encoder_outputs(arrays):
    """Identify the 3 encoder outputs by channel count (stable across SAM2 sizes)."""
    out = {}
    for a in arrays:
        if a.ndim == 4:
            c = a.shape[1]
            if c == 256:
                out["image_embed"] = a
            elif c == 32:
                out["high_res_feats_0"] = a
            elif c == 64:
                out["high_res_feats_1"] = a
    missing = {"image_embed", "high_res_feats_0", "high_res_feats_1"} - set(out)
    if missing:
        raise RuntimeError(f"could not identify encoder outputs {missing}; "
                           f"got shapes {[a.shape for a in arrays]}")
    return out


def resolve_key(input_name):
    n = input_name.lower()
    # exact match first (handles the mask_input / has_mask_input collision correctly)
    for key, al in ALIASES.items():
        if any(a == n for a in al):
            return key
    # fall back to the most-specific (longest) substring match
    best_key, best_len = None, 0
    for key, al in ALIASES.items():
        for a in al:
            if (a in n or n in a) and len(a) > best_len:
                best_key, best_len = key, len(a)
    return best_key


def build_decoder_feed(dec_sess, enc, point_xy, orig_wh):
    W0, H0 = orig_wh
    px, py = point_xy
    # point coords live in the 1024x1024 encoder space
    coords = np.array([[[px / W0 * ENC_SIZE, py / H0 * ENC_SIZE]]], np.float32)  # (1,1,2)
    labels = np.array([[1]], np.float32)                                        # 1 = positive
    canonical = {
        "image_embed": enc["image_embed"],
        "high_res_feats_0": enc["high_res_feats_0"],
        "high_res_feats_1": enc["high_res_feats_1"],
        "point_coords": coords,
        "point_labels": labels,
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.array([0], np.float32),
        "orig_im_size": np.array([H0, W0], np.int64),
    }
    feed, unmatched = {}, []
    for inp in dec_sess.get_inputs():
        key = resolve_key(inp.name)
        if key is None or key not in canonical:
            unmatched.append(inp.name)
            continue
        val = canonical[key]
        want = ORT_TO_NP.get(inp.type)
        if want is not None and val.dtype != want:
            val = val.astype(want)
        feed[inp.name] = val
    if unmatched:
        print(f"  ! decoder inputs not auto-filled (add an alias if needed): {unmatched}")
    return feed


def postprocess(masks, ious, orig_wh):
    W0, H0 = orig_wh
    masks = np.asarray(masks)
    ious = np.asarray(ious).reshape(-1)
    if masks.ndim == 4:          # (1, M, 256, 256)
        masks = masks[0]
    best = int(np.argmax(ious)) if ious.size else 0
    logits = masks[best]         # (256, 256) float logits
    m = Image.fromarray(logits).resize((W0, H0), Image.BILINEAR)
    mask = np.asarray(m) > 0.0   # threshold logits at 0
    return mask, ious, best


def visualize(orig, mask, point_xy, out_path):
    base = orig.convert("RGB")
    overlay = np.asarray(base).copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
    res = Image.fromarray(overlay)
    d = ImageDraw.Draw(res)
    x, y = point_xy
    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 255, 0), outline=(0, 0, 0))
    res.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--encoder", default="sam2.1_hiera_tiny.encoder.onnx")
    ap.add_argument("--decoder", default="sam2.1_hiera_tiny.decoder.onnx")
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--out", default="sam2_mask.png")
    args = ap.parse_args()

    import onnxruntime as ort  # lazy: keeps helpers importable/testable without ORT

    orig = Image.open(args.image)
    W0, H0 = orig.size
    px = args.x if args.x is not None else W0 / 2
    py = args.y if args.y is not None else H0 / 2

    enc_sess = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])
    print_sig("encoder", enc_sess)
    print_sig("decoder", dec_sess)

    # --- encode once ---
    x = preprocess(orig)
    enc_names = [o.name for o in enc_sess.get_outputs()]
    enc_arrays = enc_sess.run(None, {enc_sess.get_inputs()[0].name: x})
    enc = map_encoder_outputs(enc_arrays)

    # --- decode for one point ---
    feed = build_decoder_feed(dec_sess, enc, (px, py), (W0, H0))
    dec_out = dec_sess.run(None, feed)

    # find masks tensor (4D, 256 spatial) and iou tensor among outputs
    masks = next(a for a in dec_out if np.ndim(a) == 4)
    ious = next((a for a in dec_out if np.ndim(a) <= 2), np.array([1.0]))
    mask, ious, best = postprocess(masks, ious, (W0, H0))

    print(f"\npoint = ({px:.0f}, {py:.0f})  |  IoU scores = "
          f"{np.round(ious, 3).tolist()}  |  chose mask #{best}")
    print(f"mask covers {100.0 * mask.mean():.1f}% of the frame")
    visualize(orig, mask, (px, py), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
