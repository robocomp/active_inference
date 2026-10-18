# View Atlas — Phase 2a Implementation Plan (module fixes before any live run)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clear the VIEW_ATLAS.md §9 blockers in `common/view_atlas/` — stride-aware crops for retina's decimated masks (with a golden crop test), the evidence term and pixel-count-dependent noise in the likelihood, repeat checks across bin edges, an atomic archive save — and re-measure the four phase-1 criteria offline.

**Architecture:** All changes stay inside the module and its tools; no agent or retina code changes. `view_crop.h` learns the mask's sampling stride from the pixel coordinates and rasterises each sample as a stride × stride block. `view_atlas.h` adds prototype uncertainty `D·Σ_n·D` (D = diag(1/√(N_s+n0), 1/√(n_c+n0))) to the own-hypothesis covariance inside the existing blend, and checks repeats in the neighbouring bins. `fit_constants.py` fits `Σ_n`, `Σ_own` and `quality_px_scale` from residuals binned by mask size. `atlas_io.h` writes temp files and renames. The replay harness uses a uniform prior over {archived candidates, new} and is re-run.

**Tech Stack:** C++23, Eigen 3, OpenCV 4, ONNX Runtime 1.26 CPU, Python 3 (numpy, scipy, opencv, ultralytics YOLOE, onnxruntime).

**Spec:** `common/view_atlas/VIEW_ATLAS.md` (§3, §8 phase-1 results, §9 blockers). Phase 1 plan: `docs/superpowers/plans/2026-10-17-view-atlas-phase1.md`.

## Global Constraints

- C++23; `and`/`or`/`not`, never `&&`/`||`/`!` in C++.
- Numbers parsed with `std::from_chars` only; data files written via `imbue(std::locale::classic())` + `std::setprecision(std::numeric_limits<float>::max_digits10)`; file-parsing tests call `setlocale(LC_ALL, "es_ES.UTF-8")` (fallback `""`) first.
- No thresholds/gates/clamps in model code; flagged constants only (after this plan: `n0`, `lambda_nats`, `max_rate_hz`, `pi_move`). Index-safety clamps are not model thresholds.
- No `-march=native`; builds with the existing g++ commands / `tools/build_tools.sh`; `make -j8` at most.
- Retina's mask sampling (retina/src/graph_publisher.cpp:509-539): `for (row = min_y; row < max_y; row += s) for (col = min_x; col < max_x; col += s)`, s = `VOXEL_DECIMATION_FACTOR` (default 2), anchored at the YOLO bbox corner (min_x, min_y), keeping samples whose binary mask is non-zero.
- The four phase-1 criteria (VIEW_ATLAS.md §6/§8: A swaps < 1 %; B moved and stayed chair ≥ 95 %, other chair ≤ 2 %; C ≤ 60 encodings/object-min — ungradeable on this data; D ±0.1 per posterior band with ≥ 20 decisions) are NOT changed. A FAIL is a result.
- Inputs live in the main checkout, untracked: `retina/etc/view_atlas_dataset/`, `retina/etc/zed_orbits/`, `retina/models/dinov3/dinov3_vits16_crop224.onnx` (+ `.onnx.data`), `retina/models/yolo26/_yoloe_work/`. In a worktree, SYMLINK them (never copy, never commit).
- Commit only named files with explicit `git add` paths.

## Review Focus

1. A mask published with stride 2 whose bbox corner is odd (sampling grid offset from the mask's own extent) → the crop must still match the dense crop within the golden tolerances (test in Task 2, both anchor offsets).
2. A one-row or one-column mask, a single pixel, and a mask whose samples share no common step (irregular, e.g. a dense mask with holes) → stride inference returns a stride that reproduces the input, never 0 (tests in Task 1).
3. A stride-2 sample on the last image row/column → its block is clipped to the image and `trunc_frac` still counts it as border (test in Task 1).
4. An atlas with a single view vs twenty distinct views at the same bearing → the single-view hypothesis must be wider (less confident), and an empty atlas still gives Λ exactly 0 (tests in Task 3).
5. A save interrupted after the blob but before the index → the previous archive on disk stays loadable (test in Task 5).

---

### Task 1: Stride-aware crop

**Files:**
- Modify: `common/view_atlas/view_crop.h` (make_crop raster loop; add `infer_stride`)
- Modify: `common/view_atlas/view_crop_test.cpp`

**Interfaces:**
- Produces: `int rc::atlas::infer_stride(const std::vector<Eigen::Vector2f>& mask_px)` — the gcd of all integer offsets (x − x_min) and (y − y_min); 1 when that gcd is 0 (single pixel) or the input is empty.
- Changes: `make_crop(bgr, mask_px)` keeps its signature; it now rasterises each sample (x, y) as the block [x, x+s) × [y, y+s) clipped to the image, s = `infer_stride(mask_px)`. `Crop::mask_px` counts filled (dense-equivalent) pixels; `trunc_frac` = filled pixels on the image border / filled pixels; the mask bbox is the filled extent.

- [ ] **Step 1: Write the failing tests** — append to `view_crop_test.cpp` before the summary line in `main()`:

```cpp
    std::printf("stride inference\n");
    check(infer_stride(rect_px(10, 10, 20, 20)) == 1, "dense mask: stride 1");
    auto dec = [](const std::vector<Eigen::Vector2f>& v, int s, int ax, int ay) {
        std::vector<Eigen::Vector2f> o;
        for (const auto& p : v)
            if ((int(p.x()) - ax) % s == 0 and (int(p.y()) - ay) % s == 0) o.push_back(p);
        return o;
    };
    check(infer_stride(dec(rect_px(600, 300, 700, 400), 2, 600, 300)) == 2, "decimated by 2: stride 2");
    check(infer_stride(dec(rect_px(600, 300, 700, 400), 3, 600, 300)) == 3, "decimated by 3: stride 3");
    check(infer_stride({Eigen::Vector2f(5.f, 5.f)}) == 1, "single pixel: stride 1");
    check(infer_stride({}) == 1, "empty: stride 1");
    check(infer_stride(dec(rect_px(600, 300, 700, 301), 2, 600, 300)) == 2, "one row: stride from x alone");
    {
        auto holes = rect_px(10, 10, 30, 30);
        holes.erase(holes.begin() + 5);                         // a dense mask with a hole stays stride 1
        check(infer_stride(holes) == 1, "dense mask with a hole: stride 1");
    }

    std::printf("stride-aware crop\n");
    auto dense = make_crop(img, rect_px(600, 300, 700, 400));
    auto sparse = make_crop(img, dec(rect_px(600, 300, 700, 400), 2, 600, 300));
    check(sparse.has_value() and dense.has_value(), "decimated mask yields a crop");
    if (sparse and dense)
    {
        check(sparse->mask_px == dense->mask_px, "decimated mask counts dense-equivalent pixels");
        cv::Mat diff;
        cv::absdiff(sparse->rgb224, dense->rgb224, diff);
        check(cv::countNonZero(diff.reshape(1)) == 0, "grid-aligned decimated rectangle: crop identical to dense");
    }
    auto edge = make_crop(img, dec(rect_px(1200, 300, 1280, 400), 2, 1200, 300));
    check(edge.has_value() and edge->trunc_frac > 0.f, "decimated mask at the right border still reports truncation");
```

- [ ] **Step 2: Run to verify failure**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3 opencv4) view_crop_test.cpp -o /tmp/view_crop_test $(pkg-config --libs opencv4)`
Expected: compile error, `infer_stride` not declared.

- [ ] **Step 3: Implement** — in `view_crop.h` add `#include <numeric>` and, before `make_crop`:

```cpp
/// Sampling stride of a published mask: retina keeps every s-th pixel on a grid anchored at the YOLO bbox
/// corner (retina/src/graph_publisher.cpp). Every sample's offset from the minimum sample is then a
/// multiple of s, so the gcd of all offsets recovers s; a dense mask (offsets 0,1,2,...) gives 1.
inline int infer_stride(const std::vector<Eigen::Vector2f>& mask_px)
{
    if (mask_px.empty()) return 1;
    long xmin = std::lround(mask_px[0].x()), ymin = std::lround(mask_px[0].y());
    for (const auto& p : mask_px) { xmin = std::min(xmin, std::lround(p.x())); ymin = std::min(ymin, std::lround(p.y())); }
    long g = 0;
    for (const auto& p : mask_px)
    {
        g = std::gcd(g, std::lround(p.x()) - xmin);
        g = std::gcd(g, std::lround(p.y()) - ymin);
        if (g == 1) break;
    }
    return g == 0 ? 1 : int(g);
}
```

In `make_crop`, replace the raster loop body so each sample fills its block (keep the CONTRACT comment):

```cpp
    const int s = infer_stride(mask_px);
    for (const auto& p : mask_px)
    {
        // CONTRACT: (comment unchanged)
        const int px = int(std::lround(p.x())), py = int(std::lround(p.y()));
        for (int y = py; y < py + s; ++y)
            for (int x = px; x < px + s; ++x)
            {
                if (x < 0 or y < 0 or x >= bgr.cols or y >= bgr.rows) continue;
                if (mask.at<std::uint8_t>(y, x) == 0)
                {
                    ++n;
                    if (x == 0 or y == 0 or x == bgr.cols - 1 or y == bgr.rows - 1) ++border;
                }
                mask.at<std::uint8_t>(y, x) = 255;
                x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
            }
    }
```

Update the make_crop doc comment: "Decimated masks (retina's VOXEL_DECIMATION_FACTOR) are rasterised as stride × stride blocks, which reproduces the dense mask up to a one-block rim; Task 2's golden test bounds the effect."

- [ ] **Step 4: Run tests**

Run: same g++ command, then `/tmp/view_crop_test`. Expected: all `ok`, `view_crop_test: PASS` (the existing dense tests must still pass: stride 1 is the old behaviour).

- [ ] **Step 5: Commit**

```bash
git add common/view_atlas/view_crop.h common/view_atlas/view_crop_test.cpp
git commit -m "view_atlas: stride-aware crop — infer the mask sampling stride, rasterise samples as blocks (+tests)"
```

---

### Task 2: Golden crop test (Python mirror parity + decimated-vs-dense)

**Files:**
- Create: `common/view_atlas/tools/export_crop_golden.py`
- Create: `common/view_atlas/tools/crop_golden.cpp`
- Modify: `common/view_atlas/tools/build_tools.sh` (build crop_golden)
- Create (generated, committed, small): `common/view_atlas/tools/crop_golden/` — `masks.txt`, `mask_XX.png`, `py_crop_XX.png`, `py_colour.f32`

**Interfaces:**
- Consumes: `make_crop`, `colour_hist` (Task 1), `ViewEncoder` (phase 1), `build_dataset.py::make_crop` / `colour_hist` (Python mirror).
- Produces: `bin/crop_golden <fixture_dir> <frames_dir> <model.onnx>` exiting 0 iff every check passes.

Pass criteria (declared): (i) dense C++ crop vs Python crop: max absolute pixel difference ≤ 1, colour-hist cosine ≥ 0.9999, on every mask; (ii) decimated (stride 2, grid anchored at the mask's bbox corner AND at corner + (1,1)) vs dense C++: square-crop mask IoU ≥ 0.98, DINO cosine ≥ 0.99, colour cosine ≥ 0.99, worst case over all masks.

- [ ] **Step 1: Write the fixture exporter** — `export_crop_golden.py`:

```python
#!/usr/bin/env python3
"""Golden crop fixture: 20 real YOLOE masks from the orbit recording, with the PYTHON mirror's crop and
colour histogram (build_dataset.py::make_crop / colour_hist). crop_golden.cpp checks the C++ crop against
them and checks decimated masks (retina's stride-2 sampling) against dense ones.
usage (from active_inference/): python3 common/view_atlas/tools/export_crop_golden.py"""
import importlib.util, os, sys
sys.modules["torchaudio"] = None
import numpy as np, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
spec = importlib.util.spec_from_file_location("bd", os.path.join(ROOT, "common/view_atlas/tools/build_dataset.py"))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
spec = importlib.util.spec_from_file_location("exp", os.path.join(ROOT, "retina/tools/export_yoloe_household.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

def main():
    frames = os.path.join(ROOT, "retina/etc/zed_orbits"); out = os.path.join(ROOT, "common/view_atlas/tools/crop_golden")
    os.makedirs(out, exist_ok=True)
    from ultralytics import YOLOE
    cwd = os.getcwd(); os.chdir(os.path.join(ROOT, "retina/models/yolo26/_yoloe_work"))
    yolo = YOLOE("yoloe-26x-seg.pt"); yolo.set_classes(exp.VOCAB); os.chdir(cwd)
    files = sorted(f for f in os.listdir(frames) if f.endswith(".jpg") and "_depth" not in f)[::40]
    rows, cols = [], []
    for f in files:
        img = cv2.imread(os.path.join(frames, f))
        r = yolo.predict(img, conf=0.25, verbose=False, retina_masks=True)[0]
        if r.masks is None: continue
        for m in r.masks.data.cpu().numpy() > 0.5:
            if m.sum() < 2000 or len(rows) >= 20: continue
            k = len(rows)
            rgb, box, mbox, npx, trunc = bd.make_crop(img, m)
            cv2.imwrite(os.path.join(out, f"mask_{k:02d}.png"), m.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(out, f"py_crop_{k:02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cols.append(bd.colour_hist(box, mbox).astype("<f4")); rows.append(f)
        if len(rows) >= 20: break
    with open(os.path.join(out, "masks.txt"), "w") as fh:
        for k, f in enumerate(rows): fh.write(f"{k} {f}\n")
    np.stack(cols).tofile(os.path.join(out, "py_colour.f32"))
    print("golden masks:", len(rows))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python3 common/view_atlas/tools/export_crop_golden.py` (from active_inference/). Expected: `golden masks: 20`.

- [ ] **Step 3: Write the C++ checker** — `crop_golden.cpp`:

```cpp
// crop_golden — (i) C++ dense crop == Python mirror crop; (ii) stride-2 decimated crop ~ dense crop.
// usage: crop_golden <fixture_dir> <frames_dir> <model.onnx>
#include "../view_crop.h"
#include "../view_encoder.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

using namespace rc::atlas;

static std::vector<Eigen::Vector2f> pixels(const cv::Mat& m, int s, int ax, int ay)
{
    std::vector<Eigen::Vector2f> v;
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x)
            if (m.at<std::uint8_t>(y, x) and (x - ax) % s == 0 and (y - ay) % s == 0 and x >= ax and y >= ay)
                v.emplace_back(float(x), float(y));
    return v;
}
static float iou(const cv::Mat& a, const cv::Mat& b)
{
    if (a.size() != b.size()) return 0.f;
    const int i = cv::countNonZero(cv::Mat(a & b)), u = cv::countNonZero(cv::Mat(a | b));
    return u ? float(i) / float(u) : 1.f;
}
// The blanked 224² crop encodes the mask: background pixels equal the ImageNet-mean fill exactly.
static cv::Mat crop_mask(const cv::Mat& rgb224)
{
    cv::Mat m;
    cv::inRange(rgb224, cv::Scalar(124, 116, 104), cv::Scalar(124, 116, 104), m);
    return 255 - m;
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: %s fixture_dir frames_dir model.onnx\n", argv[0]); return 2; }
    const std::string fx = argv[1], fr = argv[2];
    EncoderConfig cfg; cfg.model_path = argv[3];
    ViewEncoder enc(cfg);
    if (not enc.ok()) return 2;
    std::ifstream list(fx + "/masks.txt");
    std::ifstream col(fx + "/py_colour.f32", std::ios::binary | std::ios::ate);
    const auto bytes = std::size_t(col.tellg()); col.seekg(0);
    std::vector<float> pyc(bytes / sizeof(float));
    col.read(reinterpret_cast<char*>(pyc.data()), std::streamsize(bytes));
    int k; std::string frame; bool ok = true;
    float worst_px = 0, worst_pyc = 1, worst_iou = 1, worst_dino = 1, worst_col = 1;
    while (list >> k >> frame)
    {
        char name[64];
        const cv::Mat img = cv::imread(fr + "/" + frame);
        std::snprintf(name, sizeof name, "/mask_%02d.png", k);
        const cv::Mat m = cv::imread(fx + name, cv::IMREAD_GRAYSCALE);
        std::snprintf(name, sizeof name, "/py_crop_%02d.png", k);
        cv::Mat py = cv::imread(fx + name), py_rgb;
        if (img.empty() or m.empty() or py.empty()) { std::printf("mask %d: missing input\n", k); return 1; }
        cv::cvtColor(py, py_rgb, cv::COLOR_BGR2RGB);
        const auto d = make_crop(img, pixels(m, 1, 0, 0));
        if (not d) { std::printf("mask %d: dense crop failed\n", k); return 1; }
        cv::Mat diff; cv::absdiff(d->rgb224, py_rgb, diff);
        double mx; cv::minMaxLoc(diff.reshape(1), nullptr, &mx);
        const Eigen::VectorXf hc = colour_hist(d->bgr_box, d->mask_box);
        const Eigen::Map<const Eigen::VectorXf> hp(pyc.data() + std::size_t(k) * 64, 64);
        worst_px = std::max(worst_px, float(mx)); worst_pyc = std::min(worst_pyc, hc.dot(hp));
        const auto ed = enc.encode(d->rgb224);
        cv::Rect bb = cv::boundingRect(m);
        for (int off = 0; off <= 1; ++off)
        {
            const auto s = make_crop(img, pixels(m, 2, bb.x + off, bb.y + off));
            if (not s or not ed) { std::printf("mask %d: decimated crop failed\n", k); return 1; }
            const auto es = enc.encode(s->rgb224);
            worst_iou = std::min(worst_iou, iou(crop_mask(s->rgb224), crop_mask(d->rgb224)));
            worst_dino = std::min(worst_dino, es->dot(*ed));
            worst_col = std::min(worst_col, colour_hist(s->bgr_box, s->mask_box).dot(hc));
        }
    }
    const bool i_ok = worst_px <= 1.f and worst_pyc >= 0.9999f;
    const bool ii_ok = worst_iou >= 0.98f and worst_dino >= 0.99f and worst_col >= 0.99f;
    std::printf("(i)  C++ vs Python dense: worst pixel diff %.0f, worst colour cosine %.6f -> %s\n",
                worst_px, worst_pyc, i_ok ? "PASS" : "FAIL");
    std::printf("(ii) stride 2 vs dense: worst mask IoU %.4f, worst DINO cosine %.4f, worst colour cosine %.4f -> %s\n",
                worst_iou, worst_dino, worst_col, ii_ok ? "PASS" : "FAIL");
    ok = i_ok and ii_ok;
    return ok ? 0 : 1;
}
```

In `tools/build_tools.sh`, after the encoder_parity line, add:

```bash
g++ $CFLAGS crop_golden.cpp ../view_encoder.cpp -o bin/crop_golden $LIBS
```

- [ ] **Step 4: Build and run**

Run: `bash common/view_atlas/tools/build_tools.sh && common/view_atlas/tools/bin/crop_golden common/view_atlas/tools/crop_golden retina/etc/zed_orbits retina/models/dinov3/dinov3_vits16_crop224.onnx`
Expected: two lines ending `PASS`, exit 0. A FAIL on (ii) is a RESULT about the stride approximation: report the worst values and which mask, do not loosen the tolerances. A FAIL on (i) means C++ and Python crops disagree: report the first mask and pixel location (the phase-1 constants depend on this mirror).

- [ ] **Step 5: Commit**

```bash
git add common/view_atlas/tools/export_crop_golden.py common/view_atlas/tools/crop_golden.cpp \
        common/view_atlas/tools/build_tools.sh common/view_atlas/tools/crop_golden
git commit -m "view_atlas: golden crop test — C++ vs Python mirror, stride-2 vs dense (fixture + checker)"
```

---

### Task 3: Prototype uncertainty in the own hypothesis

**Files:**
- Modify: `common/view_atlas/view_atlas.h` (`ViewAtlas::hypothesis`)
- Modify: `common/view_atlas/view_atlas_test.cpp`

**Interfaces:**
- Changes: `hypothesis(bearing, quality)` keeps its signature. Inside, `Lo` is the Cholesky factor of `cov_own + D·cov_n·D` with `D = diag(1/sqrt(Ns + n0), 1/sqrt(n_c + n0))` instead of `cov_own`. The blend, the bracket form and the exact-zero property are unchanged (at R = 0 the result does not depend on `Lo`).

- [ ] **Step 1: Write the failing test** — add to `view_atlas_test.cpp` and call it from `main()`:

```cpp
static void test_prototype_uncertainty()
{
    std::printf("prototype uncertainty\n");
    // cov_bg == cov_own isolates the evidence term: the blend then cannot change the covariance, so only
    // D·Σn·D can make one view less certain than twenty.
    AtlasParams P; P.cov_bg = P.cov_own;
    ViewAtlas one(P, 4, 3), many(P, 4, 3);
    one.update(obs(0, 0, 0.f));
    for (int i = 0; i < 20; ++i) many.update(obs(0, 0, -0.5f + 0.05f * float(i), 1.0f + 0.3f * float(i)));
    const Gauss2 h1 = one.hypothesis(0.f, 1.f), h20 = many.hypothesis(0.f, 1.f);
    check(h1.cov(1, 1) > h20.cov(1, 1), "colour: one view is less certain than twenty");
    check(h1.cov(0, 0) > h20.cov(0, 0), "shape: one view is less certain than twenty");
    ViewAtlas empty(AtlasParams{}, 4, 3);
    check(empty.log_ratio(obs(0, 0, 0.f), 0.f) == 0.f, "empty atlas: still exactly 0 with the evidence term");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o /tmp/view_atlas_test && /tmp/view_atlas_test`
Expected: both "less certain" checks FAIL (with cov_bg == cov_own and no evidence term, both hypotheses have covariance cov_own + quality term exactly), summary `FAIL`.

- [ ] **Step 3: Implement** — in `hypothesis()`, replace the `Lo` line:

```cpp
        // Prototype uncertainty (spec §3.2, Σ_own + Σ_n/N): a bin mean built from few views is itself noisy.
        // D scales the per-view noise by the evidence behind each channel; n0 keeps it finite at N = 0, where
        // R = 0 removes Lo from the blend anyway (so the exact-zero property is untouched).
        const Eigen::Matrix2f D = Eigen::Vector2f(1.f / std::sqrt(Ns + p_.n0), 1.f / std::sqrt(s_.n_c + p_.n0)).asDiagonal();
        const Eigen::Matrix2f Lo = Eigen::LLT<Eigen::Matrix2f>(p_.cov_own + D * p_.cov_n * D).matrixL();
```

- [ ] **Step 4: Run tests**

Run: same command. Expected: `view_atlas_test: PASS`, including every phase-1 check (exact zero, off-diagonal cov_bg, unseen side, shift marginal, window).

- [ ] **Step 5: Commit**

```bash
git add common/view_atlas/view_atlas.h common/view_atlas/view_atlas_test.cpp
git commit -m "view_atlas: prototype uncertainty D·Σn·D in the own hypothesis (spec §3.2) (+test)"
```

---

### Task 4: Fit per-view noise against mask size

**Files:**
- Modify: `common/view_atlas/tools/fit_constants.py` (`pairs`, new `fit_noise`, `fit`, `selftest`, `main`)
- Regenerate (commit): `common/view_atlas/view_atlas_constants.txt`

**Interfaces:**
- Consumes: views.csv column `mask_px`; `load_params` keys `cov_own_*`, `cov_n_*`, `quality_px_scale` (already accepted).
- Produces: constants with `cov_own_*` = large-mask (q → 1) own covariance, `cov_n_*` and `quality_px_scale` fitted, header line `# noise model: cov(px) = cov_own + cov_n (1-q)/q, q = 1-exp(-px/quality_px_scale); bands=<n>, sse=<v>`.

Model (matches `quality_cov` in view_atlas.h with trunc_frac = 0, visible = 1): for own pairs at gap < 15°, residual covariance of (s_d, s_c) for queries in mask-size band b is `C(b) = cov_own + cov_n·(1 − q_b)/q_b`, `q_b = 1 − exp(−px_b / quality_px_scale)`, px_b = band median. For each candidate `quality_px_scale` on a log grid (64 values, 100…50 000 px), `cov_own` and `cov_n` follow by elementwise linear least squares over the bands; the scale with the lowest summed squared error wins.

- [ ] **Step 1: Extend `pairs` to return the query's mask size** — change the own append and the return:

```python
            own.append((abs(wrap(v.bearing.values[q] - bb)), S[q] @ ps, C[q] @ pc, v.mask_px.values[q]))
    ...
    return np.array(own), np.array(bg)
```

and in `fit`, use only columns 0–2 where it previously used `own` (e.g. `own[:, 0]`, `own[:, 1]`, `near[:, 1:3]`).

- [ ] **Step 2: Add `fit_noise`**:

```python
def fit_noise(near, n_bands=6):
    """near: own pairs at gap < 15 deg, columns (D, s_d, s_c, px). Returns (cov_own, cov_n, px_scale, sse)."""
    edges = np.quantile(near[:, 3], np.linspace(0, 1, n_bands + 1))
    bands = [near[(near[:, 3] >= lo) & (near[:, 3] <= hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    bands = [b for b in bands if len(b) >= 30]
    px = np.array([np.median(b[:, 3]) for b in bands])
    C = np.array([np.cov(b[:, 1:3].T) for b in bands])            # (B, 2, 2)
    best = None
    for scale in np.geomspace(100, 50000, 64):
        q = 1 - np.exp(-px / scale); x = (1 - q) / q                 # C(b) = A + x_b * N, elementwise
        X = np.c_[np.ones_like(x), x]
        coef, res, *_ = np.linalg.lstsq(X, C.reshape(len(px), 4), rcond=None)
        sse = float(((X @ coef - C.reshape(len(px), 4)) ** 2).sum())
        if best is None or sse < best[3]:
            best = (coef[0].reshape(2, 2), coef[1].reshape(2, 2), float(scale), sse)
    return best + (len(bands),)
```

- [ ] **Step 3: Use it in `fit` and write the header** — in `fit`, after `near` is computed:

```python
    cov_own_fit, cov_n_fit, px_scale, sse, nb = fit_noise(near)
```

then emit `cov_own_*` from `cov_own_fit`, `cov_n_*` from `cov_n_fit`, `quality_px_scale=px_scale`, and keep `sd = math.sqrt(cov_own_fit[0, 0])` for the repeat derivation. Return `_noise_header=f"bands={nb}, sse={sse:.3g}"` alongside `_repeat_ratio`, pop it in `main` and write the header line given in Interfaces. If `cov_own_fit` or `cov_n_fit` is not positive definite (np.linalg.eigvalsh ≤ 0), print `WARNING: fitted <name> not positive definite: <eigenvalues>` — `load_params` will then refuse the file, which is the correct, loud outcome; do NOT repair the matrix.

- [ ] **Step 4: Extend the self-test** — in `selftest`, add synthetic own pairs with a known noise model (px drawn log-uniform 300…30 000, scale 2000, cov_own diag(4e-4, 4e-4), cov_n diag(2e-3, 2e-3)), sample (s_d, s_c) with that covariance, call `fit_noise`, and require the recovered scale within a factor 1.5 of 2000 and `cov_n` diagonals within 30 %:

```python
    pxs = np.exp(rng.uniform(np.log(300), np.log(30000), 6000)); qq = 1 - np.exp(-pxs / 2000.0)
    var = 4e-4 + 2e-3 * (1 - qq) / qq
    syn = np.c_[np.zeros(6000), 0.95 + rng.normal(0, 1, 6000) * np.sqrt(var), 0.97 + rng.normal(0, 1, 6000) * np.sqrt(var), pxs]
    a_, n_, s_, _, _ = fit_noise(syn)
    ok = ok and (2000 / 1.5 < s_ < 2000 * 1.5) and abs(n_[0, 0] - 2e-3) / 2e-3 < 0.3 and abs(n_[1, 1] - 2e-3) / 2e-3 < 0.3
    print(f"selftest noise: scale {s_:.0f} (truth 2000), cov_n_dd {n_[0,0]:.2e} (truth 2.0e-03)")
```

- [ ] **Step 5: Run self-test, re-fit, load-check**

Run: `python3 common/view_atlas/tools/fit_constants.py --selftest` → both selftest lines and `PASS`.
Run: `python3 common/view_atlas/tools/fit_constants.py retina/etc/view_atlas_dataset common/view_atlas/view_atlas_constants.txt`
Run the phase-1 load check (`/tmp/load_check.cpp` from the phase-1 plan, Task 6 Step 5) → `loaded: ...`. Report the old → new values of cov_own, cov_n, quality_px_scale. Values outside expectations are findings, not things to tune.

- [ ] **Step 6: Commit**

```bash
git add common/view_atlas/tools/fit_constants.py common/view_atlas/view_atlas_constants.txt
git commit -m "view_atlas: fit per-view noise against mask size (cov_n, cov_own, quality_px_scale)"
```

---

### Task 5: Repeat check across bin edges; atomic archive save

**Files:**
- Modify: `common/view_atlas/view_atlas.h` (`repeat_similarity`, its caller in `update`)
- Modify: `common/view_atlas/atlas_io.h` (`save_archive`)
- Modify: `common/view_atlas/view_atlas_test.cpp`, `common/view_atlas/atlas_io_test.cpp`

**Interfaces:**
- Changes: `repeat_similarity(int bin_index, float b, float lr)` checks the poses of bins `bin_index − 1, bin_index, bin_index + 1` (indices wrapped modulo n_bins); `update` passes `bin_of(b)`.
- Changes: `save_archive` writes `base + ".blob.tmp"` and `base + ".idx.tmp"`, then renames blob, then index (`std::filesystem::rename`); on any failure before the renames, the previous `base.blob`/`base.idx` are untouched and the temp files are removed.

- [ ] **Step 1: Failing tests** — view_atlas_test.cpp:

```cpp
static void test_repeat_across_bin_edge()
{
    std::printf("repeat check across a bin edge\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    const float edge = 0.f;                                   // boundary between bins 5 and 6
    a.update(obs(0, 0, edge - 0.005f));
    float before = 0.f; for (const auto& b : a.state().bins) before += b.n;
    a.update(obs(0, 0, edge + 0.005f));                       // 0.6 deg away, other side of the edge
    float after = 0.f; for (const auto& b : a.state().bins) after += b.n;
    check(after - before < 0.01f, "a look 0.6 deg away across a bin edge is a repeat");
}
```

atlas_io_test.cpp (before the cleanup line):

```cpp
    std::printf("atomic save\n");
    check(save_archive(base, recs, &why), "baseline archive saved");
    std::filesystem::create_directories(base + ".idx.tmp");   // the index temp cannot be created as a file
    check(not save_archive(base, {make_record(42)}, &why), "save fails when the index temp cannot be written");
    std::vector<ArchiveRecord> still;
    check(load_archive(base, P, 4, 3, still, &why) and still.size() == recs.size(), "previous archive still loads");
    check(not std::filesystem::exists(base + ".blob.tmp"), "blob temp removed after the failed save");
    std::filesystem::remove_all(base + ".idx.tmp");
    check(save_archive(base, recs, &why) and not std::filesystem::exists(base + ".idx.tmp"), "normal save leaves no temp files");
```

- [ ] **Step 2: Run to verify failure** — both test binaries (commands as in their headers). Expected: the bin-edge check FAILs; the "previous archive still loads" check FAILs (today the blob is overwritten in place).

- [ ] **Step 3: Implement** — view_atlas.h:

```cpp
    [[nodiscard]] float repeat_similarity(int bin_index, float b, float lr) const
    {
        const Eigen::Vector2f view(b, lr);
        float k = 0.f;
        for (int d = -1; d <= 1; ++d)                             // a repeat can sit across a bin edge
        {
            const int j = (bin_index + d + p_.n_bins) % p_.n_bins;
            for (const auto& ps : s_.bins[std::size_t(j)].poses) k = std::max(k, pose_similarity(view, ps));
        }
        return k;
    }
```

and in `update`: `const float w = o.quality * (1.f - repeat_similarity(bin_of(b), b, lr));`.

atlas_io.h `save_archive`: add `#include <filesystem>`; open `base + ".blob.tmp"` instead of `base + ".blob"` and `base + ".idx.tmp"` instead of `base + ".idx"`; on every failure path after the blob temp was opened, `std::filesystem::remove(base + ".blob.tmp", ec)` (and the index temp) before returning; at the end:

```cpp
    f.close();
    if (not f) { std::error_code ec; std::filesystem::remove(base + ".blob.tmp", ec); std::filesystem::remove(base + ".idx.tmp", ec);
                 return fail("write failed: " + base + ".idx.tmp"); }
    std::error_code ec;
    std::filesystem::rename(base + ".blob.tmp", base + ".blob", ec);
    if (ec) return fail("rename failed: " + base + ".blob: " + ec.message());
    std::filesystem::rename(base + ".idx.tmp", base + ".idx", ec);
    if (ec) return fail("rename failed: " + base + ".idx: " + ec.message());
    return true;
```

(The loader's record-count check catches a crash between the two renames.)

- [ ] **Step 4: Run tests** — all three view_atlas test binaries + `bash common/run_tests.sh` (only the two pre-existing Qt mission tests may fail). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add common/view_atlas/view_atlas.h common/view_atlas/atlas_io.h common/view_atlas/view_atlas_test.cpp common/view_atlas/atlas_io_test.cpp
git commit -m "view_atlas: repeat check across bin edges; atomic archive save (temp + rename) (+tests)"
```

---

### Task 6: Re-measure offline and update the spec

**Files:**
- Modify: `common/view_atlas/tools/atlas_replay.cpp` (priors)
- Modify: `common/view_atlas/VIEW_ATLAS.md` (§3.2, §3.5, new §10)

**Interfaces:**
- Consumes: everything above; dataset and constants from Task 4.

- [ ] **Step 1: Uniform prior over hypotheses** — in atlas_replay.cpp replace the hand-set priors:

```cpp
    // Uniform over {archived candidates of the class, new}: the spec's class-birth-rate prior needs live
    // statistics (phase 2b shadow mode); until then no hypothesis is favoured a priori.
    const float log_prior_each = -std::log(float(chairs.size() + 1));
    const float log_prior_new = log_prior_each, log_prior_k = log_prior_each;
```

- [ ] **Step 2: Build and run**

Run: `bash common/view_atlas/tools/build_tools.sh && common/view_atlas/tools/bin/atlas_replay retina/etc/view_atlas_dataset common/view_atlas/view_atlas_constants.txt retina/etc/view_atlas_replay`
Record the COMPLETE stdout. Criteria unchanged; FAILs are results.

- [ ] **Step 3: Update the spec**
  - §3.2: replace the H_k covariance description with the implemented form — blend of Cholesky factors from (μ_bg, Σ_bg) toward (μ_own, Σ_own + D Σ_n D), D = diag(1/√(N_s+n0), 1/√(n_c+n0)), plus the view-quality term Σ_n (1−q)/q on both hypotheses; state that this is exactly 0 at no evidence for any constants. Effective gap: nearest populated bin (state it).
  - §3.5: `Σ_own`, `Σ_n`, `quality_px_scale` FITTED by `fit_noise` (remove `quality_px_scale` from the flagged list); hypothesis priors uniform until 2b measures the class birth rate.
  - §9: mark (a) stride, (b) covariance, (c) repeat edge, (d) atomic save as done, citing Task 2's golden numbers and Task 4's fitted values.
  - New §10 "Phase 2a results": a table of A–D (phase 1 corrected vs phase 2a), the gap-band table, the golden-crop worst values, the fitted noise constants (old → new), and one paragraph reading the change in D. State measured facts; assert no unmeasured cause.

- [ ] **Step 4: Commit**

```bash
git add common/view_atlas/tools/atlas_replay.cpp common/view_atlas/VIEW_ATLAS.md
git commit -m "view_atlas: phase 2a — uniform hypothesis prior, re-measured replay, spec §3/§9/§10"
```

---

## After phase 2a (separate plans)

- **2b — chair_concept shadow mode** (needs its own plan after exploring chair_concept: tracker birth path, RGB ingestor, threading): `ai_common::view_atlas` CMake target (ORT discovery as in retina/src/CMakeLists.txt); `[ViewAtlas] enabled=false`; FrameHistory fed from the RGB ingestor (first verify `MasksPacket::timestamp_ms == RgbIngestor::stamp_ms()` for the same frame); encoder worker thread with deep-copied crops; per-instance atlases; archive on death; at each birth candidate LOG the re-ID posterior next to what the tracker did, without acting; measure criterion C and the class birth rate live.
- **2c — exploration toward unseen sides**: expose `learning_value` per object to the controller/NBV as an epistemic target; re-run the moved-chair test on a tour that circles the chairs.
