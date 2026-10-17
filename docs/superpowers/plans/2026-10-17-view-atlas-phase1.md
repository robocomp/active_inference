# View Atlas — Phase 1 Implementation Plan (module, tests, encoder, constants, offline harness)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `common/view_atlas/` — per-object appearance atlas math, archive persistence, the on-demand DINOv3 crop encoder, fitted constants and an offline replay harness — so phase 2 can wire it into chair_concept in shadow mode with measured, not assumed, behaviour.

**Architecture:** A header-only, pure-Eigen `view_atlas.h` holds the generative model (12 azimuth bins of DINOv3 shape in the object frame, one colour prototype, angular-gap likelihood, yaw-shift marginalisation, re-ID posterior, value of encoding). `atlas_io.h` persists dead objects' atlases (text index + binary blob, locale-safe) and loads fitted constants. `frame_history.h` / `view_crop.h` (OpenCV) build the background-blanked crop and colour histogram; `view_encoder.{h,cpp}` runs DINOv3 ViT-S/16 through ONNX Runtime on the CPU. Python tools export the model, build a labelled views dataset from the two Webots recordings, and fit the constants; a C++ replay harness runs the module itself over that dataset.

**Tech Stack:** C++23, Eigen 3, OpenCV 4, ONNX Runtime (CPU EP, `/usr/local/onnxruntime`), Python 3 (torch, transformers, ultralytics YOLOE, onnxruntime, scipy, shapely, opencv).

**Spec:** `common/view_atlas/VIEW_ATLAS.md` (on main since 509825a8). Read it before starting; this plan argues from it.

## Global Constraints

- C++23; use `and`/`or`/`not`, never `&&`/`||`/`!` (repo rule, CLAUDE.md "Coding").
- Parse numbers with `std::from_chars` only; write data files through `imbue(std::locale::classic())` with `std::setprecision(std::numeric_limits<float>::max_digits10)`. Any test that parses a file calls `setlocale(LC_ALL, "es_ES.UTF-8")` (fallback `""`) first.
- Binary formats are written field by field, never `fwrite` of a struct.
- No thresholds/gates/clamps/σ-floors in model code. The only flagged constants are `lambda_nats` (price of one encoding), `max_rate_hz` (machine guard), `pi_move` (prior probability an archived object was moved), `n0` (prior pseudo-count, "worth one view"), `quality_px_scale`. Index-safety clamps (array bounds) are not model thresholds.
- No `-march=native`, no `-DEIGEN_MAX_ALIGN_BYTES=0` anywhere (libdsr ABI).
- Builds: `make -j8` at most; check `pgrep -a cc1plus` first (another session may be compiling).
- Encoder: DINOv3 **ViT-S/16** (`facebook/dinov3-vits16-pretrain-lvd1689m`), NOT ViT-S+; input 224×224 RGB, ImageNet mean/std; output = L2-normalised mean of the L2-normalised patch tokens (skip CLS + 4 registers), 384-d. ONNX Runtime **CPU** execution provider.
- Crop: square, side 1.15 × max(mask-extent w, h), centred on the mask extent, clipped to the image, background pixels set to ImageNet-mean BGR (104, 116, 124), resized to 224 with `INTER_AREA`, BGR→RGB.
- Colour: 16 (hue, [0,180)) × 4 (saturation, [0,256)) histogram over mask pixels inside the mask's bounding box, sqrt, L2-normalised.
- Atlas: 12 azimuth bins of 30° in the object frame; bearing = atan2 of (camera − object) in the world frame minus the object yaw, wrapped to (−π, π].
- Every `cv::Mat` crossing a thread boundary is deep-copied (`clone()`).
- No decoder (latent → image) is part of this design; do not add one.
- Do not touch retina's runtime code, the graph contract, or `common/appearance_belief`.

## Review Focus

1. Archive or constants file written in one locale and read under `es_ES.UTF-8` → must read back bit-exact, never truncated at the decimal point (test in Task 3).
2. A brand-new object with no views (empty atlas) is scored → log-ratio must be exactly 0, never NaN/inf, and the re-ID posterior must still sum to 1 (test in Task 2).
3. Bearings on either side of ±π (bin 11 ↔ bin 0) → treated as neighbours, not opposite sides (test in Task 1).
4. A truncated, wrong-magic, wrong-version or wrong-dimension archive file → load refuses with a reason naming the field, returns false, leaves the output empty, never crashes (test in Task 3).
5. A mask at the image border, a 1-pixel mask, mask pixels outside the image, or a frame stamp not in the history → crop/hist/lookup return "nothing" or a clipped crop, never throw (tests in Task 5).

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `common/view_atlas/view_atlas.h` | Params, atlas state, update, prediction, likelihood, yaw marginalisation, re-ID, value of encoding. Pure Eigen, header-only. | 1, 2 |
| `common/view_atlas/view_atlas_test.cpp` | Unit tests for `view_atlas.h` (picked up by `common/run_tests.sh`). | 1, 2 |
| `common/view_atlas/atlas_io.h` | `ArchiveRecord`, `save_archive`/`load_archive`, `load_params`. Header-only. | 3 |
| `common/view_atlas/atlas_io_test.cpp` | Round-trip, locale, corruption tests. | 3 |
| `retina/tools/export_dinov3_crop.py` | Export the ONNX encoder + write a parity fixture. | 4 |
| `common/view_atlas/frame_history.h` | Ring of recent BGR frames by stamp. Header-only, OpenCV. | 5 |
| `common/view_atlas/view_crop.h` | `make_crop`, `colour_hist`. Header-only, OpenCV. | 5 |
| `common/view_atlas/view_crop_test.cpp` | Crop/hist/history tests (OpenCV only). | 5 |
| `common/view_atlas/view_encoder.h`, `view_encoder.cpp` | ONNX Runtime CPU session: 224² RGB → 384-d. | 5 |
| `common/view_atlas/tools/encoder_parity.cpp`, `tools/build_tools.sh` | Parity check vs the Python fixture; builds the ORT-linked tools. | 5, 7 |
| `common/view_atlas/tools/build_dataset.py` | Labelled views dataset from `zed_frames` + `zed_orbits`. | 6 |
| `common/view_atlas/tools/fit_constants.py` | Fit the model constants; `--selftest`. | 6 |
| `common/view_atlas/view_atlas_constants.txt` | Fitted constants with provenance (generated, committed). | 6 |
| `common/view_atlas/tools/atlas_replay.cpp` | Offline replay through the module; metrics + pass/fail. | 7 |

---

### Task 1: Atlas state, update, prediction, value of learning

**Files:**
- Create: `common/view_atlas/view_atlas.h`
- Create: `common/view_atlas/view_atlas_test.cpp`
- Modify: `common/view_atlas/VIEW_ATLAS.md` (§7: add "no decoder")

**Interfaces:**
- Consumes: nothing.
- Produces (namespace `rc::atlas`):
  - `float wrap_pi(float a)`
  - `struct AtlasParams` (fields listed in the code below)
  - `struct Bin { Eigen::VectorXf mean; float n; float bearing; std::vector<Eigen::Vector2f> poses; std::uint64_t stamp_ms; }`
  - `struct AtlasState { int shape_dim; int colour_dim; std::vector<Bin> bins; Eigen::VectorXf colour; float n_c; }`
  - `struct ViewObs { Eigen::VectorXf shape, colour; float bearing_rad, range_m, quality; std::uint64_t stamp_ms; }`
  - `float view_quality(float trunc_frac, float visible_frac, int mask_px, float px_scale)`
  - `class ViewAtlas` with `ViewAtlas(const AtlasParams&, int shape_dim, int colour_dim)`, `static ViewAtlas from_state(const AtlasParams&, AtlasState)`, `const AtlasState& state() const`, `const AtlasParams& params() const`, `int bin_of(float) const`, `float kernel(float) const`, `float effective_count(float) const`, `float learning_value(float bearing, float quality) const`, `float gap(float) const`, `struct Prediction { Eigen::VectorXf shape; float gap_rad; bool has_shape; }`, `Prediction predict(float) const`, `void update(const ViewObs&)`.

- [ ] **Step 1: Write the failing test**

Create `common/view_atlas/view_atlas_test.cpp`:

```cpp
/*
 * view_atlas_test.cpp — unit tests for view_atlas.h. Standalone (common/run_tests.sh):
 *     g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o t && ./t
 */
#include "view_atlas.h"

#include <cmath>
#include <cstdio>
#include <numbers>

using namespace rc::atlas;
static constexpr float kPi = std::numbers::pi_v<float>;

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}
static Eigen::VectorXf unit(int dim, int k)
{
    Eigen::VectorXf v = Eigen::VectorXf::Zero(dim);
    v[k] = 1.f;
    return v;
}
static ViewObs obs(int k_shape, int k_col, float bearing, float range = 1.5f, float q = 1.f)
{
    ViewObs o;
    o.shape = unit(4, k_shape); o.colour = unit(3, k_col);
    o.bearing_rad = bearing; o.range_m = range; o.quality = q; o.stamp_ms = 1;
    return o;
}

static void test_bins_and_wrap()
{
    std::printf("bins and wrap\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    check(a.bin_of(0.f) == 6, "bearing 0 is bin 6 of 12");
    check(a.bin_of(kPi - 1e-4f) == 11, "just below +pi is bin 11");
    check(a.bin_of(-kPi + 1e-4f) == 0, "just above -pi is bin 0");
    check(a.bin_of(3.f * kPi) == a.bin_of(kPi), "bearings are wrapped first");
    a.update(obs(0, 0, kPi - 0.01f));
    check(std::fabs(a.gap(-kPi + 0.01f) - 0.02f) < 1e-3f, "gap across +-pi is 0.02 rad, not 2pi-0.02");
}

static void test_empty_atlas()
{
    std::printf("empty atlas\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    check(a.effective_count(0.3f) == a.params().n0, "empty atlas: N(beta) == n0");
    check(std::isinf(a.gap(0.f)), "empty atlas: gap is +inf");
    check(not a.predict(0.f).has_shape, "empty atlas: no shape prediction");
    check(a.kernel(a.gap(0.f)) == 0.f, "kernel(+inf) == 0");
}

static void test_update_and_repeat_discount()
{
    std::printf("update and repeat-look discount\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(0, 0, 0.f));
    const float n1 = a.state().bins[6].n;
    check(std::fabs(n1 - 1.f) < 1e-6f, "first view in an empty bin adds its full quality");
    a.update(obs(0, 0, 0.f));
    check(a.state().bins[6].n - n1 < 0.01f, "an identical repeat look adds ~nothing");
    a.update(obs(1, 0, 0.4f));          // same bin (bin 6 = [0, 30 deg)), 23 deg away
    check(a.state().bins[6].n > 1.9f, "a view 23 deg away counts as a new view");
    check(std::fabs(a.state().bins[6].mean.norm() - 1.f) < 1e-5f, "bin mean stays unit length");
    const float before = a.state().bins[6].n;
    a.update(obs(0, 0, 0.f, 1.5f, 0.f));
    check(a.state().bins[6].n == before, "quality 0 adds nothing");
    ViewObs bad = obs(0, 0, 0.f); bad.shape = Eigen::VectorXf::Ones(7);
    a.update(bad);
    check(a.state().bins[6].n == before, "wrong shape dimension is ignored");
    a.update(obs(0, 0, 0.f, 4.0f));     // same bearing, much farther: log-range differs by ~1
    check(a.state().bins[6].n > before + 0.5f, "same bearing at a very different range is a new view");
}

static void test_count_and_learning_value()
{
    std::printf("effective count and learning value\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(0, 0, 0.f));
    const float n0 = a.params().n0;
    check(std::fabs(a.effective_count(0.f) - (n0 + 1.f)) < 1e-3f, "N(0) == n0 + 1 after one view at 0");
    check(a.effective_count(kPi) < n0 + 0.01f, "N(pi) ~ n0: the far side is unseen");
    check(a.learning_value(0.f, 1.f) < a.learning_value(kPi, 1.f), "seen side is worth less than the unseen side");
    check(a.learning_value(kPi, 0.f) == 0.f, "quality 0 is worth nothing");
    check(a.learning_value(0.f, 1.f) > 0.f, "learning value is positive");
}

static void test_prediction()
{
    std::printf("prediction\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(2, 0, 0.f));
    auto p = a.predict(kPi / 2);
    check(p.has_shape, "a populated atlas predicts a shape everywhere");
    check(std::fabs(p.gap_rad - kPi / 2) < 1e-4f, "gap to the only view is 90 deg");
    check(std::fabs(p.shape[2] - 1.f) < 1e-5f, "prediction is the stored mean");
}

static void test_quality()
{
    std::printf("view quality\n");
    check(view_quality(0.f, 1.f, 100000, 2000.f) > 0.99f, "large, untruncated, visible mask ~ 1");
    check(view_quality(0.5f, 1.f, 100000, 2000.f) < 0.51f, "half truncated ~ halves quality");
    check(view_quality(0.f, 1.f, 100, 2000.f) < 0.06f, "tiny mask has low quality");
    check(view_quality(0.f, 1.f, 0, 2000.f) == 0.f, "empty mask has zero quality");
}

int main()
{
    test_bins_and_wrap();
    test_empty_atlas();
    test_update_and_repeat_discount();
    test_count_and_learning_value();
    test_prediction();
    test_quality();
    std::printf("\nview_atlas_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o /tmp/view_atlas_test`
Expected: compile error `view_atlas.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `common/view_atlas/view_atlas.h`:

```cpp
/*
 * view_atlas.h — per-object appearance atlas (spec: VIEW_ATLAS.md, same directory).
 *
 * For one object: 12 azimuth bins of DINOv3 shape in the OBJECT frame + one colour prototype. Evidence is
 * counted in VIEWS, not pixels, and a repeat look from the same place adds ~nothing (update()).
 * Header-only, pure Eigen: no DSR/DDS/OpenCV/ONNX, so every function here is unit-testable in isolation.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

namespace rc::atlas
{

inline constexpr float kPi = std::numbers::pi_v<float>;

/// Wrap an angle to (-pi, pi].
inline float wrap_pi(float a)
{
    float r = std::remainder(a, 2.f * kPi);
    if (r <= -kPi) r += 2.f * kPi;
    return r;
}

/// Model constants. Defaults are the 2026-10-17 spike's measurements; view_atlas_constants.txt (fitted,
/// Task 6) overrides them via atlas_io.h::load_params. FLAGGED (not fittable, see VIEW_ATLAS.md §3.5/§4):
/// n0, lambda_nats, max_rate_hz, pi_move, quality_px_scale.
struct AtlasParams
{
    int   n_bins          = 12;
    float ell_rad         = 0.90f;   // shape relaxation kernel kappa(D) = exp(-D^2 / 2 ell^2)
    float repeat_ell_rad  = 0.10f;   // views closer than this in azimuth are repeats...
    float repeat_ell_logr = 0.20f;   // ...and closer than this in log-range
    int   poses_per_bin   = 4;       // stored view poses per bin for the repeat discount
    float n0              = 1.0f;    // FLAGGED: prior pseudo-count, "worth one view"
    Eigen::Vector2f mu_own{0.96f, 0.97f};      // (s_shape at gap 0, s_colour) for "it is this object"
    Eigen::Vector2f mu_bg {0.87f, 0.50f};      // same for "another object of this class"
    Eigen::Matrix2f cov_own = Eigen::Vector2f(9e-4f, 9e-4f).asDiagonal();
    Eigen::Matrix2f cov_bg  = Eigen::Vector2f(2.5e-3f, 6.25e-2f).asDiagonal();
    Eigen::Matrix2f cov_n   = Eigen::Vector2f(9e-4f, 9e-4f).asDiagonal();   // per-view noise
    float lambda_nats      = 0.05f;  // FLAGGED: price of one encoding (CPU cost, outside the model)
    float max_rate_hz      = 4.0f;   // FLAGGED: machine guard, not part of the model
    float pi_move          = 0.2f;   // FLAGGED: prior probability an archived object was moved
    float quality_px_scale = 2000.f; // FLAGGED: mask pixel count at which resolution quality is 1 - 1/e
};

struct Bin
{
    Eigen::VectorXf             mean;          // unit vector, or zeros while n == 0
    float                       n = 0.f;       // effective view count
    float                       bearing = 0.f; // circular mean bearing of the views it holds (object frame)
    std::vector<Eigen::Vector2f> poses;        // last poses (bearing, log range), newest last
    std::uint64_t               stamp_ms = 0;
};

struct AtlasState
{
    int              shape_dim = 0, colour_dim = 0;
    std::vector<Bin> bins;
    Eigen::VectorXf  colour;                   // unit vector, or zeros while n_c == 0
    float            n_c = 0.f;
};

/// One encoded view, already reduced to the object frame by the caller.
struct ViewObs
{
    Eigen::VectorXf shape, colour;
    float bearing_rad = 0.f;   // object-frame azimuth of the camera seen from the object
    float range_m = 1.f;
    float quality = 1.f;       // view_quality(): 0 = no evidence, 1 = full view
    std::uint64_t stamp_ms = 0;
};

/// Continuous view quality in [0,1]: untruncated fraction x visible fraction x resolution factor.
inline float view_quality(float trunc_frac, float visible_frac, int mask_px, float px_scale)
{
    if (not (mask_px > 0) or not (px_scale > 0.f)) return 0.f;
    const float untrunc = std::clamp(1.f - trunc_frac, 0.f, 1.f);     // fractions are bounded by definition
    const float vis     = std::clamp(visible_frac, 0.f, 1.f);
    const float res     = 1.f - std::exp(-float(mask_px) / px_scale);
    return untrunc * vis * res;
}

class ViewAtlas
{
public:
    ViewAtlas(const AtlasParams& p, int shape_dim, int colour_dim) : p_(p)
    {
        s_.shape_dim = shape_dim; s_.colour_dim = colour_dim;
        s_.bins.resize(std::size_t(p_.n_bins));
        for (auto& b : s_.bins) b.mean = Eigen::VectorXf::Zero(shape_dim);
        s_.colour = Eigen::VectorXf::Zero(colour_dim);
    }
    static ViewAtlas from_state(const AtlasParams& p, AtlasState s)
    {
        ViewAtlas a(p, s.shape_dim, s.colour_dim);
        a.s_ = std::move(s);
        return a;
    }
    [[nodiscard]] const AtlasState&  state()  const noexcept { return s_; }
    [[nodiscard]] const AtlasParams& params() const noexcept { return p_; }

    [[nodiscard]] int bin_of(float bearing) const
    {
        const float u = (wrap_pi(bearing) + kPi) / (2.f * kPi);        // (0, 1]
        return std::clamp(int(std::floor(u * float(p_.n_bins))), 0, p_.n_bins - 1);   // index safety only
    }
    [[nodiscard]] float kernel(float d) const
    {
        if (std::isinf(d)) return 0.f;
        return std::exp(-0.5f * d * d / (p_.ell_rad * p_.ell_rad));
    }
    /// N(beta) = n0 + sum_b n_b kappa(beta - bearing_b): how well the object is known from beta.
    [[nodiscard]] float effective_count(float bearing) const
    {
        float N = p_.n0;
        for (const auto& b : s_.bins)
            if (b.n > 0.f) N += b.n * kernel(wrap_pi(bearing - b.bearing));
        return N;
    }
    /// Information (nats) one view of this quality at this bearing would add: 1/2 log((N + w) / N).
    [[nodiscard]] float learning_value(float bearing, float quality) const
    {
        if (not (quality > 0.f)) return 0.f;
        const float N = effective_count(bearing);
        return 0.5f * std::log((N + quality) / N);
    }
    /// Angular distance to the nearest populated bin; +inf when the atlas holds no shape at all.
    [[nodiscard]] float gap(float bearing) const
    {
        float g = std::numeric_limits<float>::infinity();
        for (const auto& b : s_.bins)
            if (b.n > 0.f) g = std::min(g, std::fabs(wrap_pi(bearing - b.bearing)));
        return g;
    }

    struct Prediction { Eigen::VectorXf shape; float gap_rad = 0.f; bool has_shape = false; };
    [[nodiscard]] Prediction predict(float bearing) const
    {
        Prediction pr;
        pr.shape = Eigen::VectorXf::Zero(s_.shape_dim);
        for (const auto& b : s_.bins)
            if (b.n > 0.f) pr.shape += b.n * kernel(wrap_pi(bearing - b.bearing)) * b.mean;
        pr.gap_rad = gap(bearing);
        const float nrm = pr.shape.norm();
        pr.has_shape = nrm > 0.f;
        if (pr.has_shape) pr.shape /= nrm;
        return pr;
    }

    /// Fold one encoded view in. Weight = quality x (1 - similarity to the bin's stored view poses), so a
    /// repeat look from the same place adds ~nothing. No process noise: appearance does not drift.
    void update(const ViewObs& o)
    {
        if (not (o.quality > 0.f) or not (o.range_m > 0.f)) return;
        if (o.shape.size() != s_.shape_dim or o.colour.size() != s_.colour_dim) return;
        const float sn = o.shape.norm(), cn = o.colour.norm();
        if (not (sn > 0.f) or not std::isfinite(sn) or not std::isfinite(cn)) return;

        const float b  = wrap_pi(o.bearing_rad);
        const float lr = std::log(o.range_m);
        Bin& bin = s_.bins[std::size_t(bin_of(b))];
        const float w = o.quality * (1.f - repeat_similarity(bin, b, lr));

        if (w > 0.f)
        {
            Eigen::VectorXf m = bin.n * bin.mean + w * (o.shape / sn);
            bin.mean = m / m.norm();
            const float sx = bin.n * std::sin(bin.bearing) + w * std::sin(b);
            const float cx = bin.n * std::cos(bin.bearing) + w * std::cos(b);
            bin.bearing = std::atan2(sx, cx);
            bin.n += w;
            if (cn > 0.f)
            {
                Eigen::VectorXf c = s_.n_c * s_.colour + w * (o.colour / cn);
                s_.colour = c / c.norm();
                s_.n_c += w;
            }
        }
        bin.poses.emplace_back(b, lr);
        if (int(bin.poses.size()) > p_.poses_per_bin) bin.poses.erase(bin.poses.begin());
        bin.stamp_ms = o.stamp_ms;
    }

private:
    [[nodiscard]] float repeat_similarity(const Bin& bin, float b, float lr) const
    {
        float k = 0.f;
        for (const auto& ps : bin.poses)
        {
            const float da = wrap_pi(b - ps.x()) / p_.repeat_ell_rad;
            const float dr = (lr - ps.y()) / p_.repeat_ell_logr;
            k = std::max(k, std::exp(-0.5f * (da * da + dr * dr)));
        }
        return k;
    }

    AtlasParams p_;
    AtlasState  s_;
};

} // namespace rc::atlas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o /tmp/view_atlas_test && /tmp/view_atlas_test`
Expected: every line `ok`, last line `view_atlas_test: PASS`.

- [ ] **Step 5: Add "no decoder" to the spec's out-of-scope list**

In `common/view_atlas/VIEW_ATLAS.md` §7, append to the paragraph:

```markdown
No decoder (latent → image): the model predicts in descriptor space, which is the observation the
likelihood scores; a pixel decoder would add a trained model and score lighting and background, which
the descriptors were chosen to ignore. A per-bin stored thumbnail covers the display/debug use.
```

- [ ] **Step 6: Run the whole common suite and commit**

Run: `bash common/run_tests.sh`
Expected: last line `common: all test binaries passed`.

```bash
git add common/view_atlas/view_atlas.h common/view_atlas/view_atlas_test.cpp common/view_atlas/VIEW_ATLAS.md
git commit -m "view_atlas: atlas state, repeat-discounted update, prediction, learning value (+tests)"
```

---

### Task 2: Likelihood, yaw-shift marginalisation, re-ID posterior, identity value

**Files:**
- Modify: `common/view_atlas/view_atlas.h` (add inside `namespace rc::atlas`, after `class ViewAtlas`; add three public methods to the class)
- Modify: `common/view_atlas/view_atlas_test.cpp`

**Interfaces:**
- Consumes: Task 1's `ViewAtlas`, `ViewObs`, `AtlasParams`, `wrap_pi`.
- Produces:
  - `struct Gauss2 { Eigen::Vector2f mu; Eigen::Matrix2f cov; }`, `float logpdf(const Eigen::Vector2f&, const Gauss2&)`
  - `ViewAtlas::similarities(const ViewObs&, float bearing) const -> Eigen::Vector2f`
  - `ViewAtlas::hypothesis(float bearing, float quality) const -> Gauss2`
  - `Gauss2 background(const AtlasParams&, float quality)`
  - `ViewAtlas::log_ratio(const ViewObs&, float bearing) const -> float`
  - `std::vector<float> shift_log_weights(int n_bins, float yaw_sigma_rad)`
  - `ViewAtlas::log_ratio_marginal(const ViewObs&, float bearing, const std::vector<float>& shift_logw) const -> float`
  - `float location_log_lik(const Eigen::Vector2f& x, const Eigen::Vector2f& mu, const Eigen::Matrix2f& cov, float pi_move, float room_area_m2)`
  - `struct ReidCandidate { float log_ratio, log_loc, log_prior; }`, `std::vector<float> reid_posterior(const std::vector<ReidCandidate>&, float log_prior_new, float log_loc_new)` (last element = "new"), `float entropy(const std::vector<float>&)`
  - `struct IdHyp { Gauss2 own, bg; float log_prior, log_loc; }`, `float identity_value(const std::vector<IdHyp>&, float log_prior_new, float log_loc_new, int n_samples, std::uint32_t seed)`

- [ ] **Step 1: Write the failing tests**

Add to `view_atlas_test.cpp` before `main`:

```cpp
static ViewAtlas atlas_with(int k_shape, int k_col, float bearing, int n_views = 3)
{
    ViewAtlas a(AtlasParams{}, 4, 3);
    for (int i = 0; i < n_views; ++i) a.update(obs(k_shape, k_col, bearing + 0.3f * float(i) - 0.3f));
    return a;
}

static void test_log_ratio()
{
    std::printf("log ratio\n");
    ViewAtlas empty(AtlasParams{}, 4, 3);
    const float l0 = empty.log_ratio(obs(0, 0, 0.f), 0.f);
    check(l0 == 0.f, "empty atlas: log ratio is exactly 0");
    check(std::isfinite(l0), "empty atlas: log ratio is finite");

    ViewAtlas a = atlas_with(0, 0, 0.f);
    check(a.log_ratio(obs(0, 0, 0.f), 0.f) > 2.f, "matching shape+colour at a seen bearing: strong evidence for");
    check(a.log_ratio(obs(1, 1, 0.f), 0.f) < -2.f, "different shape+colour at a seen bearing: evidence against");

    // shape evidence must vanish on the unseen side while colour evidence stays
    const float match_far = a.log_ratio(obs(0, 0, kPi), kPi);
    const float other_far = a.log_ratio(obs(1, 0, kPi), kPi);
    const float match_near = a.log_ratio(obs(0, 0, 0.f), 0.f);
    const float other_near = a.log_ratio(obs(1, 0, 0.f), 0.f);
    check(std::fabs(match_far - other_far) < 0.05f, "unseen side: shape no longer discriminates");
    check(match_near - other_near > 2.f, "seen side: shape discriminates");
    check(a.log_ratio(obs(0, 0, 0.f, 1.5f, 1e-6f), 0.f) < 1e-2f, "quality ~0: no evidence either way");
}

static void test_shift_marginal()
{
    std::printf("yaw shift marginalisation\n");
    const auto sharp = shift_log_weights(12, 0.f);
    check(sharp[0] == 0.f and std::isinf(sharp[1]), "sigma 0: all weight on shift 0");
    const auto flat = shift_log_weights(12, 1e3f);
    check(std::fabs(flat[0] - flat[6]) < 1e-3f, "huge sigma: flat over shifts");
    float lse = -std::numeric_limits<float>::infinity();
    for (float w : flat) lse = std::max(lse, w) + std::log1p(std::exp(-std::fabs(lse - w)));
    check(std::fabs(lse) < 1e-3f, "shift weights are normalised");

    ViewAtlas a = atlas_with(0, 0, 0.f);
    const float off = 3.f * 2.f * kPi / 12.f;                 // observed 3 bins away (wrong yaw)
    const float l_fixed = a.log_ratio(obs(0, 0, off), off);
    const float l_flat  = a.log_ratio_marginal(obs(0, 0, off), off, flat);
    const float l_sharp = a.log_ratio_marginal(obs(0, 0, off), off, sharp);
    check(std::fabs(l_sharp - l_fixed) < 1e-4f, "sigma 0 reproduces the fixed-yaw ratio");
    check(l_flat > l_fixed, "unknown yaw: marginalising recovers evidence a wrong yaw hid");
}

static void test_reid()
{
    std::printf("re-identification posterior\n");
    std::vector<ReidCandidate> c = {{8.f, 0.f, 0.f}, {-3.f, 0.f, 0.f}};
    auto p = reid_posterior(c, 0.f, 0.f);
    float s = 0.f; for (float v : p) s += v;
    check(p.size() == 3 and std::fabs(s - 1.f) < 1e-5f, "posterior has candidates + new and sums to 1");
    check(p[0] > 0.99f, "strong appearance evidence picks candidate 0");
    auto pe = reid_posterior({}, 0.f, 0.f);
    check(pe.size() == 1 and pe[0] == 1.f, "no candidates: certainly new");
    check(std::fabs(entropy({0.5f, 0.5f}) - std::log(2.f)) < 1e-5f, "entropy of a fair coin is log 2");

    const Eigen::Vector2f mu(0.f, 0.f);
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * 0.01f;
    const float near_stay = location_log_lik({0.f, 0.f}, mu, cov, 0.f, 50.f);
    check(std::fabs(near_stay - (-std::log(2.f * kPi * 0.01f))) < 1e-3f, "pi_move 0: plain Gaussian at the mean");
    const float far = location_log_lik({20.f, 0.f}, mu, cov, 0.2f, 50.f);
    check(std::fabs(far - std::log(0.2f / 50.f)) < 1e-3f, "far away: only the moved-uniform term remains");
}

static void test_identity_value()
{
    std::printf("identity value of encoding\n");
    const AtlasParams P;
    ViewAtlas a = atlas_with(0, 0, 0.f), b = atlas_with(0, 0, 0.f);
    std::vector<IdHyp> same = {{a.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f},
                               {b.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f}};
    std::vector<IdHyp> none = {};
    const float v_none = identity_value(none, 0.f, 0.f, 64, 1);
    check(v_none == 0.f, "no candidates: nothing to learn about identity");
    ViewAtlas empty(P, 4, 3);
    std::vector<IdHyp> blind = {{empty.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f}};
    check(identity_value(blind, 0.f, 0.f, 256, 1) < 0.02f, "an atlas that predicts nothing: ~0 value");
    std::vector<IdHyp> one = {{a.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f}};
    check(identity_value(one, 0.f, 0.f, 256, 1) > 0.1f, "a seen candidate vs new: encoding is worth something");
    check(std::isfinite(identity_value(same, 0.f, 0.f, 256, 1)), "two identical candidates: finite value");
}
```

Add to `main()` before the summary line:

```cpp
    test_log_ratio();
    test_shift_marginal();
    test_reid();
    test_identity_value();
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o /tmp/view_atlas_test`
Expected: compile errors (`log_ratio`, `shift_log_weights`, `reid_posterior` … not declared).

- [ ] **Step 3: Write the implementation**

Add `#include <random>` to the includes of `view_atlas.h`. Before `class ViewAtlas`, add:

```cpp
struct Gauss2 { Eigen::Vector2f mu = Eigen::Vector2f::Zero(); Eigen::Matrix2f cov = Eigen::Matrix2f::Identity(); };

inline float logpdf(const Eigen::Vector2f& x, const Gauss2& g)
{
    const Eigen::Vector2f d = x - g.mu;
    const Eigen::LLT<Eigen::Matrix2f> llt(g.cov);
    const Eigen::Vector2f z = llt.matrixL().solve(d);
    const float logdet = 2.f * (std::log(llt.matrixL()(0, 0)) + std::log(llt.matrixL()(1, 1)));
    return -0.5f * z.squaredNorm() - 0.5f * logdet - std::log(2.f * kPi);
}

/// Observation-quality noise added to BOTH hypotheses: q -> 0 makes them indistinguishable (ratio -> 0).
inline Eigen::Matrix2f quality_cov(const AtlasParams& p, float quality)
{
    return p.cov_n * ((1.f - quality) / quality);
}

/// "Another object of this class" (H_bg).
inline Gauss2 background(const AtlasParams& p, float quality)
{
    return {p.mu_bg, p.cov_bg + quality_cov(p, quality)};
}
```

Inside `class ViewAtlas`, in the public section after `update`, add:

```cpp
    /// (s_shape, s_colour): the view against the shape predicted at `bearing` and against the colour prototype.
    /// A channel with no stored evidence reads 0 — its hypothesis has relaxed to the background there.
    [[nodiscard]] Eigen::Vector2f similarities(const ViewObs& o, float bearing) const
    {
        const auto pr = predict(bearing);
        const float sn = o.shape.norm(), cn = o.colour.norm();
        const float sd = (pr.has_shape and sn > 0.f) ? o.shape.dot(pr.shape) / sn : 0.f;
        const float sc = (s_.n_c > 0.f and cn > 0.f) ? o.colour.dot(s_.colour) / cn : 0.f;
        return {sd, sc};
    }

    /// H_k at this bearing: relaxes from (mu_own, cov_own) to the background as the angular gap grows
    /// (shape) or as evidence vanishes (both). Cholesky factors are blended, so the covariance is always
    /// PSD and equals cov_bg EXACTLY when there is no evidence — that is what makes the ratio exactly 0.
    [[nodiscard]] Gauss2 hypothesis(float bearing, float quality) const
    {
        const float Ns = effective_count(bearing) - p_.n0;
        const float r_d = kernel(gap(bearing)) * Ns / (Ns + p_.n0);
        const float r_c = s_.n_c / (s_.n_c + p_.n0);
        const Eigen::Matrix2f R = Eigen::Vector2f(r_d, r_c).asDiagonal();
        const Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
        const Eigen::Matrix2f Lb = Eigen::LLT<Eigen::Matrix2f>(p_.cov_bg).matrixL();
        const Eigen::Matrix2f Lo = Eigen::LLT<Eigen::Matrix2f>(p_.cov_own).matrixL();
        const Eigen::Matrix2f L = (I - R) * Lb + R * Lo;
        Gauss2 g;
        g.mu  = p_.mu_bg + R * (p_.mu_own - p_.mu_bg);
        g.cov = L * L.transpose() + quality_cov(p_, quality);
        return g;
    }

    /// Lambda = log N(s; H_k) - log N(s; H_bg). Exactly 0 when the atlas holds no evidence.
    [[nodiscard]] float log_ratio(const ViewObs& o, float bearing) const
    {
        if (not (o.quality > 0.f)) return 0.f;
        const Eigen::Vector2f s = similarities(o, bearing);
        return logpdf(s, hypothesis(bearing, o.quality)) - logpdf(s, background(p_, o.quality));
    }

    /// Yaw unknown or uncertain: marginalise the cyclic shift, log sum_k w_k exp(Lambda(bearing + k*bin)).
    [[nodiscard]] float log_ratio_marginal(const ViewObs& o, float bearing, const std::vector<float>& shift_logw) const
    {
        const float step = 2.f * kPi / float(p_.n_bins);
        float m = -std::numeric_limits<float>::infinity();
        std::vector<float> t(shift_logw.size());
        for (std::size_t k = 0; k < shift_logw.size(); ++k)
        {
            t[k] = std::isinf(shift_logw[k]) ? shift_logw[k]
                                             : shift_logw[k] + log_ratio(o, bearing + step * float(k));
            m = std::max(m, t[k]);
        }
        if (std::isinf(m)) return 0.f;
        float acc = 0.f;
        for (float v : t) acc += std::exp(v - m);
        return m + std::log(acc);
    }
```

After `class ViewAtlas`, add:

```cpp
/// Log-weights over the n_bins cyclic shifts from a wrapped-Gaussian yaw posterior of sd sigma.
/// sigma <= 0: all weight on shift 0. Normalised (log-sum-exp == 0).
inline std::vector<float> shift_log_weights(int n_bins, float yaw_sigma_rad)
{
    std::vector<float> w(std::size_t(n_bins), -std::numeric_limits<float>::infinity());
    if (not (yaw_sigma_rad > 0.f)) { w[0] = 0.f; return w; }
    const float step = 2.f * kPi / float(n_bins);
    float m = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < n_bins; ++k)
    {
        const float d = wrap_pi(step * float(k)) / yaw_sigma_rad;
        w[std::size_t(k)] = -0.5f * d * d;
        m = std::max(m, w[std::size_t(k)]);
    }
    float acc = 0.f;
    for (float v : w) acc += std::exp(v - m);
    const float lse = m + std::log(acc);
    for (float& v : w) v -= lse;
    return w;
}

/// Where was it? Mixture of "stayed" (Gaussian at the archived pose) and "moved" (uniform over the room).
inline float location_log_lik(const Eigen::Vector2f& x, const Eigen::Vector2f& mu, const Eigen::Matrix2f& cov,
                              float pi_move, float room_area_m2)
{
    const float stay = std::exp(logpdf(x, {mu, cov}));
    return std::log((1.f - pi_move) * stay + pi_move / room_area_m2);
}

struct ReidCandidate { float log_ratio = 0.f, log_loc = 0.f, log_prior = 0.f; };

/// Posterior over {candidate 0..n-1, new}; the last element is "new".
inline std::vector<float> reid_posterior(const std::vector<ReidCandidate>& c, float log_prior_new, float log_loc_new)
{
    std::vector<float> lw;
    for (const auto& k : c) lw.push_back(k.log_prior + k.log_loc + k.log_ratio);
    lw.push_back(log_prior_new + log_loc_new);
    const float m = *std::max_element(lw.begin(), lw.end());
    float acc = 0.f;
    for (float& v : lw) { v = std::exp(v - m); acc += v; }
    for (float& v : lw) v /= acc;
    return lw;
}

inline float entropy(const std::vector<float>& p)
{
    float h = 0.f;
    for (float v : p) if (v > 0.f) h -= v * std::log(v);
    return h;
}

struct IdHyp { Gauss2 own, bg; float log_prior = 0.f, log_loc = 0.f; };

/// Expected entropy drop over {candidates, new} from encoding one view (nats), by Monte Carlo: under
/// "it is j", candidate j's similarities follow its own hypothesis and every other candidate's follow
/// that candidate's background; under "new", all follow their backgrounds.
inline float identity_value(const std::vector<IdHyp>& c, float log_prior_new, float log_loc_new,
                            int n_samples, std::uint32_t seed)
{
    if (c.empty() or n_samples <= 0) return 0.f;
    std::vector<ReidCandidate> prior;
    for (const auto& h : c) prior.push_back({0.f, h.log_loc, h.log_prior});
    const auto p0 = reid_posterior(prior, log_prior_new, log_loc_new);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    auto draw = [&](const Gauss2& h) {
        const Eigen::Matrix2f L = Eigen::LLT<Eigen::Matrix2f>(h.cov).matrixL();
        return Eigen::Vector2f(h.mu + L * Eigen::Vector2f(g(rng), g(rng)));
    };
    float expected_h = 0.f;
    for (std::size_t t = 0; t <= c.size(); ++t)            // t == c.size() is "new"
    {
        float acc = 0.f;
        for (int s = 0; s < n_samples; ++s)
        {
            std::vector<ReidCandidate> obs = prior;
            for (std::size_t j = 0; j < c.size(); ++j)
            {
                const Eigen::Vector2f x = draw(t == j ? c[j].own : c[j].bg);
                obs[j].log_ratio = logpdf(x, c[j].own) - logpdf(x, c[j].bg);
            }
            acc += entropy(reid_posterior(obs, log_prior_new, log_loc_new));
        }
        expected_h += p0[t] * acc / float(n_samples);
    }
    return std::max(0.f, entropy(p0) - expected_h);   // MC noise only; mutual information is >= 0
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o /tmp/view_atlas_test && /tmp/view_atlas_test`
Expected: all `ok`, `view_atlas_test: PASS`. If "matching shape+colour ... strong evidence" fails, print `a.similarities(...)` and `a.hypothesis(0,1).mu` — with the default constants s = (1, 1), H_k.mu ≈ (0.96, 0.97) and H_bg.mu = (0.87, 0.50); the ratio must be large and positive.

- [ ] **Step 5: Commit**

```bash
git add common/view_atlas/view_atlas.h common/view_atlas/view_atlas_test.cpp
git commit -m "view_atlas: likelihood ratio, yaw-shift marginalisation, re-ID posterior, identity value (+tests)"
```

---

### Task 3: Archive persistence and constants loader

**Files:**
- Create: `common/view_atlas/atlas_io.h`
- Create: `common/view_atlas/atlas_io_test.cpp`
- Modify: `common/view_atlas/VIEW_ATLAS.md` (§5: cross-reference the episodic memory design)

**Interfaces:**
- Consumes: Task 1/2 `AtlasParams`, `AtlasState`, `Bin`.
- Produces (namespace `rc::atlas`):
  - `struct ArchiveRecord { std::uint64_t object_uid; std::string agent, cls, name; Eigen::Vector3f pose; Eigen::Matrix3f cov; std::uint64_t born_ms, died_ms; AtlasState atlas; }`
  - `bool save_archive(const std::string& base, const std::vector<ArchiveRecord>&, std::string* why_not)` — writes `base + ".idx"` (text) and `base + ".blob"` (binary)
  - `bool load_archive(const std::string& base, const AtlasParams&, int shape_dim, int colour_dim, std::vector<ArchiveRecord>& out, std::string* why_not)`
  - `bool load_params(const std::string& path, AtlasParams& p, std::string* why_not)` — `key = value` lines, `#` comments

- [ ] **Step 1: Write the failing test**

Create `common/view_atlas/atlas_io_test.cpp`:

```cpp
/*
 * atlas_io_test.cpp — archive round-trip, locale, corruption; constants loader. PARSES FILES, so it sets
 * the agents' locale first (CLAUDE.md): a "C"-locale harness would answer a different question.
 */
#include "atlas_io.h"

#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace rc::atlas;
static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}

static ArchiveRecord make_record(std::uint64_t uid)
{
    AtlasParams P;
    ViewAtlas a(P, 4, 3);
    ViewObs o; o.shape = Eigen::Vector4f(0.1f, 0.2f, 0.3f, 0.9f); o.colour = Eigen::Vector3f(0.3f, 0.3f, 0.9f);
    o.bearing_rad = 0.123456789f; o.range_m = 1.987654321f; o.quality = 0.7f; o.stamp_ms = 1790000000123ull;
    a.update(o);
    o.bearing_rad = -2.5f; a.update(o);
    ArchiveRecord r;
    r.object_uid = uid; r.agent = "chair_concept"; r.cls = "chair"; r.name = "chair_2";
    r.pose = Eigen::Vector3f(-6.90769f, 1.43326f, 1.0472f);
    r.cov = Eigen::Matrix3f::Identity() * 0.0123456f; r.cov(0, 1) = r.cov(1, 0) = 0.001f;
    r.born_ms = 1790000000000ull; r.died_ms = 1790000099999ull;
    r.atlas = a.state();
    return r;
}

static bool same(const ArchiveRecord& a, const ArchiveRecord& b)
{
    if (a.object_uid != b.object_uid or a.agent != b.agent or a.cls != b.cls or a.name != b.name) return false;
    if (a.pose != b.pose or a.cov != b.cov or a.born_ms != b.born_ms or a.died_ms != b.died_ms) return false;
    if (a.atlas.n_c != b.atlas.n_c or a.atlas.colour != b.atlas.colour) return false;
    for (std::size_t i = 0; i < a.atlas.bins.size(); ++i)
    {
        const auto &x = a.atlas.bins[i], &y = b.atlas.bins[i];
        if (x.n != y.n or x.bearing != y.bearing or x.stamp_ms != y.stamp_ms or x.mean != y.mean) return false;
        if (x.poses.size() != y.poses.size()) return false;
        for (std::size_t k = 0; k < x.poses.size(); ++k) if (x.poses[k] != y.poses[k]) return false;
    }
    return true;
}

int main()
{
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    std::printf("locale: %s\n", std::setlocale(LC_NUMERIC, nullptr));
    const auto dir = std::filesystem::temp_directory_path() / "atlas_io_test";
    std::filesystem::create_directories(dir);
    const std::string base = (dir / "atlas_chair_concept_22").string();
    const AtlasParams P;

    std::printf("round trip\n");
    std::vector<ArchiveRecord> recs = {make_record(7), make_record(8)};
    std::string why;
    check(save_archive(base, recs, &why), "save succeeds");
    std::vector<ArchiveRecord> back;
    check(load_archive(base, P, 4, 3, back, &why), "load succeeds");
    check(back.size() == 2 and same(back[0], recs[0]) and same(back[1], recs[1]), "records read back bit-exact under es_ES");

    std::printf("refusals\n");
    back.clear();
    check(not load_archive(base, P, 5, 3, back, &why) and why.find("shape_dim") != std::string::npos,
          "wrong shape dimension refused, reason names it");
    check(back.empty(), "refused load leaves output empty");
    {
        std::ifstream in(base + ".blob", std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), {});
        std::ofstream out(base + ".blob", std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), std::streamsize(bytes.size() / 2));
    }
    check(not load_archive(base, P, 4, 3, back, &why) and why.find("blob short") != std::string::npos,
          "truncated blob refused");
    {
        std::ofstream out(base + ".blob", std::ios::binary | std::ios::trunc);
        out.write("XXXX", 4);
    }
    check(not load_archive(base, P, 4, 3, back, &why) and why.find("magic") != std::string::npos,
          "wrong magic refused");
    check(not load_archive((dir / "missing").string(), P, 4, 3, back, &why), "missing files refused");
    ArchiveRecord bad = make_record(9); bad.name = "chair,2";
    check(not save_archive(base, {bad}, &why) and why.find("comma") != std::string::npos,
          "a comma in a text field is refused at save time");

    std::printf("constants loader\n");
    const std::string cpath = (dir / "constants.txt").string();
    {
        std::ofstream f(cpath);
        f << "# fitted test\nell_rad = 0.873\nmu_own_d = 0.9512\ncov_bg_dc = -0.00125\nrepeat_ell_rad=0.0912\n";
    }
    AtlasParams q;
    check(load_params(cpath, q, &why), "constants file loads");
    check(q.ell_rad == 0.873f and q.mu_own.x() == 0.9512f, "decimals parsed with points under es_ES");
    check(q.cov_bg(0, 1) == -0.00125f and q.cov_bg(1, 0) == -0.00125f, "off-diagonal set symmetrically");
    check(q.repeat_ell_rad == 0.0912f, "no spaces around '=' is fine");
    { std::ofstream f(cpath); f << "ell_radd = 0.5\n"; }
    check(not load_params(cpath, q, &why) and why.find("ell_radd") != std::string::npos, "unknown key refused");
    { std::ofstream f(cpath); f << "ell_rad = 0,5\n"; }
    check(not load_params(cpath, q, &why), "a comma decimal is refused, not read as 0");

    std::filesystem::remove_all(dir);
    std::printf("\natlas_io_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) atlas_io_test.cpp -o /tmp/atlas_io_test`
Expected: compile error `atlas_io.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `common/view_atlas/atlas_io.h`:

```cpp
/*
 * atlas_io.h — archive of dead objects' atlases + the fitted-constants loader (spec: VIEW_ATLAS.md).
 *
 * Text index (greppable) + binary blob (descriptors, no decimal round-trip), the place_map.h precedent.
 * Locale per CLAUDE.md: imbue(classic) + max_digits10 to write, std::from_chars to read. Binary written
 * field by field. Every refusal says which field disagreed. Record format is agent-agnostic on purpose:
 * a future long-term object memory absorbs these files unchanged.
 */
#pragma once

#include "view_atlas.h"

#include <charconv>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rc::atlas
{

inline constexpr std::uint32_t kArchiveVersion = 1;

struct ArchiveRecord
{
    std::uint64_t   object_uid = 0;
    std::string     agent, cls, name;
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();       // x, y, yaw at death (world frame)
    Eigen::Matrix3f cov  = Eigen::Matrix3f::Identity();
    std::uint64_t   born_ms = 0, died_ms = 0;
    AtlasState      atlas;
};

namespace detail
{
template <class T> void put(std::ofstream& b, T v) { b.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> bool get(std::ifstream& b, T& v) { return bool(b.read(reinterpret_cast<char*>(&v), sizeof(T))); }
inline void put_vec(std::ofstream& b, const Eigen::VectorXf& v, int dim)
{
    for (int i = 0; i < dim; ++i) put<float>(b, i < v.size() ? v[i] : 0.f);
}
inline bool get_vec(std::ifstream& b, Eigen::VectorXf& v, int dim)
{
    v.resize(dim);
    for (int i = 0; i < dim; ++i) if (not get<float>(b, v[i])) return false;
    return true;
}
template <class T> bool parse(std::string_view s, T& out)
{
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() and r.ptr == s.data() + s.size();
}
inline std::vector<std::string_view> split(std::string_view s, char sep)
{
    std::vector<std::string_view> out;
    std::size_t a = 0;
    for (std::size_t i = 0; i <= s.size(); ++i)
        if (i == s.size() or s[i] == sep) { out.push_back(s.substr(a, i - a)); a = i + 1; }
    return out;
}
inline std::string_view trim(std::string_view s)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
    while (not s.empty() and (s.back() == ' ' or s.back() == '\t' or s.back() == '\r')) s.remove_suffix(1);
    return s;
}
} // namespace detail

inline bool save_archive(const std::string& base, const std::vector<ArchiveRecord>& recs, std::string* why_not)
{
    auto fail = [&](std::string m) { if (why_not) *why_not = std::move(m); return false; };
    for (const auto& r : recs)
        for (const auto* s : {&r.agent, &r.cls, &r.name})
            if (s->find(',') != std::string::npos or s->find('\n') != std::string::npos)
                return fail("text field contains a comma or newline: '" + *s + "'");
    const int sd = recs.empty() ? 0 : recs.front().atlas.shape_dim;
    const int cd = recs.empty() ? 0 : recs.front().atlas.colour_dim;
    const int nb = recs.empty() ? 0 : int(recs.front().atlas.bins.size());
    for (const auto& r : recs)
        if (r.atlas.shape_dim != sd or r.atlas.colour_dim != cd or int(r.atlas.bins.size()) != nb)
            return fail("records disagree on shape_dim/colour_dim/n_bins");

    std::ofstream b(base + ".blob", std::ios::binary | std::ios::trunc);
    if (not b) return fail("cannot open " + base + ".blob");
    b.write("RCVA", 4);
    for (std::uint32_t f : {kArchiveVersion, std::uint32_t(sd), std::uint32_t(cd), std::uint32_t(nb),
                            std::uint32_t(recs.size())})
        detail::put(b, f);
    for (const auto& r : recs)
    {
        detail::put_vec(b, r.atlas.colour, cd);
        detail::put<float>(b, r.atlas.n_c);
        for (const auto& bin : r.atlas.bins)
        {
            detail::put<float>(b, bin.n);
            detail::put<float>(b, bin.bearing);
            detail::put<std::uint64_t>(b, bin.stamp_ms);
            detail::put<std::uint32_t>(b, std::uint32_t(bin.poses.size()));
            for (const auto& p : bin.poses) { detail::put<float>(b, p.x()); detail::put<float>(b, p.y()); }
            detail::put_vec(b, bin.mean, sd);
        }
    }
    if (not b) return fail("write failed: " + base + ".blob");
    b.close();

    std::ofstream f(base + ".idx", std::ios::trunc);
    if (not f) return fail("cannot open " + base + ".idx");
    f.imbue(std::locale::classic());
    f << std::setprecision(std::numeric_limits<float>::max_digits10);
    f << "# view_atlas archive v" << kArchiveVersion << " shape_dim=" << sd << " colour_dim=" << cd
      << " n_bins=" << nb << "\n"
      << "uid,agent,cls,name,x,y,yaw,c00,c01,c02,c11,c12,c22,born_ms,died_ms\n";
    for (const auto& r : recs)
        f << r.object_uid << ',' << r.agent << ',' << r.cls << ',' << r.name << ','
          << r.pose.x() << ',' << r.pose.y() << ',' << r.pose.z() << ','
          << r.cov(0, 0) << ',' << r.cov(0, 1) << ',' << r.cov(0, 2) << ','
          << r.cov(1, 1) << ',' << r.cov(1, 2) << ',' << r.cov(2, 2) << ','
          << r.born_ms << ',' << r.died_ms << '\n';
    return bool(f) or fail("write failed: " + base + ".idx");
}

inline bool load_archive(const std::string& base, const AtlasParams& p, int shape_dim, int colour_dim,
                         std::vector<ArchiveRecord>& out, std::string* why_not)
{
    out.clear();
    auto fail = [&](std::string m) { out.clear(); if (why_not) *why_not = std::move(m); return false; };

    std::ifstream f(base + ".idx");
    if (not f) return fail("cannot open " + base + ".idx");
    std::vector<ArchiveRecord> recs;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line[0] == '#' or line.starts_with("uid,")) continue;
        const auto c = detail::split(line, ',');
        if (c.size() != 15) return fail("index line has " + std::to_string(c.size()) + " fields, expected 15");
        ArchiveRecord r;
        float v[9];
        bool ok = detail::parse(c[0], r.object_uid) and detail::parse(c[13], r.born_ms) and detail::parse(c[14], r.died_ms);
        for (int i = 0; i < 9; ++i) ok = ok and detail::parse(c[std::size_t(4 + i)], v[i]);
        if (not ok) return fail("unparseable number in index line: " + line);
        r.agent = std::string(c[1]); r.cls = std::string(c[2]); r.name = std::string(c[3]);
        r.pose = Eigen::Vector3f(v[0], v[1], v[2]);
        r.cov << v[3], v[4], v[5], v[4], v[6], v[7], v[5], v[7], v[8];
        recs.push_back(std::move(r));
    }

    std::ifstream b(base + ".blob", std::ios::binary);
    if (not b) return fail("cannot open " + base + ".blob");
    char magic[4];
    if (not b.read(magic, 4) or std::memcmp(magic, "RCVA", 4) != 0) return fail("bad magic in " + base + ".blob");
    std::uint32_t ver, sd, cd, nb, n;
    if (not (detail::get(b, ver) and detail::get(b, sd) and detail::get(b, cd) and detail::get(b, nb) and detail::get(b, n)))
        return fail("blob short: header");
    if (ver != kArchiveVersion) return fail("version " + std::to_string(ver) + " != " + std::to_string(kArchiveVersion));
    if (int(sd) != shape_dim) return fail("shape_dim " + std::to_string(sd) + " != " + std::to_string(shape_dim));
    if (int(cd) != colour_dim) return fail("colour_dim " + std::to_string(cd) + " != " + std::to_string(colour_dim));
    if (int(nb) != p.n_bins) return fail("n_bins " + std::to_string(nb) + " != " + std::to_string(p.n_bins));
    if (n != recs.size()) return fail("record count: blob " + std::to_string(n) + ", index " + std::to_string(recs.size()));

    for (auto& r : recs)
    {
        r.atlas.shape_dim = shape_dim; r.atlas.colour_dim = colour_dim;
        if (not detail::get_vec(b, r.atlas.colour, colour_dim) or not detail::get(b, r.atlas.n_c))
            return fail("blob short: colour");
        r.atlas.bins.resize(nb);
        for (auto& bin : r.atlas.bins)
        {
            std::uint32_t np;
            if (not (detail::get(b, bin.n) and detail::get(b, bin.bearing) and detail::get(b, bin.stamp_ms)
                     and detail::get(b, np)))
                return fail("blob short: bin header");
            if (int(np) > p.poses_per_bin) return fail("poses per bin " + std::to_string(np) + " > poses_per_bin");
            bin.poses.resize(np);
            for (auto& ps : bin.poses)
            {
                float x, y;
                if (not (detail::get(b, x) and detail::get(b, y))) return fail("blob short: poses");
                ps = Eigen::Vector2f(x, y);
            }
            if (not detail::get_vec(b, bin.mean, shape_dim)) return fail("blob short: bin mean");
        }
    }
    out = std::move(recs);
    return true;
}

/// `key = value` lines; '#' starts a comment. Unknown keys and unparseable values are refused (loudly):
/// a misspelt constant silently keeping its default is the failure this loader exists to prevent.
inline bool load_params(const std::string& path, AtlasParams& p, std::string* why_not)
{
    auto fail = [&](std::string m) { if (why_not) *why_not = std::move(m); return false; };
    std::ifstream f(path);
    if (not f) return fail("cannot open " + path);
    AtlasParams q = p;
    auto sym = [](Eigen::Matrix2f& m, float v) { m(0, 1) = m(1, 0) = v; };
    std::string line;
    while (std::getline(f, line))
    {
        std::string_view s = detail::trim(line);
        if (s.empty() or s.front() == '#') continue;
        const auto eq = s.find('=');
        if (eq == std::string_view::npos) return fail("no '=' in line: " + line);
        const std::string key(detail::trim(s.substr(0, eq)));
        float v;
        if (not detail::parse(detail::trim(s.substr(eq + 1)), v)) return fail("unparseable value for " + key + ": " + line);
        if      (key == "ell_rad")          q.ell_rad = v;
        else if (key == "repeat_ell_rad")   q.repeat_ell_rad = v;
        else if (key == "repeat_ell_logr")  q.repeat_ell_logr = v;
        else if (key == "n0")               q.n0 = v;
        else if (key == "mu_own_d")         q.mu_own.x() = v;
        else if (key == "mu_own_c")         q.mu_own.y() = v;
        else if (key == "mu_bg_d")          q.mu_bg.x() = v;
        else if (key == "mu_bg_c")          q.mu_bg.y() = v;
        else if (key == "cov_own_dd")       q.cov_own(0, 0) = v;
        else if (key == "cov_own_dc")       sym(q.cov_own, v);
        else if (key == "cov_own_cc")       q.cov_own(1, 1) = v;
        else if (key == "cov_bg_dd")        q.cov_bg(0, 0) = v;
        else if (key == "cov_bg_dc")        sym(q.cov_bg, v);
        else if (key == "cov_bg_cc")        q.cov_bg(1, 1) = v;
        else if (key == "cov_n_dd")         q.cov_n(0, 0) = v;
        else if (key == "cov_n_dc")         sym(q.cov_n, v);
        else if (key == "cov_n_cc")         q.cov_n(1, 1) = v;
        else if (key == "lambda_nats")      q.lambda_nats = v;
        else if (key == "max_rate_hz")      q.max_rate_hz = v;
        else if (key == "pi_move")          q.pi_move = v;
        else if (key == "quality_px_scale") q.quality_px_scale = v;
        else return fail("unknown key: " + key);
    }
    p = q;
    return true;
}

} // namespace rc::atlas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) atlas_io_test.cpp -o /tmp/atlas_io_test && /tmp/atlas_io_test`
Expected: first line `locale: es_ES.UTF-8` (or the user's default), all `ok`, `atlas_io_test: PASS`.

- [ ] **Step 5: Cross-reference the episodic memory design**

In `common/view_atlas/VIEW_ATLAS.md` §5, after the `Identity` row's table, add:

```markdown
`object_uid` is also the key the episodic memory needs (`docs/superpowers/specs/2026-10-17-episodic-memory-design.md`
snapshots object beliefs keyed by object node): without it a resurrected object reads there as a new one.
That archive is the intended home of the future long-term object memory; the per-agent atlas archive must
not grow into a second long-term store.
```

- [ ] **Step 6: Run the common suite and commit**

Run: `bash common/run_tests.sh`
Expected: `common: all test binaries passed`.

```bash
git add common/view_atlas/atlas_io.h common/view_atlas/atlas_io_test.cpp common/view_atlas/VIEW_ATLAS.md
git commit -m "view_atlas: archive (text index + binary blob, locale-safe) and constants loader (+tests)"
```

---

### Task 4: DINOv3 crop encoder export and parity fixture

**Files:**
- Create: `retina/tools/export_dinov3_crop.py`
- Create (generated, not committed — the model is ~85 MB): `retina/models/dinov3/dinov3_vits16_crop224.onnx`
- Create (generated, committed — small): `common/view_atlas/tools/parity_fixture/` (`crop_00.png` … `crop_07.png`, `expected.f32`, `README.txt`)

**Interfaces:**
- Consumes: HF model `facebook/dinov3-vits16-pretrain-lvd1689m` (cached in `~/.cache/huggingface`); frames in `retina/etc/zed_orbits/`.
- Produces: ONNX file with input `pixel_values` float32 [1,3,224,224] (ImageNet-normalised RGB) and output `embedding` float32 [1,384] (L2-normalised); fixture of 8 RGB 224² PNG crops + `expected.f32` (8×384 little-endian float32, row-major).

- [ ] **Step 1: Write the export script**

Create `retina/tools/export_dinov3_crop.py`:

```python
#!/usr/bin/env python3
"""Export DINOv3 ViT-S/16 as the view-atlas crop encoder (spec: common/view_atlas/VIEW_ATLAS.md).

Output = L2-normalised mean of the L2-normalised PATCH tokens (CLS and the 4 register tokens skipped),
384-d, for a 224x224 ImageNet-normalised RGB crop. ViT-S/16, NOT S+: ORT's graph optimiser corrupted the
S+ export (memory: dinov3-vs-dinov2-place-descriptor).

Also writes a parity fixture for common/view_atlas/tools/encoder_parity.cpp: 8 background-blanked crops
(RGB PNG, already 224x224) and the torch embeddings, so the C++ encoder can be checked number for number.

    python3 retina/tools/export_dinov3_crop.py            # from active_inference/
"""
import argparse, os, sys
sys.modules["torchaudio"] = None
import numpy as np, cv2, torch, torch.nn as nn

HF_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
N_SKIP = 1 + 4                                   # CLS + registers
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

class CropEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.b = backbone
    def forward(self, pixel_values):
        h = self.b(pixel_values=pixel_values).last_hidden_state[:, N_SKIP:]
        h = nn.functional.normalize(h, dim=-1).mean(1)
        return nn.functional.normalize(h, dim=-1)

def to_tensor(rgb224):
    x = (rgb224.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())

def fixture_crops(orbit_dir, n=8):
    """Centre crops of the first n orbit frames with a disc 'mask' blanked outside — deterministic, no YOLO."""
    files = sorted(f for f in os.listdir(orbit_dir) if f.endswith(".jpg"))[: n * 50 : 50]
    out = []
    for f in files:
        bgr = cv2.imread(os.path.join(orbit_dir, f))
        h, w = bgr.shape[:2]; s = min(h, w)
        crop = bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2].copy()
        yy, xx = np.mgrid[:s, :s]; m = (yy - s / 2) ** 2 + (xx - s / 2) ** 2 < (0.4 * s) ** 2
        crop[~m] = (104, 116, 124)                # ImageNet-mean BGR
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        out.append(rgb)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="retina/models/dinov3/dinov3_vits16_crop224.onnx")
    ap.add_argument("--orbits", default="retina/etc/zed_orbits")
    ap.add_argument("--fixture", default="common/view_atlas/tools/parity_fixture")
    a = ap.parse_args()
    from transformers import AutoModel
    enc = CropEncoder(AutoModel.from_pretrained(HF_ID).eval()).eval()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.onnx.export(enc, to_tensor(np.zeros((224, 224, 3), np.uint8)), a.out, opset_version=17,
                      input_names=["pixel_values"], output_names=["embedding"], dynamic_axes=None)
    print("exported", a.out)

    crops = fixture_crops(a.orbits)
    os.makedirs(a.fixture, exist_ok=True)
    with torch.no_grad():
        ref = np.stack([enc(to_tensor(c))[0].numpy() for c in crops]).astype("<f4")
    for i, c in enumerate(crops):
        cv2.imwrite(os.path.join(a.fixture, f"crop_{i:02d}.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    ref.tofile(os.path.join(a.fixture, "expected.f32"))
    with open(os.path.join(a.fixture, "README.txt"), "w") as f:
        f.write(f"Parity fixture for encoder_parity.cpp. {len(crops)} RGB crops (stored as BGR PNG), 224x224,\n"
                f"already blanked; expected.f32 = {len(crops)}x384 little-endian float32 from torch ({HF_ID}).\n"
                f"Generated by retina/tools/export_dinov3_crop.py.\n")

    import onnxruntime as ort
    for level, name in [(ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "DISABLE_ALL"),
                        (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "ENABLE_ALL")]:
        so = ort.SessionOptions(); so.graph_optimization_level = level
        s = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
        got = np.stack([s.run(None, {"pixel_values": to_tensor(c).numpy()})[0][0] for c in crops])
        cos = (got * ref).sum(1)
        print(f"onnx {name:11s} vs torch: min cosine {cos.min():.6f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run (from `active_inference/`): `python3 retina/tools/export_dinov3_crop.py`
Expected: `exported retina/models/dinov3/dinov3_vits16_crop224.onnx`, then two lines `onnx DISABLE_ALL vs torch: min cosine 0.99999x` and `onnx ENABLE_ALL ...`. **Both must be ≥ 0.999.** If ENABLE_ALL is below and DISABLE_ALL is above, record that: Task 5's encoder must then default to `ORT_DISABLE_ALL` (set `EncoderConfig::graph_opt_all = false`).

- [ ] **Step 3: Commit the script and fixture (not the model)**

```bash
git add retina/tools/export_dinov3_crop.py common/view_atlas/tools/parity_fixture
git commit -m "view_atlas: DINOv3 ViT-S/16 crop-encoder export + parity fixture"
```

---

### Task 5: Frame history, crop + colour histogram, ONNX encoder, parity check

**Files:**
- Create: `common/view_atlas/frame_history.h`
- Create: `common/view_atlas/view_crop.h`
- Create: `common/view_atlas/view_crop_test.cpp`
- Create: `common/view_atlas/view_encoder.h`, `common/view_atlas/view_encoder.cpp`
- Create: `common/view_atlas/tools/encoder_parity.cpp`
- Create: `common/view_atlas/tools/build_tools.sh`

**Interfaces:**
- Consumes: Task 4's ONNX model path and fixture format.
- Produces (namespace `rc::atlas`):
  - `class FrameHistory { explicit FrameHistory(std::size_t capacity = 8); void push(std::uint64_t stamp_ms, const cv::Mat& bgr); const cv::Mat* at(std::uint64_t stamp_ms) const; std::size_t size() const; }`
  - `struct Crop { cv::Mat rgb224; cv::Mat bgr_box; cv::Mat mask_box; int mask_px = 0; float trunc_frac = 0.f; }`
  - `std::optional<Crop> make_crop(const cv::Mat& bgr, const std::vector<Eigen::Vector2f>& mask_px)`
  - `Eigen::VectorXf colour_hist(const cv::Mat& bgr_box, const cv::Mat& mask_box)` (64-d)
  - `struct EncoderConfig { std::string model_path; int threads = 2; bool graph_opt_all = true; }`
  - `class ViewEncoder { explicit ViewEncoder(EncoderConfig); bool ok() const; std::optional<Eigen::VectorXf> encode(const cv::Mat& rgb224) const; }`

- [ ] **Step 1: Write the failing test (OpenCV-only parts)**

Create `common/view_atlas/view_crop_test.cpp`:

```cpp
/*
 * view_crop_test.cpp — crop geometry, background blanking, colour histogram, frame history. OpenCV only.
 */
#include "frame_history.h"
#include "view_crop.h"

#include <cstdio>

using namespace rc::atlas;
static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}
static std::vector<Eigen::Vector2f> rect_px(int x0, int y0, int x1, int y1)
{
    std::vector<Eigen::Vector2f> v;
    for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) v.emplace_back(float(x), float(y));
    return v;
}

int main()
{
    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255));          // red background
    cv::rectangle(img, {600, 300}, {700, 400}, cv::Scalar(255, 0, 0), cv::FILLED);   // blue object

    std::printf("crop\n");
    auto c = make_crop(img, rect_px(600, 300, 700, 400));
    check(c.has_value(), "a normal mask yields a crop");
    check(c->rgb224.size() == cv::Size(224, 224) and c->rgb224.type() == CV_8UC3, "crop is 224x224 RGB");
    check(c->mask_px == 100 * 100, "mask pixel count");
    const cv::Vec3b corner = c->rgb224.at<cv::Vec3b>(2, 2), centre = c->rgb224.at<cv::Vec3b>(112, 112);
    check(corner == cv::Vec3b(124, 116, 104), "background blanked to ImageNet mean (RGB order)");
    check(centre[2] > 200 and centre[0] < 50, "object pixels kept, converted to RGB (blue in channel 2)");
    check(c->trunc_frac == 0.f, "interior mask is not truncated");

    auto e = make_crop(img, rect_px(1200, 300, 1280, 400));
    check(e.has_value() and e->rgb224.size() == cv::Size(224, 224), "mask at the right border: clipped crop");
    check(e->trunc_frac > 0.f, "border mask reports truncation");
    check(not make_crop(img, {}).has_value(), "empty mask: no crop");
    auto one = make_crop(img, {Eigen::Vector2f(10.f, 10.f)});
    check(one.has_value() and one->mask_px == 1, "1-pixel mask: crop, no throw");
    check(not make_crop(img, {Eigen::Vector2f(-5.f, 3.f), Eigen::Vector2f(5000.f, 3.f)}).has_value(),
          "all mask pixels outside the image: no crop");

    std::printf("colour histogram\n");
    const Eigen::VectorXf h = colour_hist(c->bgr_box, c->mask_box);
    check(h.size() == 64, "64 bins");
    check(std::fabs(h.norm() - 1.f) < 1e-5f, "unit length");
    int argmax; h.maxCoeff(&argmax);
    check(argmax == (120 * 16 / 180) * 4 + 255 * 4 / 256, "dominant bin is blue (hue 120, saturation 255)");
    check(colour_hist(c->bgr_box, cv::Mat::zeros(c->bgr_box.size(), CV_8U)).norm() == 0.f, "empty mask: zero histogram");

    std::printf("frame history\n");
    FrameHistory fh(3);
    for (std::uint64_t s = 1; s <= 4; ++s) fh.push(s * 100, img);
    check(fh.size() == 3, "capacity respected");
    check(fh.at(100) == nullptr, "oldest frame evicted");
    check(fh.at(400) != nullptr, "newest frame present");
    check(fh.at(250) == nullptr, "a stamp never pushed: nothing");
    cv::Mat src = img.clone(); fh.push(500, src); src.setTo(cv::Scalar(0, 0, 0));
    check(fh.at(500)->at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 0, 255), "push deep-copies (caller may reuse its buffer)");

    std::printf("\nview_crop_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3 opencv4) view_crop_test.cpp -o /tmp/view_crop_test $(pkg-config --libs opencv4)`
Expected: compile error `frame_history.h: No such file or directory`.

- [ ] **Step 3: Implement `frame_history.h` and `view_crop.h`**

Create `common/view_atlas/frame_history.h`:

```cpp
/*
 * frame_history.h — the last few ZED BGR frames by capture stamp, so a mask can be cropped from the frame
 * it was computed on. Fed from the agent's RgbIngestor pump; read on the same (compute) thread. push()
 * deep-copies: the ingestor reuses its buffer, and a shallow cv::Mat copy across that is a heap smash.
 */
#pragma once

#include <cstdint>
#include <deque>
#include <utility>

#include <opencv2/core.hpp>

namespace rc::atlas
{
class FrameHistory
{
public:
    explicit FrameHistory(std::size_t capacity = 8) : cap_(capacity) {}
    void push(std::uint64_t stamp_ms, const cv::Mat& bgr)
    {
        if (bgr.empty()) return;
        q_.emplace_back(stamp_ms, bgr.clone());
        while (q_.size() > cap_) q_.pop_front();
    }
    /// Exact stamp match (the producer stamps masks with the RGB frame's capture stamp); nullptr if gone.
    [[nodiscard]] const cv::Mat* at(std::uint64_t stamp_ms) const
    {
        for (const auto& [s, m] : q_) if (s == stamp_ms) return &m;
        return nullptr;
    }
    [[nodiscard]] std::size_t size() const noexcept { return q_.size(); }
private:
    std::size_t cap_;
    std::deque<std::pair<std::uint64_t, cv::Mat>> q_;
};
} // namespace rc::atlas
```

Create `common/view_atlas/view_crop.h`:

```cpp
/*
 * view_crop.h — the view-atlas crop and colour histogram (spec: VIEW_ATLAS.md §2). Must match
 * common/view_atlas/tools/build_dataset.py::make_crop exactly, or the fitted constants describe a
 * different descriptor than the one the agent computes.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rc::atlas
{
struct Crop
{
    cv::Mat rgb224;    // 224x224 RGB, background = ImageNet mean
    cv::Mat bgr_box;   // the mask's bounding box, original pixels (for the colour histogram)
    cv::Mat mask_box;  // CV_8U 0/255, same size as bgr_box
    int     mask_px = 0;
    float   trunc_frac = 0.f;   // fraction of mask pixels on the image border row/column
};

inline std::optional<Crop> make_crop(const cv::Mat& bgr, const std::vector<Eigen::Vector2f>& mask_px)
{
    if (bgr.empty() or mask_px.empty()) return std::nullopt;
    cv::Mat mask = cv::Mat::zeros(bgr.size(), CV_8U);
    int x0 = bgr.cols, y0 = bgr.rows, x1 = -1, y1 = -1, n = 0, border = 0;
    for (const auto& p : mask_px)
    {
        const int x = int(std::lround(p.x())), y = int(std::lround(p.y()));
        if (x < 0 or y < 0 or x >= bgr.cols or y >= bgr.rows) continue;
        if (mask.at<std::uint8_t>(y, x) == 0) { ++n; if (x == 0 or y == 0 or x == bgr.cols - 1 or y == bgr.rows - 1) ++border; }
        mask.at<std::uint8_t>(y, x) = 255;
        x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
    }
    if (n == 0) return std::nullopt;
    Crop c;
    c.mask_px = n;
    c.trunc_frac = float(border) / float(n);
    const cv::Rect box(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    c.bgr_box = bgr(box).clone();
    c.mask_box = mask(box).clone();

    const float cx = 0.5f * float(x0 + x1 + 1), cy = 0.5f * float(y0 + y1 + 1);
    const float s = 1.15f * float(std::max(box.width, box.height));
    const int X1 = int(std::max(cx - s / 2, 0.f)), Y1 = int(std::max(cy - s / 2, 0.f));
    const int X2 = int(std::min(cx + s / 2, float(bgr.cols))), Y2 = int(std::min(cy + s / 2, float(bgr.rows)));
    const cv::Rect sq(X1, Y1, std::max(X2 - X1, 1), std::max(Y2 - Y1, 1));
    cv::Mat crop = bgr(sq).clone();
    crop.setTo(cv::Scalar(104, 116, 124), mask(sq) == 0);
    cv::Mat r;
    cv::resize(crop, r, {224, 224}, 0, 0, cv::INTER_AREA);
    cv::cvtColor(r, c.rgb224, cv::COLOR_BGR2RGB);
    return c;
}

/// 16 hue x 4 saturation over mask pixels, sqrt, L2-normalised; zeros when the mask is empty.
inline Eigen::VectorXf colour_hist(const cv::Mat& bgr_box, const cv::Mat& mask_box)
{
    Eigen::VectorXf h = Eigen::VectorXf::Zero(64);
    if (bgr_box.empty()) return h;
    cv::Mat hsv;
    cv::cvtColor(bgr_box, hsv, cv::COLOR_BGR2HSV);
    for (int y = 0; y < hsv.rows; ++y)
        for (int x = 0; x < hsv.cols; ++x)
            if (mask_box.at<std::uint8_t>(y, x))
            {
                const auto v = hsv.at<cv::Vec3b>(y, x);
                h[(v[0] * 16 / 180) * 4 + v[1] * 4 / 256] += 1.f;
            }
    h = h.cwiseSqrt();
    const float nrm = h.norm();
    if (nrm > 0.f) h /= nrm;
    return h;
}
} // namespace rc::atlas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd common/view_atlas && g++ -std=c++23 -O1 $(pkg-config --cflags eigen3 opencv4) view_crop_test.cpp -o /tmp/view_crop_test $(pkg-config --libs opencv4) && /tmp/view_crop_test`
Expected: all `ok`, `view_crop_test: PASS`.

- [ ] **Step 5: Implement the ONNX encoder**

Create `common/view_atlas/view_encoder.h`:

```cpp
/*
 * view_encoder.h — DINOv3 ViT-S/16 crop encoder, ONNX Runtime CPU (spec: VIEW_ATLAS.md §2). One instance
 * per agent, owned by the agent's encoder worker thread (ORT sessions are not shared across agents).
 */
#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace Ort { class Env; class Session; }

namespace rc::atlas
{
struct EncoderConfig
{
    std::string model_path = "models/dinov3/dinov3_vits16_crop224.onnx";
    int  threads = 2;
    bool graph_opt_all = true;   // Task 4 decides: false if ENABLE_ALL broke parity
};

class ViewEncoder
{
public:
    explicit ViewEncoder(EncoderConfig cfg);
    ~ViewEncoder();
    [[nodiscard]] bool ok() const noexcept { return session_ != nullptr; }
    /// rgb224: CV_8UC3 RGB 224x224 (make_crop().rgb224). Returns the unit 384-d embedding, or nothing.
    [[nodiscard]] std::optional<Eigen::VectorXf> encode(const cv::Mat& rgb224) const;
private:
    EncoderConfig cfg_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};
} // namespace rc::atlas
```

Create `common/view_atlas/view_encoder.cpp`:

```cpp
#include "view_encoder.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <print>
#include <vector>

namespace rc::atlas
{
ViewEncoder::ViewEncoder(EncoderConfig cfg) : cfg_(std::move(cfg))
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ViewEncoder");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(cfg_.threads);
        opts.SetGraphOptimizationLevel(cfg_.graph_opt_all ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                                          : GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, cfg_.model_path.c_str(), opts);
    }
    catch (const Ort::Exception& e)
    {
        std::println("[ViewEncoder] cannot load {}: {}", cfg_.model_path, e.what());
        session_.reset();
    }
}

ViewEncoder::~ViewEncoder() = default;

std::optional<Eigen::VectorXf> ViewEncoder::encode(const cv::Mat& rgb224) const
{
    if (not session_ or rgb224.rows != 224 or rgb224.cols != 224 or rgb224.type() != CV_8UC3) return std::nullopt;
    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> t(3 * 224 * 224);
    for (int y = 0; y < 224; ++y)
        for (int x = 0; x < 224; ++x)
        {
            const auto p = rgb224.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c)
                t[std::size_t(c) * 224 * 224 + std::size_t(y) * 224 + std::size_t(x)] = (float(p[c]) / 255.f - mean[c]) / stdv[c];
        }
    try
    {
        const std::array<std::int64_t, 4> shape{1, 3, 224, 224};
        const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, t.data(), t.size(), shape.data(), shape.size());
        const char* in_names[] = {"pixel_values"};
        const char* out_names[] = {"embedding"};
        auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        const float* o = outs[0].GetTensorData<float>();
        Eigen::VectorXf v = Eigen::Map<const Eigen::VectorXf>(o, 384);
        const float n = v.norm();
        if (not (n > 0.f)) return std::nullopt;
        return v / n;
    }
    catch (const Ort::Exception& e)
    {
        std::println("[ViewEncoder] run failed: {}", e.what());
        return std::nullopt;
    }
}
} // namespace rc::atlas
```

- [ ] **Step 6: Write the parity tool and build script**

Create `common/view_atlas/tools/encoder_parity.cpp`:

```cpp
// encoder_parity — the C++ encoder must reproduce torch on the fixture crops (cosine >= 0.999).
// usage: encoder_parity <model.onnx> <fixture_dir> [--no-opt]
#include "../view_encoder.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.onnx fixture_dir [--no-opt]\n", argv[0]); return 2; }
    rc::atlas::EncoderConfig cfg; cfg.model_path = argv[1];
    cfg.graph_opt_all = not (argc > 3 and std::string(argv[3]) == "--no-opt");
    rc::atlas::ViewEncoder enc(cfg);
    if (not enc.ok()) return 2;
    const std::string dir = argv[2];
    std::ifstream f(dir + "/expected.f32", std::ios::binary | std::ios::ate);
    if (not f) { std::printf("cannot open %s/expected.f32\n", dir.c_str()); return 2; }
    const auto bytes = std::size_t(f.tellg());
    f.seekg(0);
    std::vector<float> ref(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(bytes));
    const std::size_t n = ref.size() / 384;
    float worst = 1.f;
    for (std::size_t i = 0; i < n; ++i)
    {
        char name[64]; std::snprintf(name, sizeof name, "/crop_%02zu.png", i);
        cv::Mat bgr = cv::imread(dir + name), rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        const auto v = enc.encode(rgb);
        if (not v) { std::printf("crop %zu: encode failed\n", i); return 1; }
        const Eigen::Map<const Eigen::VectorXf> r(ref.data() + i * 384, 384);
        const float cos = v->dot(r);
        worst = std::min(worst, cos);
        std::printf("crop %zu: cosine %.6f\n", i, cos);
    }
    std::printf("encoder_parity (%s): worst cosine %.6f -> %s\n", cfg.graph_opt_all ? "ENABLE_ALL" : "DISABLE_ALL",
                worst, worst >= 0.999f ? "PASS" : "FAIL");
    return worst >= 0.999f ? 0 : 1;
}
```

Create `common/view_atlas/tools/build_tools.sh`:

```bash
#!/usr/bin/env bash
# Build the ORT-linked view_atlas tools (not part of run_tests.sh: they need ONNX Runtime and data).
set -euo pipefail
cd "$(dirname "$0")"
ORT=${ONNXRUNTIME_ROOT:-/usr/local/onnxruntime}
CFLAGS="-std=c++23 -O2 $(pkg-config --cflags eigen3 opencv4) -I$ORT/include"
LIBS="$(pkg-config --libs opencv4) -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib"
mkdir -p bin
g++ $CFLAGS encoder_parity.cpp ../view_encoder.cpp -o bin/encoder_parity $LIBS
if [ -f atlas_replay.cpp ]; then g++ $CFLAGS atlas_replay.cpp -o bin/atlas_replay $LIBS; fi
echo "built: $(ls bin)"
```

- [ ] **Step 7: Build and run the parity check**

Run: `bash common/view_atlas/tools/build_tools.sh && common/view_atlas/tools/bin/encoder_parity retina/models/dinov3/dinov3_vits16_crop224.onnx common/view_atlas/tools/parity_fixture`
Expected: 8 lines `crop i: cosine 0.99999x` and `encoder_parity (ENABLE_ALL): worst cosine ... -> PASS`. If it FAILS, rerun with `--no-opt`; if that passes, change `EncoderConfig::graph_opt_all` default to `false` and say why in its comment. If both fail, check the resize/normalisation first (the fixture crops are already 224², so the only preprocessing left is `/255`, mean/std, NCHW order).

- [ ] **Step 8: Add `bin/` to .gitignore, run the common suite, commit**

```bash
echo "bin/" > common/view_atlas/tools/.gitignore
bash common/run_tests.sh
git add common/view_atlas/frame_history.h common/view_atlas/view_crop.h common/view_atlas/view_crop_test.cpp \
        common/view_atlas/view_encoder.h common/view_atlas/view_encoder.cpp \
        common/view_atlas/tools/encoder_parity.cpp common/view_atlas/tools/build_tools.sh common/view_atlas/tools/.gitignore
git commit -m "view_atlas: frame history, crop + colour histogram (+tests), ONNX CPU encoder, parity check"
```

Expected from `run_tests.sh`: `common: all test binaries passed` (it now also runs `view_crop_test`).

---

### Task 6: Labelled views dataset and fitted constants

**Files:**
- Create: `common/view_atlas/tools/build_dataset.py`
- Create: `common/view_atlas/tools/fit_constants.py`
- Create (generated, committed): `common/view_atlas/view_atlas_constants.txt`
- Modify: `common/view_atlas/VIEW_ATLAS.md` (§3.5: n0 = 1 "worth one view"; ℓ_r derived rule)
- Modify: `.gitignore` (add `retina/etc/zed_orbits/` and `retina/etc/view_atlas_dataset/` if not covered)

**Interfaces:**
- Consumes: `retina/etc/zed_frames/` (apartment dump, `poses.csv`, Webots GT), `retina/etc/zed_orbits/` (orbit recording, `poses.csv` + `furniture_lap{0,1}.json`), `room_concept/estimation_scripts/apartment/furniture.json`, YOLOE (`retina/models/yolo26/_yoloe_work/yoloe-26x-seg.pt`, vocab from `retina/tools/export_yoloe_household.py`), the HF DINOv3 model.
- Produces: dataset directory `retina/etc/view_atlas_dataset/` with
  - `views.csv` columns: `view_id,dataset,stamp_ms,lap,inst,inst_name,cls,cam_x,cam_y,obj_x,obj_y,obj_yaw,range_m,mask_px,trunc_frac`
  - `shape.f32` (N×384), `colour.f32` (N×64), little-endian float32, row = `view_id`
- Produces: `view_atlas_constants.txt` with keys `ell_rad, repeat_ell_rad, repeat_ell_logr, mu_own_d, mu_own_c, mu_bg_d, mu_bg_c, cov_own_dd, cov_own_dc, cov_own_cc, cov_bg_dd, cov_bg_dc, cov_bg_cc, cov_n_dd, cov_n_dc, cov_n_cc` (exactly the names `load_params` accepts).

- [ ] **Step 1: Write the dataset builder**

Create `common/view_atlas/tools/build_dataset.py`:

```python
#!/usr/bin/env python3
"""Labelled views dataset for the view atlas: YOLOE masks -> ground-truth instance by casting the mask's
bottom-contact pixel onto the floor into the Webots furniture footprints (labeller from the 2026-10-17
spike, colour-verified), then the SAME crop + DINOv3 + colour histogram the C++ module computes
(common/view_atlas/view_crop.h). Two sources:
  zed_frames : retina ZedDump of an apartment tour (gt_* = robot, ZED 0.075 m behind along forward)
  zed_orbits : orbit recorder (same CSV convention; lap 1 moves the yellow chair)
usage (from active_inference/):
  python3 common/view_atlas/tools/build_dataset.py --out retina/etc/view_atlas_dataset
"""
import argparse, csv, importlib.util, json, math, os, sys
sys.modules["torchaudio"] = None
import numpy as np, pandas as pd, cv2, torch
from shapely.geometry import shape, Point

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RETINA = os.path.join(ROOT, "retina")
F, CX, CY, CAM_Z, MOUNT_Y = 448.0, 640.0, 360.0, 0.9775, -0.075
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
OK = {"RoundTable": {"table", "dining table", "coffee table", "desk"}, "Table": {"table", "dining table", "coffee table", "desk"},
      "Desk": {"desk", "table", "workbench"}, "desk": {"desk", "table", "workbench"},
      "Chair": {"chair", "stool", "armchair"}, "chair": {"chair", "stool", "armchair"}, "Fridge": {"refrigerator"},
      "Worktop": {"countertop", "kitchen island", "cabinet", "shelf", "sink", "stove", "oven", "dishwasher"},
      "BookshelfCubes": {"bookshelf", "shelf", "cube shelf", "cabinet"}, "Pillar": {"column"}, "Heater": {"radiator"}}
# Object yaw (world) per lap: only the chairs' yaw matters (the only object that moves and turns);
# static objects keep yaw 0 because only bearing DIFFERENCES within one object are used for them.
YAW = {("Chair", 0): 1.0472, ("Chair", 1): math.pi / 2, ("chair(1)", 0): -2.8797953, ("chair(1)", 1): -2.8797953}

def ok_labels(n): return next((v for k, v in OK.items() if n.startswith(k)), set())
def cls_of(n):
    for k, c in [("RoundTable", "table"), ("Table", "table"), ("Desk", "desk"), ("desk", "desk"), ("Chair", "chair"),
                 ("chair", "chair"), ("Fridge", "refrigerator"), ("Worktop", "worktop"), ("Bookshelf", "bookshelf"),
                 ("Pillar", "pillar"), ("Heater", "heater")]:
        if n.startswith(k): return c
    return n.lower()

def make_crop(bgr, mask):
    """Mirror of view_crop.h::make_crop. mask: bool HxW."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return None
    H, W = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    border = np.count_nonzero((xs == 0) | (ys == 0) | (xs == W - 1) | (ys == H - 1))
    cx, cy = 0.5 * (x0 + x1 + 1), 0.5 * (y0 + y1 + 1)
    s = 1.15 * max(x1 - x0 + 1, y1 - y0 + 1)
    X1, Y1 = int(max(cx - s / 2, 0.0)), int(max(cy - s / 2, 0.0))
    X2, Y2 = int(min(cx + s / 2, float(W))), int(min(cy + s / 2, float(H)))
    X2, Y2 = max(X2, X1 + 1), max(Y2, Y1 + 1)
    crop = bgr[Y1:Y2, X1:X2].copy(); crop[~mask[Y1:Y2, X1:X2]] = (104, 116, 124)
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    box, mbox = bgr[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]
    return rgb, box, mbox, len(xs), border / len(xs)

def colour_hist(box, mbox):
    """Mirror of view_crop.h::colour_hist."""
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)[mbox]
    h = np.zeros(64, np.float32)
    if len(hsv): np.add.at(h, (hsv[:, 0].astype(int) * 16 // 180) * 4 + hsv[:, 1].astype(int) * 4 // 256, 1.0)
    h = np.sqrt(h); n = np.linalg.norm(h)
    return h / n if n > 0 else h

def sources(a):
    yield "frames", os.path.join(RETINA, "etc", "zed_frames"), {0: os.path.join(ROOT, "room_concept/estimation_scripts/apartment/furniture.json")}
    o = os.path.join(RETINA, "etc", "zed_orbits")
    yield "orbits", o, {0: os.path.join(o, "furniture_lap0.json"), 1: os.path.join(o, "furniture_lap1.json")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    spec = importlib.util.spec_from_file_location("exp", os.path.join(RETINA, "tools", "export_yoloe_household.py"))
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp); VOCAB = exp.VOCAB
    from ultralytics import YOLOE
    from transformers import AutoModel
    cwd = os.getcwd(); os.chdir(os.path.join(RETINA, "models/yolo26/_yoloe_work"))
    yolo = YOLOE("yoloe-26x-seg.pt"); yolo.set_classes(VOCAB); os.chdir(cwd)
    dino = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").eval().cuda().half()

    rows, S, C = [], [], []
    for ds, d, furn_files in sources(a):
        poses = pd.read_csv(os.path.join(d, "poses.csv"))
        poses = poses[[os.path.exists(os.path.join(d, f)) for f in poses.file]].reset_index(drop=True)
        if "lap" not in poses: poses["lap"] = 0
        if a.limit: poses = poses.iloc[:a.limit]
        furn = {lap: [f for f in json.load(open(p)) if ok_labels(f["name"])] for lap, p in furn_files.items()}
        names = [f["name"] for f in furn[0]]
        polys = {lap: [shape(f["geometry"]) for f in fs] for lap, fs in furn.items()}
        for fi, r in poses.iterrows():
            img = cv2.imread(os.path.join(d, r.file)); lap = int(r.lap)
            h = -r.gt_angle; fwd = np.array([math.cos(h), math.sin(h)]); right = np.array([math.sin(h), -math.cos(h)])
            cam = np.array([r.gt_x, r.gt_y]) + MOUNT_Y * fwd
            res = yolo.predict(img, conf=0.25, verbose=False, retina_masks=True)[0]
            if res.masks is None: continue
            for k, m in enumerate(res.masks.data.cpu().numpy() > 0.5):
                lab = VOCAB[int(res.boxes.cls[k])]; ys, xs = np.nonzero(m)
                if len(ys) < 50: continue
                vb = int(np.percentile(ys, 99))
                if vb >= img.shape[0] - 4: continue                          # bottom cut: contact pixel unseen
                ub = float(np.median(xs[ys >= vb - 6])); dz = -(vb - CY) / F
                if dz >= -1e-3: continue                                     # bottom above the horizon
                p = cam + (CAM_Z / -dz) * ((ub - CX) / F * right + fwd)
                cand = [j for j, P in enumerate(polys[lap]) if lab in ok_labels(names[j]) and P.distance(Point(p)) < 0.25]
                if len(cand) != 1: continue
                j = cand[0]; nm = names[j]
                cr = make_crop(img, m)
                if cr is None: continue
                rgb, box, mbox, npx, trunc = cr
                x = torch.from_numpy((((rgb.astype(np.float32) / 255) - MEAN) / STD).transpose(2, 0, 1)[None].copy()).cuda().half()
                with torch.no_grad():
                    t = dino(pixel_values=x).last_hidden_state[0, 5:].float()
                    v = torch.nn.functional.normalize(torch.nn.functional.normalize(t, dim=-1).mean(0), dim=0).cpu().numpy()
                ctr = np.array(polys[lap][j].centroid.coords[0])
                rows.append(dict(view_id=len(rows), dataset=ds, stamp_ms=int(r.stamp_ms), lap=lap, inst=f"{ds}:{nm}" if nm not in ("Chair", "chair(1)") else nm,
                                 inst_name=nm, cls=cls_of(nm), cam_x=cam[0], cam_y=cam[1], obj_x=ctr[0], obj_y=ctr[1],
                                 obj_yaw=YAW.get((nm, lap), 0.0), range_m=float(np.linalg.norm(ctr - cam)), mask_px=npx, trunc_frac=trunc))
                S.append(v.astype("<f4")); C.append(colour_hist(box, mbox).astype("<f4"))
            if fi % 200 == 0: print(f"{ds} {fi}/{len(poses)} views={len(rows)}", flush=True)
    with open(os.path.join(a.out, "views.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.stack(S).tofile(os.path.join(a.out, "shape.f32")); np.stack(C).tofile(os.path.join(a.out, "colour.f32"))
    print("views:", len(rows)); print(pd.DataFrame(rows).groupby(["dataset", "inst"]).size().to_string())

if __name__ == "__main__":
    main()
```

Note the instance key: both chairs keep the SAME key (`Chair`, `chair(1)`) across the two datasets because they are the same physical chairs in the same world; every other object is prefixed by its dataset so the apartment and orbit runs never mix static furniture identities by accident.

- [ ] **Step 2: Run it (GPU, ~10–15 min)**

Run (from `active_inference/`): `nvidia-smi --query-gpu=memory.used --format=csv && python3 common/view_atlas/tools/build_dataset.py --out retina/etc/view_atlas_dataset`
Expected: progress lines, then `views: N` with N ≈ 2600–2800 and a per-instance table that includes `Chair` ≥ 400 and `chair(1)` ≥ 300 views. If the chair counts are far lower, open 10 random rows' frames and check the labeller before going on.

- [ ] **Step 3: Write the constants fitter (with self-test)**

Create `common/view_atlas/tools/fit_constants.py`:

```python
#!/usr/bin/env python3
"""Fit the view-atlas constants (VIEW_ATLAS.md §3.5) from a views dataset (build_dataset.py).

For every query view q of an object and every other bin of that SAME object (views >= 10 s apart from q),
build the bin prototype and record (gap D, s_shape, s_colour). Fit
    mu_d(D) = mu_bg_d + (mu0_d - mu_bg_d) exp(-D^2 / 2 ell^2)
and take (mu_own, cov_own) from the pairs with D < 15 deg. The background (mu_bg, cov_bg) comes from the
same computation against OTHER objects of the SAME class at the same object-frame bearing.
repeat_ell_rad is DERIVED, not fitted: the angular step at which the expected shape similarity falls by one
per-view sd, i.e. views closer than that cannot be told from a repeat look. repeat_ell_logr likewise from the
similarity drop with |log range ratio| at near-equal bearing. cov_n = cov_own (per-view noise dominates).
    python3 fit_constants.py <dataset_dir> <out.txt>
    python3 fit_constants.py --selftest
"""
import argparse, datetime, math, os, sys
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

NB = 12; STEP = 2 * math.pi / NB
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

def relax(D, mu0, mubg, ell): return mubg + (mu0 - mubg) * np.exp(-D ** 2 / (2 * ell ** 2))

def load(d):
    v = pd.read_csv(os.path.join(d, "views.csv"))
    n = len(v)
    S = np.fromfile(os.path.join(d, "shape.f32"), "<f4").reshape(n, 384)
    C = np.fromfile(os.path.join(d, "colour.f32"), "<f4").reshape(n, 64)
    v["bearing"] = [wrap(math.atan2(r.cam_y - r.obj_y, r.cam_x - r.obj_x) - r.obj_yaw) for r in v.itertuples()]
    return v, S, C

def bin_protos(v, S, C, idx, exclude_t):
    """Per bin of one object's views (excluding views within 10 s of exclude_t): (mean bearing, shape proto)."""
    out = {}
    keep = idx[np.abs(v.stamp_ms.values[idx] / 1000 - exclude_t) >= 10]
    b = ((v.bearing.values[keep] + math.pi) // STEP).astype(int) % NB
    for k in np.unique(b):
        m = keep[b == k]; s = S[m].sum(0)
        out[k] = (math.atan2(np.sin(v.bearing.values[m]).sum(), np.cos(v.bearing.values[m]).sum()), s / np.linalg.norm(s),
                  C[m].sum(0) / np.linalg.norm(C[m].sum(0)))
    return out

def pairs(v, S, C, rng, max_q=4000):
    own, bg = [], []
    by_inst = {i: np.where(v.inst.values == i)[0] for i in v.inst.unique()}
    qs = rng.permutation(len(v))[:max_q]
    for q in qs:
        i, c = v.inst.values[q], v.cls.values[q]; tq = v.stamp_ms.values[q] / 1000
        for k, (bb, ps, pc) in bin_protos(v, S, C, by_inst[i], tq).items():
            own.append((abs(wrap(v.bearing.values[q] - bb)), S[q] @ ps, C[q] @ pc))
        for j, idx in by_inst.items():
            if j == i or v.cls.values[idx[0]] != c: continue
            for k, (bb, ps, pc) in bin_protos(v, S, C, idx, -1e9).items():
                D = abs(wrap(v.bearing.values[q] - bb))
                if D < math.radians(15): bg.append((D, S[q] @ ps, C[q] @ pc))
    return np.array(own), np.array(bg)

def fit(own, bg, logr_pairs=None):
    p, _ = curve_fit(relax, own[:, 0], own[:, 1], p0=[0.95, 0.85, 0.8], bounds=([0, -1, 0.05], [1, 1, math.pi]))
    mu0, mubg_d_curve, ell = p
    near = own[own[:, 0] < math.radians(15)]
    mu_own = near[:, 1:].mean(0); cov_own = np.cov(near[:, 1:].T)
    mu_bg = bg[:, 1:].mean(0); cov_bg = np.cov(bg[:, 1:].T)
    sd = math.sqrt(cov_own[0, 0]); drop = max(mu0 - mubg_d_curve, 1e-6)
    repeat = ell * math.sqrt(-2 * math.log(max(1 - sd / drop, 1e-6)))
    # small-step model s(x) = mu0 - a x^2 in x = |log range ratio|; a repeat is a step whose drop a x^2 < sd
    if logr_pairs is None or len(logr_pairs) < 20:
        repeat_logr = 0.2                                     # too few same-bearing pairs: keep the default
    else:
        a = max(np.mean(mu0 - logr_pairs[:, 1]), 1e-6) / max(np.mean(logr_pairs[:, 0] ** 2), 1e-6)
        repeat_logr = float(math.sqrt(sd / a))
    return dict(ell_rad=ell, repeat_ell_rad=repeat, repeat_ell_logr=repeat_logr,
                mu_own_d=mu_own[0], mu_own_c=mu_own[1], mu_bg_d=mu_bg[0], mu_bg_c=mu_bg[1],
                cov_own_dd=cov_own[0, 0], cov_own_dc=cov_own[0, 1], cov_own_cc=cov_own[1, 1],
                cov_bg_dd=cov_bg[0, 0], cov_bg_dc=cov_bg[0, 1], cov_bg_cc=cov_bg[1, 1],
                cov_n_dd=cov_own[0, 0], cov_n_dc=cov_own[0, 1], cov_n_cc=cov_own[1, 1])

def logr_pairs(v, S):
    """Same object, near-equal bearing (< 5 deg), different range: (|log r ratio|, shape similarity)."""
    out = []
    for i in v.inst.unique():
        idx = np.where(v.inst.values == i)[0]
        for a_ in idx[:300]:
            d = np.abs([wrap(v.bearing.values[a_] - v.bearing.values[b]) for b in idx])
            for b in idx[(d < math.radians(5)) & (idx != a_)][:5]:
                out.append((abs(math.log(v.range_m.values[a_] / v.range_m.values[b])), S[a_] @ S[b]))
    return np.array(out)

def selftest():
    rng = np.random.default_rng(0); truth = dict(mu0=0.95, mubg=0.86, ell=0.85)
    D = rng.uniform(0, math.pi, 4000)
    sd = relax(D, truth["mu0"], truth["mubg"], truth["ell"]) + rng.normal(0, 0.02, D.size)
    sc = 0.97 + rng.normal(0, 0.02, D.size)
    own = np.c_[D, sd, sc]; bg = np.c_[rng.uniform(0, 0.25, 2000), 0.86 + rng.normal(0, 0.04, 2000), 0.5 + rng.normal(0, 0.2, 2000)]
    k = fit(own, bg)
    ok = abs(k["ell_rad"] - truth["ell"]) / truth["ell"] < 0.1 and abs(k["mu_bg_d"] - 0.86) < 0.01 and abs(k["mu_own_c"] - 0.97) < 0.01
    print(f"selftest: ell {k['ell_rad']:.3f} (truth 0.85), mu_bg_d {k['mu_bg_d']:.3f}, mu_own_c {k['mu_own_c']:.3f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    v, S, C = load(a.dataset); rng = np.random.default_rng(0)
    own, bg = pairs(v, S, C, rng)
    k = fit(own, bg, logr_pairs(v, S))
    with open(a.out, "w") as f:
        f.write(f"# view_atlas constants, fitted {datetime.date.today()} by fit_constants.py\n"
                f"# dataset: {os.path.abspath(a.dataset)} ({len(v)} views, {v.inst.nunique()} objects)\n"
                f"# pairs: own {len(own)}, background {len(bg)}\n")
        for key, val in k.items(): f.write(f"{key} = {val:.9g}\n")
    print(open(a.out).read())

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the self-test, then fit**

Run: `python3 common/view_atlas/tools/fit_constants.py --selftest`
Expected: `selftest: ell 0.8xx (truth 0.85), ... -> PASS`.

Run: `python3 common/view_atlas/tools/fit_constants.py retina/etc/view_atlas_dataset common/view_atlas/view_atlas_constants.txt`
Expected: the file printed; sanity ranges from the spike — `ell_rad` 0.6–1.2, `mu_own_d` 0.9–0.98, `mu_bg_d` 0.8–0.9, `mu_own_c` > 0.85, `mu_bg_c` < `mu_own_c`. A value outside those ranges is a finding to report, not something to tune away.

- [ ] **Step 5: Check the C++ loader accepts the file**

Run:
```bash
cat > /tmp/load_check.cpp <<'EOF'
#include "atlas_io.h"
#include <clocale>
#include <cstdio>
int main(int, char** argv) {
    std::setlocale(LC_ALL, "");
    rc::atlas::AtlasParams p; std::string why;
    if (not rc::atlas::load_params(argv[1], p, &why)) { std::printf("REFUSED: %s\n", why.c_str()); return 1; }
    std::printf("loaded: ell_rad=%g mu_own=(%g,%g) mu_bg=(%g,%g)\n", p.ell_rad, p.mu_own.x(), p.mu_own.y(), p.mu_bg.x(), p.mu_bg.y());
}
EOF
g++ -std=c++23 -I common/view_atlas $(pkg-config --cflags eigen3) /tmp/load_check.cpp -o /tmp/load_check && /tmp/load_check common/view_atlas/view_atlas_constants.txt
```
Expected: `loaded: ...` with the same numbers the fitter printed.

- [ ] **Step 6: Update the spec's constants table**

In `VIEW_ATLAS.md` §3.5 replace the `N₀` row with:

```markdown
| N₀ | prior pseudo-count: "the prior is worth one view" (= 1) | model constant, FLAGGED |
```

and add a row:

```markdown
| ℓ_r (repeat) | azimuth / log-range step below which a view is a repeat look | DERIVED: the step at which expected shape similarity drops by one per-view sd |
```

- [ ] **Step 7: Ignore the data dirs and commit**

```bash
grep -q "retina/etc/zed_orbits/" .gitignore || echo "retina/etc/zed_orbits/" >> .gitignore
grep -q "retina/etc/view_atlas_dataset/" .gitignore || echo "retina/etc/view_atlas_dataset/" >> .gitignore
git add .gitignore common/view_atlas/tools/build_dataset.py common/view_atlas/tools/fit_constants.py \
        common/view_atlas/view_atlas_constants.txt common/view_atlas/VIEW_ATLAS.md
git commit -m "view_atlas: labelled views dataset builder + fitted constants (orbit + apartment)"
```

---

### Task 7: Offline replay harness through the module

**Files:**
- Create: `common/view_atlas/tools/atlas_replay.cpp`
- Modify: `common/view_atlas/VIEW_ATLAS.md` (add a "§8 Phase 1 results" section with the numbers)

**Interfaces:**
- Consumes: Tasks 1–3 (`ViewAtlas`, `ViewObs`, `view_quality`, `log_ratio_marginal`, `shift_log_weights`, `reid_posterior`, `location_log_lik`, `load_params`), Task 6 dataset + constants.
- Produces: `bin/atlas_replay <dataset_dir> <constants.txt> <out_dir>` printing four metrics with PASS/FAIL and writing `<out_dir>/replay_views.csv`.

Pass criteria (declared before running, from VIEW_ATLAS.md §6):
- **A. Swaps:** chairs, both datasets, lap 0: fraction of chair views whose own log-ratio is below the other chair's < 1 %.
- **B. Moved chair:** archive both chairs after lap 0; in lap 1, every window of 5 consecutive views of each chair is a birth candidate; with the yaw from the detection's own fit, MAP of the re-ID posterior is the right chair in ≥ 95 % of windows for the yellow chair (moved) and for the magenta chair (stayed), and the other chair in ≤ 2 %. The same windows with yaw unknown (flat shift prior) are reported as information, not graded.
- **C. Encoding rate:** with `lambda_nats` from the constants, encodings per object per minute of observation on the apartment tour (real timestamps; the orbit recorder's stamps are synthetic), reported (expect ≤ 30; > 60 is a FAIL: the trigger is not saturating).
- **D. Calibration:** re-ID posterior of the chosen hypothesis binned in [0.5,0.7), [0.7,0.9), [0.9,1.0]; observed accuracy within ±0.1 of the bin's mean posterior for every bin with ≥ 20 decisions.

- [ ] **Step 1: Write the harness**

Create `common/view_atlas/tools/atlas_replay.cpp`:

```cpp
// atlas_replay — run the view-atlas module over the labelled views dataset (VIEW_ATLAS.md §6 layer 3).
// usage: atlas_replay <dataset_dir> <constants.txt> <out_dir>
#include "../atlas_io.h"
#include "../view_atlas.h"

#include <charconv>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace rc::atlas;

struct Row { int id; std::string dataset, inst, cls; std::uint64_t stamp; int lap;
             float cam_x, cam_y, obj_x, obj_y, obj_yaw, range; int px; float trunc; };

static std::vector<Row> read_views(const std::string& path)
{
    std::vector<Row> out;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line))
    {
        const auto c = detail::split(line, ',');
        Row r{};
        auto num = [&](std::size_t i, auto& v) { detail::parse(c[i], v); };
        num(0, r.id); r.dataset = std::string(c[1]); num(2, r.stamp); num(3, r.lap);
        r.inst = std::string(c[4]); r.cls = std::string(c[6]);
        num(7, r.cam_x); num(8, r.cam_y); num(9, r.obj_x); num(10, r.obj_y); num(11, r.obj_yaw);
        num(12, r.range); num(13, r.px); num(14, r.trunc);
        out.push_back(r);
    }
    return out;
}

static std::vector<float> read_f32(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto n = std::size_t(f.tellg()); f.seekg(0);
    std::vector<float> v(n / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n));
    return v;
}

static float bearing_of(const Row& r) { return wrap_pi(std::atan2(r.cam_y - r.obj_y, r.cam_x - r.obj_x) - r.obj_yaw); }

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 4) { std::fprintf(stderr, "usage: %s dataset_dir constants.txt out_dir\n", argv[0]); return 2; }
    AtlasParams P; std::string why;
    if (not load_params(argv[2], P, &why)) { std::printf("constants refused: %s\n", why.c_str()); return 2; }
    const std::string d = argv[1];
    const auto views = read_views(d + "/views.csv");
    const auto S = read_f32(d + "/shape.f32"), C = read_f32(d + "/colour.f32");
    std::filesystem::create_directories(argv[3]);
    std::ofstream log(std::string(argv[3]) + "/replay_views.csv");
    log.imbue(std::locale::classic());
    log << "view_id,inst,lap,bearing,quality,learning_value,encoded,lr_own,lr_rival\n";

    auto obs_of = [&](const Row& r) {
        ViewObs o;
        o.shape = Eigen::Map<const Eigen::VectorXf>(S.data() + std::size_t(r.id) * 384, 384);
        o.colour = Eigen::Map<const Eigen::VectorXf>(C.data() + std::size_t(r.id) * 64, 64);
        o.bearing_rad = bearing_of(r); o.range_m = r.range;
        o.quality = view_quality(r.trunc, 1.f, r.px, P.quality_px_scale); o.stamp_ms = r.stamp;
        return o;
    };
    const auto known_yaw   = shift_log_weights(P.n_bins, 0.f);    // the object fit gives yaw: sigma 0
    const auto unknown_yaw = shift_log_weights(P.n_bins, 1e3f);   // robustness: yaw not known at all

    // ── A + C: lap 0, chronological, both datasets; atlases per instance ─────────────────────────
    std::map<std::string, ViewAtlas> atlas;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> seen;   // first/last stamp
    std::map<std::string, int> n_encoded;
    int chair_views = 0, swaps = 0;
    for (const auto& r : views)
    {
        if (r.lap != 0) continue;
        auto [it, fresh] = atlas.try_emplace(r.inst, P, 384, 64);
        const ViewObs o = obs_of(r);
        const float lv = it->second.learning_value(o.bearing_rad, o.quality);
        float lr_own = 0.f, lr_riv = 0.f;
        if (r.cls == "chair")
        {
            const std::string rival = r.inst == "Chair" ? "chair(1)" : "Chair";
            if (atlas.contains(rival) and it->second.state().n_c > 0.f)
            {
                // rival's bearing = the camera seen from the rival, in the rival's frame
                const Row* any_rival = nullptr;
                for (const auto& x : views) if (x.inst == rival and x.lap == 0) { any_rival = &x; break; }
                Row rr = r; rr.obj_x = any_rival->obj_x; rr.obj_y = any_rival->obj_y; rr.obj_yaw = any_rival->obj_yaw;
                lr_own = it->second.log_ratio_marginal(o, o.bearing_rad, known_yaw);
                lr_riv = atlas.at(rival).log_ratio_marginal(o, bearing_of(rr), known_yaw);
                ++chair_views; if (lr_riv > lr_own) ++swaps;
            }
        }
        const bool enc = lv > P.lambda_nats;
        if (enc) it->second.update(o);
        if (r.dataset == "frames")      // orbit stamps are synthetic (100 ms per frame): no real time there
        {
            if (enc) ++n_encoded[r.inst];
            auto& s = seen[r.inst]; if (s.first == 0) s.first = r.stamp; s.second = r.stamp;
        }
        log << r.id << ',' << r.inst << ',' << r.lap << ',' << o.bearing_rad << ',' << o.quality << ','
            << lv << ',' << enc << ',' << lr_own << ',' << lr_riv << '\n';
    }
    const float swap_rate = chair_views ? float(swaps) / float(chair_views) : 1.f;
    std::printf("A. chair swaps: %d / %d = %.2f %% -> %s\n", swaps, chair_views, 100.f * swap_rate, swap_rate < 0.01f ? "PASS" : "FAIL");

    float worst_rate = 0.f;
    for (const auto& [inst, n] : n_encoded)
    {
        const float minutes = std::max(1e-3f, float(seen[inst].second - seen[inst].first) / 60000.f);
        worst_rate = std::max(worst_rate, float(n) / minutes);
        std::printf("   %-28s encoded %4d over %6.1f min = %5.1f /min\n", inst.c_str(), n, minutes, float(n) / minutes);
    }
    std::printf("C. worst encodings per object-minute (apartment tour, real time): %.1f -> %s\n", worst_rate, worst_rate <= 60.f ? "PASS" : "FAIL");

    // ── B + D: archive the chairs after lap 0; lap 1 windows of 5 views are birth candidates ──────
    // The bearing of a detection comes from ITS OWN geometric fit (position + yaw), whichever identity is
    // hypothesised; identity changes which atlas is asked, not where the object is.
    const std::vector<std::string> chairs = {"Chair", "chair(1)"};
    std::map<std::string, Eigen::Vector2f> last_xy;
    for (const auto& r : views) if (r.lap == 0 and (r.inst == "Chair" or r.inst == "chair(1)")) last_xy[r.inst] = {r.obj_x, r.obj_y};
    const float room_area = 60.f;                               // apartment free space (rooms.json id 0: 60.41 m^2)
    const Eigen::Matrix2f pose_cov = Eigen::Matrix2f::Identity() * 0.05f * 0.05f;
    const float log_prior_new = std::log(0.5f), log_prior_k = std::log(0.25f);
    struct Dec { float p; bool right; };
    std::vector<Dec> decisions;
    bool b_ok = true;
    for (const auto* yaw_w : {&known_yaw, &unknown_yaw})
    {
        const bool known = yaw_w == &known_yaw;
        std::map<std::string, std::pair<int, int>> correct;     // inst -> (right, total)
        int wrong_chair = 0, total_windows = 0;
        for (const auto& truth : chairs)
        {
            std::vector<const Row*> lap1;
            for (const auto& r : views) if (r.lap == 1 and r.inst == truth) lap1.push_back(&r);
            for (std::size_t w0 = 0; w0 + 5 <= lap1.size(); w0 += 5)
            {
                std::vector<ReidCandidate> cand;
                for (const auto& k : chairs)
                {
                    ReidCandidate c; c.log_prior = log_prior_k;
                    c.log_loc = location_log_lik({lap1[w0]->obj_x, lap1[w0]->obj_y}, last_xy[k], pose_cov, P.pi_move, room_area);
                    for (std::size_t i = w0; i < w0 + 5; ++i)
                        c.log_ratio += atlas.at(k).log_ratio_marginal(obs_of(*lap1[i]), bearing_of(*lap1[i]), *yaw_w);
                    cand.push_back(c);
                }
                const auto post = reid_posterior(cand, log_prior_new, std::log(1.f / room_area));
                const std::size_t map = std::size_t(std::max_element(post.begin(), post.end()) - post.begin());
                const std::size_t want = truth == "Chair" ? 0 : 1;
                auto& cr = correct[truth]; ++cr.second; if (map == want) ++cr.first;
                if (map < chairs.size() and map != want) ++wrong_chair;
                ++total_windows;
                if (known) decisions.push_back({post[map], map == want});
            }
        }
        bool ok = true;
        for (const auto& [inst, rt] : correct)
        {
            const float acc = rt.second ? float(rt.first) / float(rt.second) : 0.f;
            ok = ok and acc >= 0.95f;
            std::printf("   [yaw %s] %-10s re-identified in %d / %d windows = %.1f %%\n", known ? "known  " : "unknown",
                        inst.c_str(), rt.first, rt.second, 100.f * acc);
        }
        const float wrong_rate = total_windows ? float(wrong_chair) / float(total_windows) : 1.f;
        ok = ok and wrong_rate <= 0.02f;
        if (known) b_ok = ok;
        std::printf("%s moved/stayed chair re-ID, yaw %s (other chair chosen %.1f %%) -> %s\n",
                    known ? "B." : "  (info)", known ? "known" : "unknown", 100.f * wrong_rate, ok ? "PASS" : "FAIL");
    }

    bool d_ok = true;
    const std::pair<float, float> bands[] = {{0.5f, 0.7f}, {0.7f, 0.9f}, {0.9f, 1.0001f}};
    for (auto [lo, hi] : bands)
    {
        float ps = 0.f, rs = 0.f; int n = 0;
        for (const auto& dd : decisions) if (dd.p >= lo and dd.p < hi) { ps += dd.p; rs += dd.right; ++n; }
        if (n == 0) { std::printf("   posterior [%.1f,%.1f): no decisions\n", lo, hi); continue; }
        const float gap = std::fabs(ps / float(n) - rs / float(n));
        if (n >= 20) d_ok = d_ok and gap <= 0.1f;
        std::printf("   posterior [%.1f,%.1f): n=%d mean p %.3f, accuracy %.3f\n", lo, hi, n, ps / float(n), rs / float(n));
    }
    std::printf("D. calibration -> %s\n", d_ok ? "PASS" : "FAIL");
    return (swap_rate < 0.01f and worst_rate <= 60.f and b_ok and d_ok) ? 0 : 1;
}
```

- [ ] **Step 2: Build and run**

Run:
```bash
bash common/view_atlas/tools/build_tools.sh
common/view_atlas/tools/bin/atlas_replay retina/etc/view_atlas_dataset common/view_atlas/view_atlas_constants.txt retina/etc/view_atlas_replay
```
Expected: four lettered lines A–D each ending in `PASS`, exit code 0. Report every number whatever the verdict. A FAIL is a result to bring back to the user with the numbers, not a reason to change the criteria or the constants by hand.

- [ ] **Step 3: Record the results in the spec**

Append to `common/view_atlas/VIEW_ATLAS.md`:

```markdown
## 8. Phase 1 results (offline replay, <date>)

Dataset: `retina/etc/view_atlas_dataset` (<N> views, <M> objects), constants `view_atlas_constants.txt`.

| Check | Criterion | Result |
|---|---|---|
| A. chair swaps | < 1 % | <value> |
| B. re-ID, yellow moved / magenta stayed | ≥ 95 % each, other chair ≤ 2 % | <value> / <value>, <value> |
| C. encodings per object-minute | ≤ 60 (expect ≤ 30) | <value> |
| D. calibration | ±0.1 per bin with ≥ 20 decisions | <per-bin values> |
```

Fill every `<…>` with the numbers atlas_replay printed (this is a results table, so it is written after the run, never before).

- [ ] **Step 4: Commit**

```bash
git add common/view_atlas/tools/atlas_replay.cpp common/view_atlas/VIEW_ATLAS.md
git commit -m "view_atlas: offline replay harness through the module + phase 1 results"
```

---

## After phase 1 (separate plan, written once Task 7's numbers exist)

Phase 2 — chair_concept shadow mode: add an `ai_common::view_atlas` CMake target (ORT discovery copied from `retina/src/CMakeLists.txt`); `[ViewAtlas]` config (`enabled = false`, model path, constants path, λ, cap); `FrameHistory` fed from chair_concept's RGB ingestor (first verify that `MasksPacket::timestamp_ms` equals `RgbIngestor::stamp_ms()` for the same frame — FrameHistory matches stamps exactly); the encoder worker thread with a deep-copy queue; per-instance atlases; atlas → archive on death; at each birth candidate, log the re-ID posterior next to what the tracker did — **without acting**. Phase 3 — acting: the cortex `object_uid` attribute (user reinstalls cortex), resurrection in the birth path, the appearance term in association, the lifecycle §6 audit row, then the live sim test of VIEW_ATLAS.md §6 layer 4.
