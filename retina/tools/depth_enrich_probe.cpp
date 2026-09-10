/*
 * tools/depth_enrich_probe.cpp — offline probe for the ricoh depth-enrichment pass.
 *
 * Two questions, neither of which needs a robot, a graph or a camera node:
 *   1. ★MODEL PARITY (point 2 of depth_enrichment.h). Re-run yolo26l-depth on the saved panoramas and
 *      compare the recomputed log_model at every recoverable stored pixel against the value the
 *      dataset holds. If this is not ~0, the stored rows and any new synthetic rows would be paired
 *      with two DIFFERENT fields and the whole enrichment is unsound.
 *   2. REGRESSION. Refit the existing dataset and check the weighted/ct-capable fit reproduces the
 *      map that is on disk — an untouched dataset must fit exactly as it did before weights existed.
 *
 * ★LOCALE (CLAUDE.md). A standalone harness has no Qt, so it stays in the "C" locale and the
 * decimal-comma bug VANISHES — a probe and the agent would then answer different questions about the
 * same file. setlocale(LC_ALL, "") below puts this binary in the same locale the agent runs in, which
 * is the only way its verdict means anything.
 *
 * Build (not part of the component target — it is a probe, not a product):
 *   see the command in the report / rebuild it with the same flags as build/.../flags.make
 */

#include "../src/depth_dataset.h"
#include "../src/depth_enrichment.h"
#include "../src/depth_processor.h"

#include <clocale>
#include <cmath>
#include <cstdlib>
#include <print>
#include <array>
#include <random>
#include <string>

namespace
{
// ── Self-test of the machinery the enrichment adds to fit(): per-sample WEIGHTS, the common-mode
// (Woodbury) marginalisation, and the ΔBIC decision on ct*t². Ground truth is known by construction,
// so this answers "is the estimator correct" without needing a robot, a room or a camera.
//
// Stage 1 builds a LIDAR-LIKE set: t spans only −0.14..+0.67, exactly the horizon stripe the real
// dataset covers. ct is not even proposed there (no synthetic rows), so it must stay off.
// Stage 2 adds CEILING-LIKE rows at t ∈ −0.66..0 with realistic precisions and a shared per-surface
// common mode, and ct must come back with the right value.
int selftest()
{
    constexpr int V = 6, F = 24;
    constexpr double a_true = 0.24, cs_true = 0.10, ct_true = -0.22;
    const std::array<double, V> b_true{0.10, -0.20, 0.35, -0.05, 0.22, -0.31};
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> ulm(3.0, 6.5), us(-1.0, 1.0);
    std::uniform_real_distribution<double> t_lidar(-0.14, 0.67), t_ceil(-0.66, 0.0);
    std::normal_distribution<double> n_meas(0.0, 0.05), n_syn(0.0, 0.05);

    const auto truth = [&](double lm, int v, double s, double t)
    { return a_true * lm + b_true[static_cast<std::size_t>(v)] + cs_true * s * s + ct_true * t * t; };

    rc::depth::DepthDataset ds;
    std::vector<rc::depth::DepthFrame> frames;
    for (int f = 0; f < F; ++f)
    {
        rc::depth::DepthFrame fr;
        fr.stamp_ms = 1000 + static_cast<std::uint64_t>(f);
        fr.rx = static_cast<float>(f);      // distinct poses so nothing is deduped
        for (int i = 0; i < 1200; ++i)
        {
            rc::depth::DepthSample s;
            s.view      = static_cast<std::uint8_t>(i % V);
            s.log_model = static_cast<float>(ulm(rng));
            s.s         = static_cast<float>(us(rng));
            s.t         = static_cast<float>(t_lidar(rng));
            s.log_range = static_cast<float>(truth(s.log_model, s.view, s.s, s.t) + n_meas(rng));
            fr.samples.push_back(s);
        }
        frames.push_back(std::move(fr));
        ds.add_frame(rc::depth::DepthFrame{frames.back()}, 0.f, 0.f);
    }

    const auto m0 = ds.fit(V);
    std::println("[selftest] stage 1 (LiDAR-like only): a={:.4f} (true {:.4f})  cs={:+.4f} (true {:+.4f})"
                 "  ct_active={}  n={}", m0.a, a_true, m0.cs, cs_true, m0.ct_active, m0.n_samples);
    // ★Stage 1's `a` is BIASED and that is the point: with the vertical term omitted, ct*t² has to go
    // somewhere, and it leaks into the parameters that remain. Assert only that ct stays off and that
    // the linear part is in the right neighbourhood; the meaningful assertion is that stage 2 moves
    // `a` CLOSER to the truth, which is checked below.
    bool ok = std::abs(m0.a - a_true) < 0.05 and std::abs(m0.cs - cs_true) < 0.03 and not m0.ct_active;

    // Stage 2: ceiling-like synthetic rows. Each frame gets ONE common-mode region (the ceiling
    // plane), a shared displacement of that plane, and per-row weights well below a LiDAR row's.
    std::normal_distribution<double> plane(0.0, 1.0);
    for (int f = 0; f < F; ++f)
    {
        std::vector<rc::depth::DepthSample> extra;
        const double dn = plane(rng);          // this frame's ceiling displacement, in h units
        for (int i = 0; i < 3000; ++i)
        {
            rc::depth::DepthSample s;
            s.view      = static_cast<std::uint8_t>(i % V);
            s.log_model = static_cast<float>(ulm(rng));
            s.s         = static_cast<float>(us(rng));
            s.t         = static_cast<float>(t_ceil(rng));
            s.src       = rc::depth::kSrcEnvelope;
            s.region    = 2;                   // the ceiling
            s.w         = 0.8f;                // ~as precise as a LiDAR row, per the derivation
            s.h         = 1.3f;                // common-mode sd, in LiDAR-row units
            // sigma_independent = sigma_lidar / sqrt(w); the common part is h * dn, shared.
            const double ind = n_syn(rng) / std::sqrt(static_cast<double>(s.w));
            s.log_range = static_cast<float>(truth(s.log_model, s.view, s.s, s.t)
                                             + ind + 0.02 * s.h * dn);
            extra.push_back(s);
        }
        ds.add_samples_to_frame(1000 + static_cast<std::uint64_t>(f), extra);
    }
    const auto m1 = ds.fit(V);
    const auto m1m = ds.fit(V, /*measured_only=*/true);
    std::println("[selftest] stage 2 (+ ceiling rows): a={:.4f}  cs={:+.4f}  ct={:+.4f} (true {:+.4f})"
                 "  ct_active={} dBIC={:+.1f}  t {:+.3f}..{:+.3f}  n={} ({} synthetic)",
                 m1.a, m1.cs, m1.ct, ct_true, m1.ct_active, m1.ct_delta_bic, m1.t_lo, m1.t_hi,
                 m1.n_samples, m1.n_synth);
    std::println("[selftest] measured-only refit of the SAME set: a={:.4f} ct_active={} n={} "
                 "(must equal stage 1 — the A/B baseline must not see the synthetic rows)",
                 m1m.a, m1m.ct_active, m1m.n_samples);
    std::println("[selftest] omitted-variable bias in a: {:+.4f} without the ceiling rows → {:+.4f} "
                 "with them (must SHRINK — that is what the enrichment buys)",
                 m0.a - a_true, m1.a - a_true);
    ok = ok and m1.ct_active and std::abs(m1.ct - ct_true) < 0.03
             and std::abs(m1.a - a_true) < std::abs(m0.a - a_true)
             and std::abs(m1.a - a_true) < 0.01
             and m1m.n_samples == m0.n_samples and not m1m.ct_active;
    std::println("[selftest] {}", ok ? "PASS" : "★FAIL");
    return ok ? 0 : 5;
}
}   // namespace

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");   // ★answer the SAME question the Qt agent answers — see the header

    std::string dataset = "etc/ricoh_depth_dataset.csv";
    std::string frames  = "etc/depth_frames";
    std::string map     = "etc/ricoh_depth_map.csv";
    int  n_strips = 6, parity_frames = 6;
    bool gnomonic = false;
    bool do_parity = true;
    bool use_trt = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if      (a == "--dataset" and i + 1 < argc) dataset = argv[++i];
        else if (a == "--frames"  and i + 1 < argc) frames  = argv[++i];
        else if (a == "--map"     and i + 1 < argc) map     = argv[++i];
        else if (a == "--strips"  and i + 1 < argc) n_strips = std::atoi(argv[++i]);
        else if (a == "--parity-frames" and i + 1 < argc) parity_frames = std::atoi(argv[++i]);
        else if (a == "--gnomonic") gnomonic = true;
        else if (a == "--trt") use_trt = true;   // match the collecting run's EP (FP16 engine)
        else if (a == "--no-parity") do_parity = false;
        else if (a == "--selftest") return selftest();
    }

    rc::depth::DepthDataset ds;
    if (not ds.load_csv(dataset))
    {
        std::println("cannot open {}", dataset);
        return 1;
    }
    std::println("[probe] {} — {} frames / {} samples", dataset, ds.frame_count(), ds.sample_count());

    // ── 2. regression: does the weighted, ct-capable fit reproduce what is on disk? ──────────────
    rc::depth::DepthFitMap on_disk;
    const bool had_map = on_disk.load(map);
    const auto refit = ds.fit(n_strips);
    std::println("[probe] refit : a={:.5f} cs={:+.5f} ct={:+.5f}{} resid={:.5f} anch={:.5f} r={:+.5f}\n"
                 "                med_rel={:.5f} d125={:.5f} range {:.4f}..{:.4f} t {:+.4f}..{:+.4f} "
                 "n_knots={} n={}",
                 refit.a, refit.cs, refit.ct, refit.ct_active ? "" : "(off)", refit.resid_rms,
                 refit.resid_anchored, refit.r, refit.med_rel, refit.delta125,
                 refit.range_lo, refit.range_hi, refit.t_lo, refit.t_hi, refit.n_knots,
                 refit.n_samples);
    if (had_map)
        std::println("[probe] ondisk: a={:.5f} cs={:+.5f} resid={:.5f} anch={:.5f} r={:+.5f} "
                     "med_rel={:.5f} d125={:.5f} range {:.4f}..{:.4f} n={}  ⇒ Δa={:+.2e} Δanch={:+.2e}",
                     on_disk.a, on_disk.cs, on_disk.resid_rms, on_disk.resid_anchored, on_disk.r,
                     on_disk.med_rel, on_disk.delta125, on_disk.range_lo, on_disk.range_hi,
                     on_disk.n_samples, refit.a - on_disk.a,
                     refit.resid_anchored - on_disk.resid_anchored);

    if (not do_parity)
        return 0;

    // ── 1. model parity ─────────────────────────────────────────────────────────────────────────
    rc::depth::EnrichConfig cfg;
    cfg.dataset_csv   = dataset;
    cfg.frames_dir    = frames;
    cfg.n_views       = n_strips;
    cfg.parity_frames = parity_frames;
    cfg.depth_cfg.model_path = "models/yolo26/yolo26l-depth.onnx";
    cfg.depth_cfg.input_size = 768;
    cfg.depth_cfg.use_gpu    = true;
    cfg.depth_cfg.use_trt    = use_trt;   // --trt to reproduce the agent's FP16 TensorRT engine
    cfg.depth360.n_strips           = n_strips;
    cfg.depth360.overlap_px         = 64;
    cfg.depth360.band_half_elev_deg = 60.0f;
    cfg.depth360.gnomonic           = gnomonic;
    cfg.depth360.zdepth_to_range    = true;

    rc::depth::DepthProcessor depth;
    depth.configure(cfg.depth_cfg);
    if (not depth.ready())
    {
        std::println("[probe] depth model did not load — parity NOT measured");
        return 2;
    }
    const auto pa = rc::depth::DatasetEnricher::measure_model_parity(ds, depth, cfg);
    if (pa.n <= 0)
    {
        std::println("[probe] PARITY UNMEASURABLE ({} frames re-run, {} samples clamped)",
                     pa.frames, pa.n_clamped);
        return 3;
    }
    std::println("[probe] ★MODEL PARITY over {} samples in {} frames ({} excluded: clamped s/t)\n"
                 "        RAW        median |Δlog_model| = {:.5f}  ⇒ {:.2f}% in depth\n"
                 "                   p95 {:.5f}   rms {:.5f}\n"
                 "        per-(frame,view) OFFSET removed (median |offset| = {:.5f} over {} views)\n"
                 "        REGISTERED median |Δlog_model| = {:.5f}  ⇒ {:.2f}% in depth\n"
                 "        verdict: {}",
                 pa.n, pa.frames, pa.n_clamped, pa.med_abs_log,
                 100.0 * (std::exp(pa.med_abs_log) - 1.0), pa.p95_abs_log, pa.rms_abs_log,
                 pa.med_offset, pa.views_registered, pa.med_abs_log_registered,
                 100.0 * (std::exp(pa.med_abs_log_registered) - 1.0),
                 pa.ok ? "SAME FIELD once registered — safe to mix"
                       : "★NOT THE SAME FIELD — do not enrich");
    return pa.ok ? 0 : 4;
}
