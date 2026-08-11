#pragma once

/*
 * depth_enrichment.{h,cpp} — OFFLINE ENRICHMENT of the ricoh monocular-depth calibration.
 *
 * ─── WHY ─────────────────────────────────────────────────────────────────────────────────────────
 * The correction map (depth_dataset.h) is supervised by helios LiDAR alone, and helios is a
 * near-horizontal scanner at ~1.1 m. Two consequences, both MEASURED, neither fixable by collecting
 * more frames:
 *   1. The dataset's vertical coordinate `t` spans −0.14..+0.67 against a band of ±0.667 — the entire
 *      upper half (the CEILING) has ZERO samples. That is why the ct*t² term was deleted: it was pure
 *      extrapolation over exactly the half of the frame nothing constrained.
 *   2. The supported range stops at ~0.39..10.16 m, because that is where a helios sweep at robot
 *      height stops returning.
 * The ROOM BELIEF knows things the LiDAR cannot say: where the ceiling is, where the floor is, and
 * where the walls are — including the far wall down a corridor that no sweep reaches. This pass
 * turns that belief into extra supervision.
 *
 * ─── THE PIPELINE, PER SAVED FRAME ───────────────────────────────────────────────────────────────
 * Every ADMITTED collection frame already has its panorama on disk as <frames_dir>/<stamp_ms>.jpg,
 * keyed on the same stamp as its CSV rows. For each of them:
 *   1. re-run yolo26l-depth on the JPEG            → log_model over the whole panorama
 *   2. run yolo26l-sem-ade20k on the same JPEG     → per-pixel ceiling / floor / wall labels
 *   3. ray-cast the room envelope through every pixel → metric range
 *   4. emit synthetic DepthSamples pairing (1) with ln(3) wherever (2) says we are looking at the shell
 *   5. refit DepthFitMap over the LiDAR rows AND the synthetic rows together
 * The measured rows are NEVER touched: enrichment only ADDS. A pass that produces nothing leaves the
 * dataset exactly as it found it, and `Report` always carries the LiDAR-only refit beside the enriched
 * one so the two can be compared without re-running anything.
 *
 * ─── FIVE THINGS THAT DECIDE WHETHER THIS IS SOUND ───────────────────────────────────────────────
 *
 * (1) THERE ARE NO LIDAR POINTS TO RE-READ, and none are needed. Raw clouds are never saved; every
 *     stored CSV row already IS a paired (log_model, log_range) sample. They are carried through
 *     verbatim.
 *
 * (2) RE-RUNNING THE DEPTH MODEL PRODUCES A NEW FIELD, and the stored rows came from an older run.
 *     Mixing the two is only legitimate if the model + config is deterministic and unchanged. That is
 *     not assumed — it is MEASURED, by check_model_parity(): `s` and `t` are stored as normalised
 *     in-view / in-panorama coordinates, so (u,v) is recoverable from (view, s, t) given the strip
 *     width and panorama size, and the recomputed log_model at that pixel can be differenced against
 *     the stored one. Samples whose s or t SATURATED at ±1 are excluded — a clamped coordinate is not
 *     invertible. If the median disagreement is not ~0 the two runs are not the same experiment and
 *     the whole enrichment is unsound; run() says so loudly and refuses rather than quietly fitting a
 *     mixture of two different fields.
 *
 * (3) THE ROOM GEOMETRY IS NOT IN THE DATASET SCHEMA. `frame,stamp_ms,rx,ry,rtheta,view,s,t,log_model,
 *     log_range` carries the ricoh's x/y/yaw but not its height, not its full orientation, and not the
 *     polygon or ceiling height it was standing in. Both halves of the fix are implemented:
 *       (a) GOING FORWARD the collector writes a per-frame SIDECAR (append_room_geometry) holding the
 *           room name, the polygon, room_height and the FULL 3x4 room_T_ricoh — which also recovers the
 *           z and the pitch/roll that (rx,ry,rtheta) throws away. A sidecar rather than new columns
 *           because the polygon is per-FRAME and the CSV row is per-SAMPLE: repeating a polygon on
 *           every one of 65,000 rows would bloat the file for no gain, and it keeps the existing 5 MB
 *           dataset loading byte-for-byte unchanged.
 *       (b) FOR LEGACY FRAMES there is only one source: the graph as it is RIGHT NOW. That silently
 *           assumes the frames were collected in this same room and that room_concept's belief has not
 *           moved since. The assumption is real and it is LOGGED BY NAME, once per run, with the count
 *           of frames it applies to.
 *
 * (4) THE SEMANTIC LABELLING DOES THE REAL WORK, and a wrong synthetic sample is LARGE. The envelope
 *     is truth only where the pixel really shows the shell. Per CLAUDE.md this is NOT a hard
 *     include/exclude gate: it is a PRECISION that falls out of the generative model — see (5).
 *
 * (5) THE FIT IS WEIGHTED. DepthSample now carries `w` (precision relative to a LiDAR row, 1.0 by
 *     definition) and a common-mode group (`region`, `h`). A synthetic row's precision is DERIVED, not
 *     tuned; the derivation is in weight_of() in the .cpp and summarised here:
 *
 *        σ² = p·σ_in² + (1−p)·σ_out²                      ← the label is a HYPOTHESIS, not a fact
 *        σ_in²  = σ_pix² + (δα·tanθ)²                     ← independent part: pixel + obliquity
 *        σ_out  = ln(1 + d_class/R)                       ← if it is NOT the shell, how far off is it
 *        h      = σ_n / (R·cosθ)                          ← COMMON mode: the believed plane's own σ
 *
 *     Three physical covariates carry all the behaviour, and none of them is a switch:
 *       · INCIDENCE. A ray grazing a wall (cosθ→0) turns a small pose error into a huge range error,
 *         so h→∞ and the sample fades out continuously. No "reject if incidence > k".
 *       · SEMANTIC CONFIDENCE. p is P(this pixel shows the surface the envelope predicts), built from
 *         the segmenter's own argmax probability AND from whether the winning class is compatible with
 *         the surface the ray-cast actually hit. A confidently-labelled table over a floor hit gets a
 *         small p, so the sample survives only with σ_out precision. No "keep only floor pixels".
 *       · WHAT THE CLASS IMPLIES IF WE ARE WRONG. d_class encodes "how far from the shell can the true
 *         surface be here": ~0.15 m for a ceiling (nothing hangs there — the reason ceiling samples are
 *         the good ones), ~0.6 m for a wall (shelves, pictures), ~1.0 m for a floor (furniture). This
 *         is where "ceiling is nearly free, floor and walls are contested" lives, as a variance.
 *     The COMMON mode matters as much as the weight: all the pixels on one believed wall move together
 *     when that wall moves, so their errors are not independent. fit() marginalises that rank-1
 *     direction exactly (Woodbury), which lets a wall's internal contrasts inform the spline at full
 *     strength while its overall offset saturates at one plane-limited observation.
 *
 * ─── THREADING ───────────────────────────────────────────────────────────────────────────────────
 * Two networks over hundreds of frames is MINUTES. The class owns ITS OWN ONNX sessions (never
 * DepthStage's / SemanticStage's — those belong to the ricoh worker thread) and runs the whole pass on
 * its own std::thread; start()/progress()/join() let a GUI keep pumping events. The DSR::CameraAPI is
 * bound from the MAIN thread before start() and then used only by the worker (ray_from_pixel reads
 * cached intrinsics and touches no graph). Every cv::Mat that crosses the boundary is deep-copied.
 * Nothing here needs a live robot except the legacy-geometry fallback, so the same class drives a
 * standalone offline tool: construct, bind a camera, set_geometry(), run().
 */

#include "depth_dataset.h"
#include "depth_processor.h"
#include "yolo_semantic.h"

#include <dsr/api/dsr_eigen_defs.h>   // Mat::RTMat

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace DSR { class CameraAPI; }

namespace rc::depth
{

// The room belief a frame was captured under. `room_T_cam` is the ricoh's pose in the room frame —
// the FULL transform, because (rx,ry,rtheta) in the dataset loses the camera height and any pitch.
struct RoomGeometry
{
    std::string        room;                                   // room node name, for the log line
    std::vector<float> poly_x, poly_y;                         // floor polygon, room frame, metres
    float              height     = 0.f;                       // ceiling z, metres
    Mat::RTMat         room_T_cam = Mat::RTMat::Identity();
    float              cam_z      = 0.f;                        // fallback camera height (legacy frames)
    bool               reconstructed = false;                   // true ⇒ came from the graph, not the sidecar

    [[nodiscard]] bool valid() const
    {
        return poly_x.size() >= 3 and poly_x.size() == poly_y.size() and height > 0.f;
    }
};

// ── Per-frame geometry sidecar (schema (b) of point 3) ───────────────────────────────────────────
// One row per collected frame:
//   stamp_ms,room,height,m00..m23,n,x0,y0,x1,y1,...
// Written by the live collector as each frame is admitted; read by the enricher. Locale-safe on both
// sides (imbue(classic) on write, std::from_chars on read) — see CLAUDE.md.
bool append_room_geometry(const std::string& path, std::uint64_t stamp_ms, const RoomGeometry& g);
bool load_room_geometry(const std::string& path, std::map<std::uint64_t, RoomGeometry>& out);

struct EnrichConfig
{
    std::string dataset_csv  = "etc/ricoh_depth_dataset.csv";
    std::string frames_dir   = "etc/depth_frames";
    std::string geometry_csv = "etc/ricoh_depth_rooms.csv";
    std::string out_dataset  = "";      // optional: write the ENRICHED set here (never over the source)
    int         n_views      = 6;

    // Must MATCH the collecting run, or the recomputed field is a different experiment — which is
    // exactly what check_model_parity() measures rather than assumes.
    DepthProcessor::Config depth_cfg;
    Depth360Config         depth360;
    rc::semantic::YoloSemanticProcessor::Config sem_cfg;

    // Emit one synthetic sample every `pixel_stride` pixels in BOTH axes. Neighbouring pixels on one
    // wall say the same thing (the same argument that put a stride on the LiDAR samples), and the
    // common-mode marginalisation makes the redundancy explicit anyway — this only keeps the file and
    // the pass cheap. 8 on a 1920x960 band ⇒ ~15k candidate pixels per frame.
    int   pixel_stride      = 8;
    int   envelope_decimate = 2;        // ray-cast grid; the envelope is piecewise planar

    // ── The generative model's own constants. Physical quantities with units, not tuning knobs. ──
    // σ of a LiDAR-anchored row in LOG range. Not the sensor's 2 cm: it is dominated by reprojection
    // parallax and by the pose used to put the return in the panorama. Everything else is expressed
    // RELATIVE to this, so it sets the exchange rate between a measurement and a belief.
    float sigma_lidar_log   = 0.02f;
    // σ of the believed surface along its own NORMAL, metres. Room-fit residual + robot localisation.
    float sigma_wall_m      = 0.07f;
    float sigma_floor_m     = 0.03f;    // z = 0 is the datum, so the floor is the best-known plane
    float sigma_ceiling_m   = 0.06f;    // room_height is one fitted scalar
    // Independent per-pixel σ in log range: JPEG, model/label misalignment, envelope decimation.
    float sigma_pixel_log   = 0.010f;
    // Angular size of the pixel-registration error, radians. Multiplied by tan(incidence) it becomes
    // the along-ray error of landing one pixel off on a slanted surface.
    float pixel_align_rad   = 0.007f;   // ~2 px on a 1920-wide equirect panorama
    // "If this pixel is NOT the shell, how far in front of it can the real surface be", metres.
    float standoff_ceiling_m = 0.15f;   // nothing hangs from a ceiling — the reason these rows are good
    float standoff_wall_m    = 0.60f;   // shelves, pictures, radiators, open doors
    float standoff_floor_m   = 1.00f;   // furniture sits on the floor
    // Ray range below which the envelope is not believed at all (the camera is inside its own body /
    // numerically degenerate). Not a modelling gate: the ray-cast itself is meaningless there.
    float min_range_m        = 0.25f;

    // ── Parity check (point 2) ───────────────────────────────────────────────────────────────────
    // MEASURED 2026-08-07 on the 53-frame set, 8 frames re-run, 9,880 comparable samples:
    //     RAW        median |Δlog_model| = 0.359  (43% in depth)
    //     of which a per-(frame,view) OFFSET of median 0.325 — the model re-drawing its arbitrary
    //     per-image scale, because the panorama on disk is a JPEG and not the bytes it first saw
    //     REGISTERED median |Δlog_model| = 0.062  (6.4% in depth)
    // TensorRT-FP16 and CUDA-FP32 gave 0.064 vs 0.062, so the execution provider is NOT the cause.
    // The verdict is taken on the REGISTERED number, because the offset is removed (see the
    // registration block in enrich_frame) and the shape is not. The cap is derived, not chosen:
    // log_model enters the map with slope a ≈ 0.24, so 0.15 of log_model is 0.036 of log_range =
    // 3.7% depth — a sixth of the map's own 0.21 anchored residual — AND that error is not merely
    // tolerated, it is added to every synthetic row's variance as (a·σ_registered)². Above the cap
    // the two runs disagree in SHAPE badly enough that the pairing means nothing.
    int   parity_frames      = 8;
    float parity_max_med_log = 0.15f;
    bool  parity_abort       = true;    // refuse to enrich when the check fails
    // A view needs at least this many MEASURED samples in a frame before that frame's synthetic rows
    // for it can be registered onto the stored field's scale. Below it, the view emits nothing —
    // an unregistered view would be off by the ~38% measured above. Not a quality gate: it is the
    // number of points a median needs to be a median.
    int   min_register_samples = 32;
};

class DatasetEnricher
{
public:
    struct Parity
    {
        long   n            = 0;      // comparable samples (unclamped s and t, both fields finite)
        long   n_clamped    = 0;      // excluded because s or t saturated ⇒ (u,v) not invertible
        int    frames       = 0;
        double med_abs_log  = -1.0;   // RAW |recomputed − stored| log_model, median
        double p95_abs_log  = -1.0;
        double rms_abs_log  = -1.0;
        // ★THE NUMBER THAT ACTUALLY DECIDES IT. Same difference after a per-(frame,view) OFFSET has
        // been removed. yolo26l-depth is scale-and-shift invariant PER INPUT IMAGE, and the panorama
        // on disk is a JPEG — not the bytes the model originally saw — so a re-run is entitled to
        // re-draw its arbitrary per-image offset. That part is registerable and is registered (see
        // `offset` in enrich_frame). What is NOT registerable is a change in SHAPE, and this is it.
        double med_abs_log_registered = -1.0;
        double med_offset             = 0.0;   // median |per-(frame,view) offset| that had to be removed
        int    views_registered       = 0;
        bool   ok           = false;
    };

    struct Report
    {
        bool        ok = false;
        std::string error;
        Parity      parity;
        DepthFitMap map_measured;      // LiDAR-only refit — the A/B baseline, same code, same frames
        DepthFitMap map_enriched;
        long        n_synth            = 0;
        int         frames_total       = 0;
        int         frames_enriched    = 0;
        int         frames_no_image    = 0;
        int         frames_no_geometry = 0;
        int         frames_legacy_geom = 0;   // used the graph fallback (assumption logged by name)
        int         views_unregistered = 0;   // (frame,view) pairs skipped: too few measured rows to register
        float       synth_range_lo = 0.f, synth_range_hi = 0.f;
        float       synth_t_lo = 0.f, synth_t_hi = 0.f;
        double      synth_weight_sum = 0.0;   // Σw of the synthetic rows, in LiDAR-row equivalents
    };

    enum class Phase { Idle, Loading, Parity, Sessions, Enriching, Fitting, Done, Failed };
    struct Progress
    {
        Phase phase = Phase::Idle;
        int   done = 0, total = 0;
        long  synth = 0;
    };
    [[nodiscard]] static const char* phase_name(Phase p);

    // ★(2) THE SOUNDNESS CHECK, free-standing on purpose. It needs neither a camera, nor a room, nor
    // a graph — only the dataset and a depth model — so it can be run from a standalone tool (or a
    // test) to answer "does re-running the model reproduce the field these rows were collected from"
    // WITHOUT a live robot. The member below just delegates to it.
    [[nodiscard]] static Parity measure_model_parity(const DepthDataset& ds, DepthProcessor& depth,
                                                     const EnrichConfig& cfg,
                                                     const std::atomic<bool>* cancel = nullptr,
                                                     std::atomic<int>* frames_done = nullptr);

    explicit DatasetEnricher(EnrichConfig cfg);
    ~DatasetEnricher();
    DatasetEnricher(const DatasetEnricher&)            = delete;
    DatasetEnricher& operator=(const DatasetEnricher&) = delete;

    // MAIN THREAD, before start(). The enricher takes ownership of its OWN CameraAPI instance so it
    // never shares one with the viewer/depth-fill path (ray_from_pixel is a pure read of cached
    // intrinsics, but an instance shared across threads is a rule this codebase does not bend).
    void bind_camera(std::unique_ptr<DSR::CameraAPI> cam);
    // Per-frame geometry from the sidecar, plus the fallback used for frames it does not cover.
    void set_geometry(std::map<std::uint64_t, RoomGeometry> per_frame, RoomGeometry legacy_fallback);

    // Asynchronous: spawns the worker and returns immediately. false ⇒ already running.
    bool start();
    [[nodiscard]] bool running() const { return running_.load(std::memory_order_acquire); }
    [[nodiscard]] Progress progress() const;
    void cancel() { cancel_.store(true, std::memory_order_release); }
    // Blocks until the worker exits and returns its report. Safe to call after start() at any time.
    Report join();

    // Synchronous, for a standalone offline tool with no event loop.
    Report run();

private:
    void work();
    [[nodiscard]] bool open_sessions();
    [[nodiscard]] Parity check_model_parity(const DepthDataset& ds);
    // Everything the pass produces for one frame. `synthetic` is empty when the frame contributes
    // nothing (no image, no geometry, nothing labelled as shell).
    [[nodiscard]] std::vector<DepthSample> enrich_frame(const DepthFrame& fr, const RoomGeometry& g);
    // Semantic labels + argmax scores over the panorama, segmented in the SAME strip decomposition the
    // depth model uses so the two fields are pixel-aligned by construction.
    [[nodiscard]] rc::semantic::SemanticMap segment_360(const cv::Mat& panorama_bgr) const;
    [[nodiscard]] RoomGeometry geometry_for(const DepthFrame& fr, bool& legacy) const;
    [[nodiscard]] cv::Mat load_frame_image(std::uint64_t stamp_ms) const;

    EnrichConfig cfg_;
    std::unique_ptr<DSR::CameraAPI>                        cam_;
    std::unique_ptr<DepthProcessor>                        depth_;
    std::unique_ptr<rc::semantic::YoloSemanticProcessor>   sem_;
    std::map<std::uint64_t, RoomGeometry>                  geom_;
    RoomGeometry                                           fallback_;
    // ★ERROR-IN-VARIABLES, measured rather than assumed. The re-run's log_model differs from the
    // stored one by the parity check's REGISTERED residual σ_reg; that lands on the fit's response
    // through the slope a, so every synthetic row carries an extra independent variance (a·σ_reg)².
    // Filled in by work() from the parity measurement and the LiDAR-only refit's own `a` — no guess.
    double model_var_log2_ = 0.0;

    std::thread       worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> cancel_{false};
    std::atomic<int>  phase_{static_cast<int>(Phase::Idle)};
    std::atomic<int>  done_{0}, total_{0};
    std::atomic<long> synth_{0};
    Report            report_;
};

}   // namespace rc::depth
