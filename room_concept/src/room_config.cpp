/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_config.h.
 */

#include "room_config.h"
#include <limits>
#include <cmath>

#include <ConfigLoader/ConfigLoader.h>
#include <QDebug>

#include "room_concept.h"
#include "epistemic_controller.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED): provenance per key

namespace rc
{
namespace
{
// The load_optional_apply shape on top of the Reader: the key registers with its code default like any
// other, and the callback runs only when the FILE supplied it (absent, shadowed or mistyped ⇒ no call).
template <class T, class F>
void reader_apply(const rc::cfg::Reader& reader, std::string_view key, std::string_view what, T def, F&& apply)
{
    reader.opt<T>(key, def, what);
    if (rc::cfg::registry().record(key).origin == rc::cfg::Origin::File)
        apply(def);
}

// An overlay that CHANGED a value, recorded on the key it changed, so the table (and the viewer's
// Config gates panel) shows origin=overlay instead of the file value that is no longer in force.
// The destination is the one registered key ending in ".<name>" outside the overlay sections; when
// that is not unique, the overlay key itself carries the record rather than a guessed destination.
// An overlay key ([Platform.<robot>.*] / [Scenario.<name>.*]) is PARSED here for every robot and
// scenario but APPLIED only for the one the graph names, so it is not a setting of its own: read it
// without registering (no "(absent)" rows for the sections that do not set it), and mark the keys
// that ARE present consumed, so the unread sweep does not report them. The value that takes effect
// is recorded on its destination key by note_overlay_applied() below.
template <class T, class LoadT = T>
std::optional<T> overlay_read_impl(const ConfigLoader& cl, const std::string& key)
{
    auto v = rc::cfg::Reader::scratch(cl).maybe<T, LoadT>(key, "");
    if (v.has_value())
        rc::cfg::registry().mark_consumed(key, "platform/scenario overlay parse (applied by apply_*)");
    return v;
}

template <class V>
void note_overlay_applied(std::string_view name, const V& value, const std::string& source)
{
    auto& reg = rc::cfg::registry();
    const std::string suffix = "." + std::string(name);
    std::string dest;
    int hits = 0;
    for (const auto& r : reg.records())
        if (r.key.ends_with(suffix) and not r.key.starts_with("Platform.") and not r.key.starts_with("Scenario."))
        {
            dest = r.key;
            ++hits;
        }
    reg.note_overlay(hits == 1 ? dest : source + suffix, rc::cfg::detail::str(value), source);
}
}  // namespace

void load_status_config(const ConfigLoader& cl, RoomConfig& p)
{
    const rc::cfg::Reader reader(cl, "room_concept");
    reader.opt<bool>("Status.Enable", p.STATUS_ENABLE,
            "[Status] — the live status stream a terminal viewer attaches to (tools/room_tui.py, launched by tools/room_run.sh)");
}

void load_room_config(const ConfigLoader& cl, RoomConfig& p,
                      rc::RoomConcept& room_concept, rc::EpistemicController& epistemic)
{
    // Every read below registers its key, its CODE DEFAULT (read out of the target before it is
    // assigned) and a one-line description, so the startup table — and the viewer's Config gates
    // panel — can say where each value came from, not just what it is (common/config_report).
    const rc::cfg::Reader reader(cl, "room_concept");

    // ── RoomConcept params ─────────────────────────────────────────────────
    reader.req<bool>("RoomConcept.PredictionEarlyExit", p.PREDICTION_EARLY_EXIT,
            "Localizer");
    reader.req<int>("RoomConcept.NumIterations", room_concept.params.num_iterations,
            "Balance between speed and convergence");
    reader.req<int>("RoomConcept.WindowSize", room_concept.params.rfe_window_size,
            "Number of past timesteps to retain");
    reader.req<int>("RoomConcept.MaxLidarPoints", room_concept.params.max_lidar_points,
            "Subsample for speed");
    reader.req<int>("RoomConcept.MaxLidarOldSlot", room_concept.params.rfe_max_lidar_per_old_slot,
            "Subsample older slots to save compute");
    // Relocalisation (reloc_search.h; RoomConcept::Params "Relocalisation"). Replaced 2026-09-16:
    // RecoveryLossThreshold, RecoveryConsecutiveCount, RecoveryCooldownFrames, GridSearch*, Symmetry*,
    // HierPrecReloc* and HierPrecEeDthetaMin no longer exist — a leftover key is ignored, so delete it.
    reader.opt<float, double>("RoomConcept.RelocOutlierFrac", room_concept.params.reloc_outlier_frac,
            "ε — clutter / through-door fraction ⚠ modelling constant");
    reader.opt<float, double>("RoomConcept.RelocWallOffsetSigma", room_concept.params.reloc_wall_offset_sigma,
            "m — per-wall common-mode error marginalised from mode Σ");
    reader.opt<float, double>("RoomConcept.RelocBodyClearance", room_concept.params.reloc_body_clearance,
            "m — robot body radius; no pose closer to a wall");
    reader.opt<int>("RoomConcept.RelocMaxPoints", room_concept.params.reloc_max_points,
            "wall-band subsample for the lattice");
    reader.opt<int>("RoomConcept.RelocMaxModes", room_concept.params.reloc_max_modes,
            "mixture budget");
    reader.opt<float, double>("RoomConcept.RelocHazard", room_concept.params.reloc_hazard,
            "per-frame P(kidnap / silent loss) — the HMM transition AND α");
    reader.opt<float, double>("RoomConcept.RelocTrackSMedian", room_concept.params.reloc_track_s_median,
            "m — prior median of s when tracking (measured p50, 09-16)");
    reader.opt<float, double>("RoomConcept.RelocTrackLogSigma", room_concept.params.reloc_track_log_sigma,
            "prior spread of ln s when tracking (p99/p50 = 0.069/0.034)");
    reader.opt<int>("RoomConcept.RelocTrackMemoryFrames", room_concept.params.reloc_track_memory_frames,
            "forgetting timescale of the learnt tracking emission");
    reader.opt<float, double>("RoomConcept.RelocLostSMin", room_concept.params.reloc_lost_s_min,
            "m — support of the lost emission (log-uniform)");
    reader.opt<float, double>("RoomConcept.RelocLostSMax", room_concept.params.reloc_lost_s_max,
            "m");
    reader.opt<bool>("RoomConcept.RelocLegacyGridSearch", room_concept.params.reloc_legacy_grid_search,
            "A/B: true = the pre-09-16 4-stage lattice for the commit");
    reader.opt<bool>("RoomConcept.RelocEnabled", room_concept.params.reloc_enabled,
            "false = relocalisation never moves the pose (lost, mode_switch, seed_validation)");

    reader.opt<float, double>("RoomConcept.OdometryNoiseFactor", p.ODOMETRY_NOISE_FACTOR,
            "");
    reader.opt<bool>("RoomConcept.OdomSampleLog", p.ODOM_SAMPLE_LOG,
            "config: RoomConcept.OdomSampleLog");
    reader.opt<bool>("ProtoRoom.Enabled", p.PROTO_ROOM_ENABLED,
            "config: ProtoRoom.Enabled");
    reader.opt<float, double>("ProtoRoom.BirthProb", p.PROTO_ROOM_BIRTH_PROB,
            "config: ProtoRoom.BirthProb");
    reader.opt<std::string>("RoomConcept.CalibStateFile", p.CALIB_STATE_FILE,
            "config: RoomConcept.CalibStateFile");
    reader.opt<float, double>("RoomConcept.OdomNoiseScale", room_concept.params.odom_noise_scale,
            "Multiplier on all odom noise params (>1 simulates worse odometry)");
    reader.opt<bool>("RoomConcept.DifferentialTest", room_concept.params.differential_test_enabled,
            "Enable shadow single-step evaluator for RFE vs baseline comparison");
    reader.opt<bool>("RoomConcept.SdfCurrentSlotOnly", room_concept.params.sdf_current_slot_only,
            "If true, SDF obs evaluated only for the newest slot");
    // Raise the published covariance to at least what the innovations demonstrate, so the sigma stops
    // being a constant. Default OFF — see Params::adaptive_cov_enabled for the acceptance test and for
    // the two measured process-noise approaches that failed before it.
    reader.opt<bool>("RoomConcept.HessianCheck", room_concept.params.hessian_check,
            "Measure the published sigma against what the window's own normal equations say about the newest pose");
    reader.opt<bool>("RoomConcept.CovarianceFromSolver", room_concept.params.covariance_from_solver,
            "Take the published covariance from the GN solver's own normal equations (the newest pose's Schur-complemented marginal) instead of a double-backward…");
    reader.opt<bool>("RoomConcept.AdaptiveCovEnabled", room_concept.params.adaptive_cov_enabled,
            "");
    reader.opt<float, double>("RoomConcept.AdaptiveCovLambda", room_concept.params.adaptive_cov_lambda,
            "EMA rate for the innovation second moment. 0.02 ~= a 50-frame memory: long enough that one bad frame cannot spike the published sigma (which would…");
    // Strided RFE window — see Params::window_stride_enabled.
    reader.opt<bool>("RoomConcept.WindowStrideEnabled", room_concept.params.window_stride_enabled,
            "===== Strided RFE window ===== The window holds rfe_window_size poses, but at the lidar rate they are all essentially the SAME pose: measured while…");
    reader.opt<float, double>("RoomConcept.WindowMinTravel", room_concept.params.window_min_travel_m,
            "admit a new slot after this much travel");
    reader.opt<float, double>("RoomConcept.WindowMinTurn", room_concept.params.window_min_turn_rad,
            "...or this much turn, whichever comes first");
    // Kinematic clamp on the published pose — see RoomConfig::POSE_CLAMP_ENABLED.
    reader.opt<bool>("RoomConcept.PoseClampEnabled", p.POSE_CLAMP_ENABLED,
            "RoomConcept.PoseClampEnabled");
    reader.opt<float, double>("RoomConcept.PoseClampVMax", p.POSE_CLAMP_V_MAX,
            "m/s — fallback; capability supersedes it");
    reader.opt<float, double>("RoomConcept.PoseClampWMax", p.POSE_CLAMP_W_MAX,
            "rad/s — fallback; capability supersedes it");
    reader.opt<float, double>("RoomConcept.PoseClampMaxDt", p.POSE_CLAMP_MAX_DT_S,
            "s");
    reader.opt<bool>("RoomConcept.PreserveBootstrapRoom", p.PRESERVE_BOOTSTRAP_ROOM,
            "Static-room mode: ADOPT a pre-seeded room/table from the bootstrap graph instead of deleting+recreating it, and do NOT write room pose / robot->room");
    reader.opt<float, double>("RoomConcept.RobotVelCovAdv", p.ROBOT_VEL_COV_ADV,
            "(0.05 m/s)²");
    reader.opt<float, double>("RoomConcept.RobotVelCovSide", p.ROBOT_VEL_COV_SIDE,
            "(0.05 m/s)²");
    reader.opt<float, double>("RoomConcept.RobotVelCovRot", p.ROBOT_VEL_COV_ROT,
            "(0.1 rad/s)²");
    reader.opt<bool>("PredictPublish.enabled", p.PREDICT_PUBLISH_ENABLED,
            "PredictPublish.enabled (drives RT from odometry)");
    reader.opt<float, double>("PredictPublish.max_coast_s", p.PREDICT_PUBLISH_MAX_COAST_S,
            "stop publishing if no lidar correction for this long");
    reader.opt<float, double>("PredictPublish.process_noise_xy", p.PREDICT_PROCESS_NOISE_XY,
            "(m/√s)² → variance growth m²/s on x,y while coasting");
    reader.opt<float, double>("PredictPublish.process_noise_theta", p.PREDICT_PROCESS_NOISE_THETA,
            "(rad/√s)² → variance growth rad²/s on theta");
    reader.opt<float, double>("PredictPublish.blend_gain", p.PREDICT_BLEND_GAIN,
            "fraction of the residual applied per odometry tick");
    reader.opt<float, double>("PredictPublish.max_dt_s", p.PREDICT_MAX_DT_S,
            "clamp per-sample integration dt (a missed/late odom");
    reader.opt<float, double>("PredictPublish.max_blend_step_m", p.PREDICT_MAX_BLEND_STEP_M,
            "max position bleed per tick (→ ≤0.3 m/s @30Hz)");
    reader.opt<float, double>("PredictPublish.max_blend_step_rad", p.PREDICT_MAX_BLEND_STEP_RAD,
            "max heading bleed per tick");
    reader.opt<float, double>("PredictPublish.snap_thresh_m", p.PREDICT_SNAP_THRESH_M,
            "Outlier-aware: a residual bigger than these is a relocalization/jump (the gentle slew would take ~30 s to catch up → the pose strands metres from…");
    reader.opt<float, double>("PredictPublish.snap_thresh_rad", p.PREDICT_SNAP_THRESH_RAD,
            "");
    reader_apply<std::string>(reader, "RoomConcept.OptimizerType", "pose optimiser backend: LBFGS | ADAM | GN", p.OptimizerType, [&](const std::string& optimizer_type)
    {
        p.OptimizerType = optimizer_type;
        room_concept.params.optimizer_type = optimizer_type;
    });
    reader_apply<std::string>(reader, "RoomConcept.RoomLayoutSvg", "layout SVG (a [Scenario.*] overlay may replace it)", p.ROOM_LAYOUT_SVG, [&](const std::string& svg_file)
    {
        p.ROOM_LAYOUT_SVG = svg_file;
    });
    reader.opt<std::string>("RoomConcept.LayoutDir", p.LAYOUT_DIR,
            "The same P3Bot runs in the WAF room and in the apartment; the same apartment hosts P3Bot and Shadow");

    // ── [RoomShape] wall-SLAM: estimate the layout instead of loading it ───────────────────────
    // MapMode is the ONE switch; absent ⇒ "given" ⇒ nothing else in this block has any effect.
    reader_apply<std::string>(reader, "RoomShape.MapMode", "given = load the layout; estimate = learn it from the LiDAR", p.MAP_MODE, [&](const std::string& m)
    {
        p.MAP_MODE = m;
        room_concept.params.map_mode = (m == "estimate") ? rc::RoomConcept::Params::MapMode::Estimate
                                                         : rc::RoomConcept::Params::MapMode::Given;
        if (m != "estimate" and m != "given")
            qWarning() << "[cfg] RoomShape.MapMode" << QString::fromStdString(m) << "is not 'given'|'estimate' — treated as given";
    });
    {
        auto& ws = room_concept.params.wall_seg;
        auto& wm = room_concept.params.wall_map;
        float deg = std::numeric_limits<float>::quiet_NaN();
        reader.opt<float, double>("RoomShape.ManhattanSigmaDeg", deg,
                "constraint: a chamfer is born class-less (no factor) when no class beats ManhattanOffPrior");
        if (std::isfinite(deg)) wm.manhattan_sigma_rad = deg * static_cast<float>(M_PI) / 180.f;
        reader.opt<float, double>("RoomShape.ManhattanOffPrior", wm.manhattan_off_prior,
                "prior mass of 'this wall obeys no class' (a chamfer)");
        reader.opt<float, double>("RoomShape.SensorSigma", ws.sensor_sigma,
                "m — PHYSICAL: LiDAR range noise perpendicular to a wall");
        reader.opt<float, double>("RoomShape.WallMapSigmaD", wm.map_sigma_d,
                "m");
        reader.opt<bool>("RoomShape.FreezeLayoutWhenPublishable", wm.freeze_when_publishable,
                "── FREEZE THE LAYOUT once it first becomes publishable (RoomShape.FreezeLayoutWhenPublishable) An EXPERIMENT switch, not a modelling term, and off by…");
        reader.opt<bool>("RoomShape.RecordWallInput", room_concept.params.record_wall_input,
                "RoomShape.RecordWallInput — DIAGNOSTIC RECORDER (2026-09-17), default OFF");
        reader.opt<bool>("RoomShape.EnableLevel2", wm.enable_level2,
                "── LEVEL 2 — the RESIDUAL PASS (user design 2026-09-02)");
        reader.opt<bool>("RoomShape.WallLineFactor", wm.wall_line_factor,
                "");
        reader.opt<bool>("RoomShape.CommonModeGate", wm.common_mode_gate,
                "`information` accumulates one conditional block per absorbed slot and never decays, so a wall re-observed at 10-20 Hz from nearly one pose is…");

        // ── THE BOX LAYOUT SIDECAR ───────────────────────────────────────────────────────────
        // Off by default. `BoxLayout` decides only WHICH polygon leaves the agent; the wall map
        // keeps running either way, so this is an A/B on the robot rather than a replacement.
        reader.opt<bool>("RoomShape.BoxLayout", room_concept.params.box.enabled,
                "OFF until a watched run says otherwise");
        reader.opt<int>("RoomShape.BoxEveryFrames", room_concept.params.box.every_frames,
                "/< how often structure is re-examined (COMPUTE, not a bar)");
        reader.opt<float, double>("RoomShape.BoxSensorSigma", room_concept.params.box.sensor_sigma,
                "m — PHYSICAL: LiDAR range noise perpendicular to a wall");
        reader.opt<float, double>("RoomShape.BoxSigmaFlat", room_concept.params.box.sigma_flat,
                "m — PHYSICAL: wall flatness / line scatter between views");
        reader.opt<float, double>("RoomShape.BoxCell", room_concept.params.box.cell,
                "COMPUTE: clustering / voxel resolution");
        reader.opt<int>("RoomShape.BoxMinCluster", room_concept.params.box.min_cluster,
                "COMPUTE: smallest cluster worth proposing");
        reader.opt<float, double>("RoomShape.BoxZMin", room_concept.params.box.z_min,
                "/< PHYSICAL: floor rejection, in the robot's own frame");
        reader.opt<float, double>("RoomShape.BoxZMax", room_concept.params.box.z_max,
                "/< PHYSICAL: below the ceiling — see");
        reader.opt<float, double>("RoomShape.BoxObsZMin", room_concept.params.box.obs_z_min,
                "body against it, the route optimiser keeps clear of it) and never as evidence about the room");
        reader.opt<float, double>("RoomShape.BoxObsZMax", room_concept.params.box.obs_z_max,
                "");
        // Model choices validated in tools/wall_slam_selftest (were WS_* env flags; see config.toml).
        reader.opt<bool>("RoomShape.BoxFreeForce", room_concept.params.box.free_force,
                "Log-odds that a cell of the room goes unswept by any beam over a whole tour");
        reader.opt<bool>("RoomShape.BoxGaugeML", room_concept.params.box.gauge_ml,
                "PHYSICAL, robot frame. Returns here are FURNITURE, not walls: the wall band is above them (z_min), which is how the robot separates the two — the…");
        reader.opt<bool>("RoomShape.BoxCoverLayout", room_concept.params.box.cover_layout,
                "Build the free-space cover on the LAYOUT grid, not the map grid (else every proposed box is aligned with the map axes and the judge rewards a frame…");
        reader.opt<bool>("RoomShape.BoxConnected", room_concept.params.box.connected,
                "Refuse a candidate cover that would make the region more disconnected than it is");
        reader.opt<bool>("RoomShape.BoxConnectedPrice", room_concept.params.box.connected_price,
                "PRICE a second region instead of forbidding it: the adoption cost gains one region's description length, 4*log(span/sigma), per extra connected…");
        reader.opt<bool>("RoomShape.BoxSeedPriorSpan", room_concept.params.box.seed_prior_span,
                "A one-scan seed's offsets carry the room's span, not sigma_flat: a box drawn round one scan is not known to a centimetre");
        reader.opt<bool>("RoomShape.BoxRegMapVar", room_concept.params.box.reg_map_var,
                "Pass RegisterOptions::map_var when this channel registers (reproject)");
        reader.opt<int>("RoomShape.BoxCoverMaxRects", room_concept.params.box.cover_max_rects,
                "COMPUTE budget of the greedy cover: how many rectangles it may propose");
        reader.opt<bool>("RoomShape.EnableRederive", wm.enable_rederive,
                "── GLOBAL re-derivation cadence (re_derive): the escape hatch from a wrong local topology runs on a slow clock, or sooner when local jumps are…");
        reader.opt<bool>("RoomShape.TwinFuseInformation", wm.twin_fuse_information,
                "because the gate prevents the duplicate births that create twins in the first place");
        reader.opt<bool>("RoomShape.SpanKillRespectsSupport", wm.span_kill_respects_support,
                "On: one residual per (slot, wall) carrying the segment fit's own information, instead of a per-point sum divided by the slot's point count");
        reader.opt<bool>("RoomShape.AbsorbSchur", wm.absorb_schur,
                "⚠ DEFAULT OFF: MEASURED WORSE, 2026-09-18, on the synthetic driven tour — IoU 0.987 -> 0.886, pose rmse 0.015 -> 0.262 m, map-frame rotation 1.80 ->…");
        reader.opt<float, double>("RoomShape.LineCommonSigmaD", wm.line_common_sigma_d,
                "m");
        float lcp = std::numeric_limits<float>::quiet_NaN();
        reader.opt<float, double>("RoomShape.LineCommonSigmaPhiDeg", lcp,
                "deg");
        if (std::isfinite(lcp)) wm.line_common_sigma_phi = lcp * static_cast<float>(M_PI) / 180.f;
        float msp = std::numeric_limits<float>::quiet_NaN();
        reader.opt<float, double>("RoomShape.WallMapSigmaPhiDeg", msp,
                "deg");
        if (std::isfinite(msp)) wm.map_sigma_phi_rad = msp * static_cast<float>(M_PI) / 180.f;
        reader.opt<float, double>("RoomShape.SegmentChi2", ws.chi2_inlier,
                "χ²₁ @95% — fallback band and the endpoint test");
        reader.opt<float, double>("RoomShape.MergeChi2", ws.chi2_merge,
                "Two polished seeds are the SAME mode when their difference is not resolvable under the pair's combined covariance: χ²₃ at 99%");
        wm.merge_chi2 = ws.chi2_merge;
        reader.opt<float, double>("RoomShape.AssocChi2", wm.assoc_chi2,
                "χ²₂ @95% — association gate on the innovation covariance");
        reader.opt<float, double>("RoomShape.RansacConfidence", ws.ransac_confidence,
                "p in k = log(1−p)/log(1−w²)");
        reader.opt<int>("RoomShape.RansacMaxIters", ws.ransac_max_iters,
                "cap on the adaptive iteration count");
        reader.opt<int>("RoomShape.MinPointsPerSegment", ws.min_points,
                "PCA needs ≥3 points for a residual DOF (resid_var)");
        reader.opt<int>("RoomShape.MaxSegments", ws.max_segments,
                "stop extracting after this many (a room has few walls)");
        reader.opt<float, double>("RoomShape.SensorRange", wm.sensor_range,
                "m — extent of the uniform prior on d (Occam term)");
        reader.opt<float, double>("RoomShape.WallBirthNats", wm.birth_nats,
                "ln 100 — decisive Bayes factor ⚠ decision constant");
        reader.opt<int>("RoomShape.BirthMinFrames", wm.birth_min_frames,
                "a jump needs a second view (tracker-only birth)");
        reader.opt<float, double>("RoomShape.PublishCornerSigma", wm.publish_corner_sigma,
                "m — every derived corner must be this sharp to publish");
        reader.opt<float, double>("RoomShape.RectPriorSigmaD", wm.rect_prior_sigma_d,
                "m — per-side offset prior strength");
        float rps = std::numeric_limits<float>::quiet_NaN();
        reader.opt<float, double>("RoomShape.RectPriorSigmaPhiDeg", rps,
                "deg — an OBB of a non-convex cloud tilts this much; a tighter prior");
        if (std::isfinite(rps)) wm.rect_prior_sigma_phi_rad = rps * static_cast<float>(M_PI) / 180.f;
        reader.opt<float, double>("RoomShape.SpliceEndTol", wm.splice_end_tol,
                "m");
        reader.opt<float, double>("RoomShape.ExistRefutePdet", wm.exist_refute_pdet,
                "P(detect): weight of a pass-through vs a support ⚠");
        reader.opt<float, double>("RoomShape.ExistBinM", wm.exist_bin_m,
                "m — extent bin width (spatial resolution of refutation)");

        reader.opt<float, double>("RoomShape.GaugeSigmaXY", room_concept.params.wall_gauge_sigma_xy,
                "m — first-pose gauge fix (a gauge, not a model term)");
        reader.opt<float, double>("RoomShape.GaugeSigmaTheta", room_concept.params.wall_gauge_sigma_theta,
                "rad");
        reader.opt<int>("RoomShape.WallMaxSlots", room_concept.params.wall_max_slots,
                "wall factors on the newest N slots (0 ⇒ every slot)");
    }
    reader.opt<bool>("RoomConcept.RecenterRoomPolygon", p.RECENTER_ROOM_POLYGON,
            "config: RoomConcept.RecenterRoomPolygon");
    // Ceiling height (m). Sets the room DSR node's room_height attribute (walls/ceiling overlay) and
    // the EXPECTED ceiling location for the LiDAR startup geometry check. Set it explicitly for rooms
    // whose ceiling is above the LiDAR's vertical reach (e.g. 3 m), where the check can't detect it.
    reader.opt<float, double>("RoomConcept.RoomHeight", p.room_height,
            "m, room DSR node attribute");

    // Media plane (RGB for the camera window + LiDAR for LidarIngestor). DDS domain +
    // topics are read from the producer's media descriptor on the graph, not config.
    reader.opt<bool>("Media.lidar_use_media", p.LIDAR_USE_MEDIA,
            "false ⇒ DSR graph laser_* only");
    reader.opt<int>("Media.lidar_stall_timeout_ms", p.LIDAR_STALL_TIMEOUT_MS,
            "Operating: no sweep for this long ⇒ back to Waiting");
    reader.opt<int>("Media.lidar_wait_log_period_ms", p.LIDAR_WAIT_LOG_PERIOD_MS,
            "Waiting: how often to reprint why we are still waiting");
    reader.opt<std::string>("Media.lidar_helios_name", p.LIDAR_HELIOS_NAME,
            "Lidar Per-device high LiDAR plane: points arrive in the DEVICE frame (metres) and must be transformed device->robot via the DSR RT tree.…");
    reader.opt<std::string>("Media.lidar_robot_frame", p.LIDAR_ROBOT_FRAME,
            "empty ⇒ auto-derived from the type-'robot' node at init");
    reader.opt<float, double>("Media.lidar_high_min_height", p.LIDAR_HIGH_MIN_HEIGHT,
            "m");
    reader.opt<float, double>("Media.lidar_high_max_height", p.LIDAR_HIGH_MAX_HEIGHT,
            "m — upper bound of the high band (excludes the ceiling)");
    reader.opt<bool>("Media.lidar_startup_geometry_check", p.LIDAR_STARTUP_GEOMETRY_CHECK,
            "Startup geometry self-check: from the first few sweeps, detect the floor plane (warn if it disagrees with the robot mount geometry -> a mis-set LiDAR…");
    reader.opt<int>("Media.lidar_startup_check_sweeps", p.LIDAR_STARTUP_CHECK_SWEEPS,
            "sweeps to accumulate before running the check");
    reader.opt<float, double>("Media.lidar_floor_tolerance", p.LIDAR_FLOOR_TOLERANCE,
            "m — warn if the measured floor is off by more");
    reader.opt<float, double>("Media.lidar_ceiling_margin", p.LIDAR_CEILING_MARGIN,
            "m — keep wall points this far below the ceiling");

    // Camera-overlay object projection: comma-separated DSR node types (e.g. "object,table,cylinder,chair").
    reader_apply<std::string>(reader, "Overlay.ObjectTypes", "DISPLAY ONLY: node types drawn as boxes on the camera overlay (comma-separated)", std::string("object"), [&](const std::string& csv)
    {
        std::vector<std::string> types;
        std::size_t start = 0;
        while (start <= csv.size())
        {
            const std::size_t comma = csv.find(',', start);
            const std::size_t end = (comma == std::string::npos) ? csv.size() : comma;
            std::string tok = csv.substr(start, end - start);
            const auto l = tok.find_first_not_of(" \t");
            const auto r = tok.find_last_not_of(" \t");
            if (l != std::string::npos)
                types.push_back(tok.substr(l, r - l + 1));
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        if (not types.empty())
            p.OVERLAY_OBJECT_TYPES = std::move(types);
    });

    reader.opt<bool>("RoomConcept.MotionCalibEnabled", room_concept.params.motion_calib.enabled,
            "OFF until a watched run says otherwise");
    reader.opt<bool>("RoomConcept.MotionCalibApply", room_concept.params.motion_calib.apply,
            "false severs the estimator's output from the odometry while leaving everything else running: episodes still close, the window still fills, the solve…");
    reader.opt<int>("RoomConcept.MotionCalibApplyMask", room_concept.params.motion_calib.apply_mask,
            "WHICH parameters may act, as a bitmask over rc::calib::Param (bit 0 = k_v, 1 = eps_yaw, 2 = k_omega, 3 = b_omega, 4 = k_lat, 5 = dk_wheel, 6 =…");
    reader.opt<bool>("RoomConcept.ImuLinearInjection", room_concept.params.imu_linear_injection,
            "Add the accelerometer's within-segment double integration to the wheel displacement. integrate_odometry_over_window holds a 10 Hz wheel velocity FLAT…");
    reader.opt<bool>("RoomConcept.HeadingFusion", room_concept.params.heading_fusion,
            "true: every segment's heading is the precision-weighted mean of the wheel factor and the gyro factor, each corrected by ITS OWN calibration (wheels:…");
    reader.opt<bool>("RoomConcept.HeadingNoiseLearning", room_concept.params.heading_noise_learning,
            "Learn the wheels' heading-noise density as d_w^2 = c0 + c1|v| + c2|omega| from the wheel-minus-gyro disagreement (truth cancels), anchored on the…");
    reader.opt<bool>("RoomConcept.OdomVarianceInjection", room_concept.params.odom_variance_injection,
            "OFF by default, like every channel that can move the published pose");
    reader.opt<float, double>("RoomConcept.MotionCalibYawP0", room_concept.params.motion_calib.yaw_p0,
            "(rad)^2 -> 1 sigma ~ 0.57 deg");
    reader.opt<float, double>("RoomConcept.MotionCalibYawQ", room_concept.params.motion_calib.yaw_q,
            "(rad)^2 per update");
    reader.opt<float, double>("RoomConcept.MotionCalibScaleP0", room_concept.params.motion_calib.scale_p0,
            "fractional^2 -> 1 sigma ~ 2%");
    reader.opt<float, double>("RoomConcept.MotionCalibScaleQ", room_concept.params.motion_calib.scale_q,
            "");
    reader.opt<float, double>("RoomConcept.MotionCalibRotModelSigma", room_concept.params.motion_calib.rot_model_sigma,
            "m per rad of turning, added in quadrature to the position R");
    reader.opt<float, double>("RoomConcept.MotionCalibFitModelGain", room_concept.params.motion_calib.fit_model_gain,
            "multiplies mean |SDF| over the episode, into the position R");
    reader.opt<float, double>("RoomConcept.MotionCalibLeverSigma", room_concept.params.motion_calib.lever_sigma,
            "m. Prior on the helios lever arm (spec §5); learnt only from ROTATION (plan 2026-10-05 Task 1)");
    reader.opt<bool>("RoomConcept.JointCalibMonitor", p.JOINT_CALIB_MONITOR,
            "LOG ONLY: one joint posterior over odometry + helios mount + camera mounts per motion solve -> tmp/joint_calib/ (plan 2026-10-05 Task 4)");
    // ── helios mount correction + sim-only injection at the LiDAR ingestor (plan 2026-10-05 Task 5) ──
    reader.opt<bool>("RoomConcept.LidarMountApply", p.LIDAR_MOUNT_APPLY,
            "apply the motion calibrator's ACTING helios (lever_x, lever_y, eps_yaw/kEps) to the LiDAR points; OFF until validated (plan 2026-10-05 Task 5/6)");
    reader.opt<float, double>("RoomConcept.LidarMountInjectX", p.LIDAR_MOUNT_INJECT_X,
            "m. SIMULATION ONLY: planted helios mount error (sensor displacement, body X)");
    reader.opt<float, double>("RoomConcept.LidarMountInjectY", p.LIDAR_MOUNT_INJECT_Y,
            "m. SIMULATION ONLY: planted helios mount error (sensor displacement, body Y)");
    reader.opt<float, double>("RoomConcept.LidarMountInjectYawDeg", p.LIDAR_MOUNT_INJECT_YAW_DEG,
            "deg. SIMULATION ONLY: planted helios mount error (sensor CCW yaw)");
    // The lever acts and the yaw moves to the LiDAR side exactly when the correction is applied there,
    // so p_applied records what acts and the odometry never applies the same yaw a second time.
    room_concept.params.motion_calib.apply_lever    = p.LIDAR_MOUNT_APPLY;
    room_concept.params.motion_calib.lidar_side_yaw = p.LIDAR_MOUNT_APPLY;
    if (p.LIDAR_MOUNT_INJECT_X != 0.f or p.LIDAR_MOUNT_INJECT_Y != 0.f or p.LIDAR_MOUNT_INJECT_YAW_DEG != 0.f)
        qWarning().nospace() << "[cfg] ⚠ LidarMountInject* is NONZERO (x " << p.LIDAR_MOUNT_INJECT_X
                             << " m, y " << p.LIDAR_MOUNT_INJECT_Y << " m, yaw " << p.LIDAR_MOUNT_INJECT_YAW_DEG
                             << " deg): a helios mount error is being PLANTED in the LiDAR points. "
                                "SIMULATION ONLY -- set all three to 0 on the real robot.";
    // ★ EXPOSED 2026-09-01 so the episode LENGTH can be A/B'd without a rebuild between legs.
    // These are identifiability triggers, not tuning: below them the Jacobian rows are ~0. But
    // information in a channel goes as (accumulated covariate)^2 while episodes arrive at 1/T, so
    // where the trigger sits sets how much a given motion teaches — a 3.14 rad hairpin closing as
    // ONE episode is worth 15.4x the same hairpin chopped at 0.20 rad. See EXPERIMENT.md §12.
    reader.opt<float, double>("RoomConcept.MotionCalibEpisodeMinTrans", room_concept.params.motion_calib.episode_min_trans,
            "/< m of accumulated forward travel");
    reader.opt<float, double>("RoomConcept.MotionCalibEpisodeMinRot", room_concept.params.motion_calib.episode_min_rot,
            "/< rad of accumulated rotation (~11 deg)");

    reader.opt<float, double>("RoomConcept.SigmaSdf", room_concept.params.sigma_sdf,
            "m — SDF observation noise (RoomConcept::Params::sigma_sdf)");
    reader.opt<float, double>("RoomConcept.PredictionTrustFactor", room_concept.params.prediction_trust_factor,
            "Base threshold = sigma_sdf * factor (~7.5cm)");
    reader.opt<int>("RoomConcept.MinTrackingSteps", room_concept.params.min_tracking_steps,
            "Wait for system to stabilize before early exit");
    reader.opt<float, double>("RoomConcept.RotationSdfCoupling", room_concept.params.rotation_sdf_coupling,
            "m/rad — extra SDF tolerance per radian of rotation");

    reader.opt<float, double>("RoomConcept.LbfgsLr", room_concept.params.lbfgs_lr,
            "Initial step size");
    reader.opt<int>("RoomConcept.LbfgsHistorySize", room_concept.params.lbfgs_history_size,
            "(s,y) pairs kept for H^-1 approximation");
    reader.opt<double>("RoomConcept.LbfgsToleranceGrad", room_concept.params.lbfgs_tolerance_grad,
            "Stop when ||grad||_inf < tol");
    reader.opt<double>("RoomConcept.LbfgsToleranceChange", room_concept.params.lbfgs_tolerance_change,
            "Stop when |Δloss| < tol");

    reader.opt<float, double>("RoomConcept.LearningRatePos", room_concept.params.learning_rate_pos,
            "Uniform LR for all pose DoF (x, y, θ)");
    reader.opt<float, double>("RoomConcept.ObsSigma", room_concept.params.rfe_obs_sigma,
            "σ_obs for SDF observation noise (m)");
    reader.opt<float, double>("RoomConcept.HuberDelta", room_concept.params.rfe_huber_delta,
            "Huber threshold (m)");
    reader.opt<float, double>("RoomConcept.ConvergenceRelTol", room_concept.params.convergence_relative_tol,
            "Relative loss-change stopping criterion");
    reader.opt<int>("RoomConcept.ConvergenceMinIters", room_concept.params.convergence_min_iters,
            "Minimum iterations before convergence check");

    reader.opt<bool>("RoomConcept.BoundaryQualityGate", room_concept.params.rfe_boundary_quality_gate,
            "===== Boundary Prior Quality Gate ===== When the previous frame's localization was poor (sdf_mse_prev > sigma_sdf), the boundary prior anchors ADAM…");
    reader.opt<float, double>("RoomConcept.BoundaryHessianQualityThreshold", room_concept.params.boundary_hessian_quality_threshold,
            "m — above this: motion-only Hessian");
    reader.opt<float, double>("RoomConcept.BoundaryMuQualityThreshold", room_concept.params.boundary_mu_quality_threshold,
            "m — above this: keep previous mu");
    reader.opt<float, double>("RoomConcept.EigenvalueClampBoundaryMax", room_concept.params.eigenvalue_clamp_boundary_max,
            "Max eigenvalue — prevents over-confident prior from contaminated scans");
    reader.opt<int>("RoomConcept.TorchNumThreads", room_concept.params.torch_num_threads,
            "intra-op CPU threads (was hard-coded 5 = the 4-core budget)");
    reader.opt<bool>("RoomConcept.BoundaryFejSchur", room_concept.params.boundary_fej_schur,
            "===== FEJ + Schur marginalization boundary prior (replaces Solutions B/C when ON) ===== The legacy boundary prior above is a rough approximation to…");
    reader.opt<float, double>("RoomConcept.BoundaryQualitySigma", room_concept.params.boundary_quality_sigma,
            "m — σ_q for the soft obs-quality weight");

    // Hierarchical precision on the boundary prior (HIERARCHICAL_PRECISION.md) — default OFF.
    reader.opt<bool>("RoomConcept.HierPrecBoundaryEnabled", room_concept.params.hier_prec_boundary_enabled,
            "===== Hierarchical precision on the boundary prior (HIERARCHICAL_PRECISION.md) ===== When enabled, boundary_weight is no longer min(1,…");
    reader.opt<float, double>("RoomConcept.HierPrecU0", room_concept.params.hier_prec_u0,
            "log-precision prior mean (exp(0)=1 → full prior)");
    reader.opt<float, double>("RoomConcept.HierPrecSigmaU2", room_concept.params.hier_prec_sigma_u2,
            "how far r_b may pull u from g(v)");
    reader.opt<float, double>("RoomConcept.HierPrecLrU", room_concept.params.hier_prec_lr_u,
            "π-step size (fast, per-frame)");
    reader.opt<float, double>("RoomConcept.HierPrecGGain", room_concept.params.hier_prec_g_gain,
            "k in g(v)=u0+k·v (top-down map)");
    reader.opt<float, double>("RoomConcept.HierPrecLrV", room_concept.params.hier_prec_lr_v,
            "v-step size (slow hyper-state)");
    reader.opt<float, double>("RoomConcept.HierPrecSigmaV2", room_concept.params.hier_prec_sigma_v2,
            "prior scale on the map_trust state v");

    reader.opt<bool>("RoomConcept.VelocityAdaptiveWeights", room_concept.params.velocity_adaptive_weights,
            "===== Velocity-Adaptive Gradient Weights ===== Adjust optimization emphasis based on current motion profile");
    reader.opt<float, double>("RoomConcept.LinearVelocityThreshold", room_concept.params.linear_velocity_threshold,
            "m/s - below this = 'not moving linearly'");
    reader.opt<float, double>("RoomConcept.AngularVelocityThreshold", room_concept.params.angular_velocity_threshold,
            "rad/s - used for velocity-adaptive gradient weights");
    reader.opt<float, double>("RoomConcept.WeightBoostFactor", room_concept.params.weight_boost_factor,
            "Multiplier for emphasized parameters");
    reader.opt<float, double>("RoomConcept.WeightReductionFactor", room_concept.params.weight_reduction_factor,
            "Multiplier for de-emphasized parameters");
    reader.opt<float, double>("RoomConcept.WeightSmoothingAlpha", room_concept.params.weight_smoothing_alpha,
            "EMA smoothing for weight transitions");

    // Master switch for the command (joystick / controller) motion prior — see
    // Params::use_command_velocity_prior. Off ⇒ the channel is still computed and logged but never
    // enters the prediction, so motion_prior_source collapses to measured / fallback_zero.
    reader.opt<bool>("RoomConcept.UseCommandVelocityPrior", room_concept.params.use_command_velocity_prior,
            "===== Dual-Prior Fusion (command + odometry) ===== Master switch for the COMMAND (joystick / controller) channel as a motion prior");
    reader.opt<float, double>("RoomConcept.CmdNoiseTrans", room_concept.params.cmd_noise_trans,
            "Fractional position noise per meter of motion");
    reader.opt<float, double>("RoomConcept.CmdNoiseRot", room_concept.params.cmd_noise_rot,
            "Fractional rotation noise per radian of COMMANDED rotation. 0.18, not 0.10, kept on measured outcome: it shifts fusion weight to the encoder (the…");
    reader.opt<float, double>("RoomConcept.CmdNoiseBase", room_concept.params.cmd_noise_base,
            "Base position noise even when stationary (m)");
    reader.opt<float, double>("RoomConcept.OdomNoiseTrans", room_concept.params.odom_noise_trans,
            "Fractional position noise per meter of motion");
    reader.opt<float, double>("RoomConcept.OdomNoiseRot", room_concept.params.odom_noise_rot,
            "Fractional rotation noise per radian of rotation");
    reader.opt<float, double>("RoomConcept.OdomNoiseBase", room_concept.params.odom_noise_base,
            "Base position noise even when stationary (m)");
    reader.opt<float, double>("RoomConcept.EncoderRotSlipK", room_concept.params.encoder_rot_slip_k,
            "dimensionless: fraction of the rotation that slips");
    reader.opt<float, double>("RoomConcept.StationaryMotionThreshold", room_concept.params.stationary_motion_threshold,
            "meters; below this = stationary for covariance");
    reader.opt<bool>("RoomConcept.ZuptEnabled", room_concept.params.zupt_enabled,
            "ZUPT gate for the gyro heading override in integrate_odometry_over_window(): below these WHEEL-reported rates the segment is treated as truly…");
    reader.opt<float, double>("RoomConcept.ZuptWheelRotEps", room_concept.params.zupt_wheel_rot_eps,
            "rad/s");
    reader.opt<float, double>("RoomConcept.ZuptWheelLinEps", room_concept.params.zupt_wheel_lin_eps,
            "m/s (adv, side)");

    // ── Preintegrated motion covariance — see Params::motion_preintegration, se2_preintegration.h ──
    // The densities are DERIVED from the legacy constants just loaded, not hard-coded, so the two
    // models cannot silently drift apart: changing EncoderRotSlipK or OdomNoiseBase moves both. That
    // matters because the whole point of the A/B is that the flag changes the SHAPE of the covariance,
    // not its magnitude at the operating point — and a stale hard-coded default would quietly turn it
    // into a re-tuning as well, which is the confound that makes an A/B unreadable.
    //
    //   sigma = (legacy per-frame std) / sqrt(dt_ref)   — a per-frame constant becomes a density
    //   scale = the legacy FRACTIONAL terms, which were always describing a correlated scale error
    //
    // dt_ref is the update interval the legacy constants were tuned at (measured median 42-50 ms).
    // Any explicit Preint* key below overrides the derived value, so a calibrated measurement of the
    // real odometry stream can replace this translation without touching code.
    {
        constexpr float dt_ref = 0.05f;                      // s — the interval the legacy tuning assumed
        const float inv_sqrt_dt = 1.f / std::sqrt(dt_ref);
        auto& po = room_concept.params.odom_preint_noise;
        auto& pc = room_concept.params.cmd_preint_noise;
        const auto& p0 = room_concept.params;

        // Translation floor: the legacy value actually in force when the robot is barely moving is
        // StationaryMotionThreshold (0.02 m live — raised from 0.001 to stop loss_motion spiking to
        // 6000 on a 3.4 cm parked residual), which is the branch the robot is in for >99% of frames.
        // Take the LOOSER of it and OdomNoiseBase so the derived floor is never tighter than either.
        const float odom_floor = std::max(p0.stationary_motion_threshold, p0.odom_noise_base);
        po.sigma_v_lat  = odom_floor * inv_sqrt_dt;
        po.sigma_v_long = odom_floor * inv_sqrt_dt;
        po.sigma_omega  = p0.rotation_noise_base * inv_sqrt_dt;
        po.scale_v      = p0.odom_noise_trans;
        // OdomNoiseRot and EncoderRotSlipK are BOTH fractions of the rotation increment and the legacy
        // model combines them in quadrature, so the equivalent single scale sigma is their hypot.
        po.scale_omega  = std::hypot(p0.odom_noise_rot, p0.encoder_rot_slip_k);

        const float cmd_floor = std::max(p0.stationary_motion_threshold, p0.cmd_noise_base);
        pc.sigma_v_lat  = cmd_floor * inv_sqrt_dt;
        pc.sigma_v_long = cmd_floor * inv_sqrt_dt;
        pc.sigma_omega  = p0.rotation_noise_base * inv_sqrt_dt;
        pc.scale_v      = p0.cmd_noise_trans;
        pc.scale_omega  = p0.cmd_noise_rot;
    }
    reader.opt<bool>("RoomConcept.MotionPreintegration", room_concept.params.motion_preintegration,
            "propagate the motion covariance from densities per sqrt(s) instead of the per-cycle legacy diagonal");
    {
        auto &po = room_concept.params.odom_preint_noise;
        reader.opt<bool>("RoomConcept.MotionNoiseProportional", po.motion_proportional,
                "odometry noise per METRE rolled and per RADIAN turned (se2_preintegration.h), not per second");
        reader.opt<float, double>("RoomConcept.PreintKLong", po.k_long, "m² per m rolled — forward");
        reader.opt<float, double>("RoomConcept.PreintKLat", po.k_lat, "m² per m rolled — lateral");
        reader.opt<float, double>("RoomConcept.PreintKLatTurn", po.k_lat_turn, "m² per rad turned — lateral");
        reader.opt<float, double>("RoomConcept.PreintKThTurn", po.k_th_turn, "rad² per rad turned — heading");
        reader.opt<float, double>("RoomConcept.PreintKTimeTrans", po.k_t_trans, "m² per s — model translation drift");
        reader.opt<float, double>("RoomConcept.PreintKTimeRot", po.k_t_rot, "rad² per s — model heading drift");
        reader.opt<bool>("RoomConcept.MotionNoiseLearn", room_concept.params.motion_noise_learn,
                "learn the coefficients above from the localiser's corrections (motion_noise_vc.h)");
        reader.opt<double>("RoomConcept.MotionNoiseMemory", room_concept.params.motion_noise_memory,
                "corrections of evidence the noise learner remembers");
        reader.opt<bool>("RoomConcept.MotionNoiseInnov", room_concept.params.motion_noise_innov,
                "learn + log the noise components from the scan-to-scan innovation (motion_noise_innov.h)");
        reader.opt<bool>("RoomConcept.PoseFieldBias", room_concept.params.pose_field_bias,
                "learn + log the slow pose-field bias of the scan-only pose (pose_field_bias.h)");
        reader.opt<bool>("RoomConcept.PoseFieldPublish", room_concept.params.pose_field_publish,
                "add the pose-field bias covariance to the PUBLISHED pose covariance");
        reader.opt<double>("RoomConcept.PoseFieldSigma0", room_concept.params.pose_field_sigma0,
                "m, prior sigma of the pose-field bias (50 pseudo-pairs)");
    }
    reader.opt<float, double>("RoomConcept.PreintOdomSigmaVLat", room_concept.params.odom_preint_noise.sigma_v_lat,
            "m/√s — lateral velocity noise floor (0.001 m / √0.05 s)");
    reader.opt<float, double>("RoomConcept.PreintOdomSigmaVLong", room_concept.params.odom_preint_noise.sigma_v_long,
            "m/√s — forward velocity noise floor");
    reader.opt<float, double>("RoomConcept.PreintOdomSigmaOmega", room_concept.params.odom_preint_noise.sigma_omega,
            "rad/√s — yaw-rate noise floor (0.010 rad / √0.05 s)");
    reader.opt<float, double>("RoomConcept.PreintOdomScaleV", room_concept.params.odom_preint_noise.scale_v,
            "fraction of |Δp| that is a consistent scale error");
    reader.opt<float, double>("RoomConcept.PreintOdomScaleOmega", room_concept.params.odom_preint_noise.scale_omega,
            "fraction of |Δθ| that is a consistent scale error");
    reader.opt<float, double>("RoomConcept.PreintCmdSigmaVLat", room_concept.params.cmd_preint_noise.sigma_v_lat,
            "m/√s — lateral velocity noise floor (0.001 m / √0.05 s)");
    reader.opt<float, double>("RoomConcept.PreintCmdSigmaVLong", room_concept.params.cmd_preint_noise.sigma_v_long,
            "m/√s — forward velocity noise floor");
    reader.opt<float, double>("RoomConcept.PreintCmdSigmaOmega", room_concept.params.cmd_preint_noise.sigma_omega,
            "rad/√s — yaw-rate noise floor (0.010 rad / √0.05 s)");
    reader.opt<float, double>("RoomConcept.PreintCmdScaleV", room_concept.params.cmd_preint_noise.scale_v,
            "fraction of |Δp| that is a consistent scale error");
    reader.opt<float, double>("RoomConcept.PreintCmdScaleOmega", room_concept.params.cmd_preint_noise.scale_omega,
            "fraction of |Δθ| that is a consistent scale error");

    // ── ZUPT: the rest hypothesis as a factor (see the NoiseModel block in se2_preintegration.h) ──
    // One set of keys drives BOTH channels: "the body is at rest" is a statement about the robot, not
    // about which stream happened to report it, and letting the two disagree would mean the prior's
    // parked stiffness depended on which channel won selection that frame.
    {
        auto& po = room_concept.params.odom_preint_noise;
        auto& pc = room_concept.params.cmd_preint_noise;
        reader.opt<bool>("RoomConcept.PreintZupt", po.zupt_enabled,
                "ZUPT gate for the gyro heading override in integrate_odometry_over_window(): below these WHEEL-reported rates the segment is treated as truly…");
        reader.opt<bool>("RoomConcept.PreintZuptAsFactor", room_concept.params.zupt_as_factor,
                "===== Preintegrated motion covariance (see se2_preintegration.h) ===== When true, the motion factor's covariance is PROPAGATED through the interval's…");
        // The two forms are EXCLUSIVE: running both counts one hypothesis twice, once shaping the
        // prior's covariance and once as its own factor. The flag therefore turns the per-sample
        // shaping off rather than leaving the caller to remember.
        reader.opt<bool>("RoomConcept.SdfPolishOnEarlyExit", room_concept.params.sdf_polish_enabled,
                "Apply the rest hypothesis to the PREDICTED INCREMENT, at prior selection — the only place it reaches the ~99% of cycles that publish without the…");
        reader.opt<bool>("RoomConcept.PreintZuptOnPrediction", room_concept.params.zupt_on_prediction,
                "parked, 3.1e-4 after driving, vs 4.5e-4 measured from wheel increments");
        reader.opt<bool>("RoomConcept.PreintZuptPredLearnRest", room_concept.params.zupt_pred_learn_rest,
                "Decide rest vs moving with RestMotionChannel (rest_density_learner.h): a velocity state per body-frame channel whose noise is learnt from its own…");
        reader.opt<float, double>("RoomConcept.PreintZuptPredVMax", room_concept.params.zupt_pred_v_max,
                "/< m/s — width of the 'moving' hypothesis, not a limit");
        reader.opt<float, double>("RoomConcept.PreintZuptPredWMax", room_concept.params.zupt_pred_w_max,
                "");
        po.zupt_as_factor = room_concept.params.zupt_as_factor;
        pc.zupt_as_factor = room_concept.params.zupt_as_factor;
        // ★ The keys were RENAMED with their units on 2026-08-26 (per-sample sigma -> density). The
        // old names are read into a sentinel purely so a config still carrying them FAILS LOUDLY
        // instead of having them silently ignored — which would leave that platform on the header
        // default while its config file appeared to say otherwise. A silently orphaned key is worse
        // than a missing one: the file documents an intent that nothing implements.
        {
            constexpr float kSentinel = -12345.f;
            float legacy_v = kSentinel, legacy_w = kSentinel;
            reader.opt<float, double>("RoomConcept.PreintZuptSigmaV", legacy_v,
                    "");
            reader.opt<float, double>("RoomConcept.PreintZuptSigmaOmega", legacy_w,
                    "");
            if (legacy_v != kSentinel or legacy_w != kSentinel)
                throw std::runtime_error(
                    "config uses PreintZuptSigmaV/Omega, which are PER-SAMPLE standard deviations and "
                    "no longer exist. They are now PreintZuptDensityV/Omega, in m/sqrt(s) and "
                    "rad/sqrt(s). Convert with density = sigma * sqrt(publish_period_s) to preserve "
                    "today's behaviour at today's rate, or re-measure with tools/odom_whiteness.py.");
        }
        reader.opt<float, double>("RoomConcept.PreintZuptDensityV", po.zupt_density_v,
                "m/√s — measured, P3Bot, 8166 parked samples");
        reader.opt<float, double>("RoomConcept.PreintZuptDensityOmega", po.zupt_density_omega,
                "rad/√s — measured, P3Bot, same run");
        reader.opt<float, double>("RoomConcept.PreintZuptLeverM", po.zupt_lever_m,
                "Lever arm converting between the two channels, so the rest hypothesis is about the WHOLE BODY: a robot pivoting in place has v_lat = v_long = 0 yet…");
        pc.zupt_enabled     = po.zupt_enabled;
        pc.zupt_density_v   = po.zupt_density_v;
        pc.zupt_density_omega = po.zupt_density_omega;
        pc.zupt_lever_m     = po.zupt_lever_m;
    }


    reader.opt<bool>("RoomConcept.EnableCornerTracking", room_concept.params.enable_corner_tracking,
            "Master switch for corner factors in Adam loss");
    // Graded-covariance corner factor (replaces the old hard rej_angle/rej_orient gates).
    reader.opt<float, double>("RoomConcept.CornerWallBand", room_concept.params.corner_wall_band,
            "perpendicular gather band (m) — MUST exceed model misfit");
    reader.opt<float, double>("RoomConcept.CornerBaseSigma", room_concept.params.corner_base_sigma,
            "detection noise floor σ0 (m) per wall");
    reader.opt<float, double>("RoomConcept.CornerOrientTauDeg", room_concept.params.corner_orient_tau_deg,
            "smooth orientation-trust scale (deg)");
    reader.opt<float, double>("RoomConcept.CornerMergeChi2", room_concept.params.corner_merge_chi2,
            "χ²₂ @95%; 0 disables the exclusion test");
    reader.opt<float, double>("RoomConcept.CornerMergePriorSigma", room_concept.params.corner_merge_prior_sigma,
            "m; 0 ⇒ use the detector's search_radius");
    reader.opt<float, double>("RoomConcept.CornerMinWallMapSigmas", room_concept.params.corner_min_wall_map_sigmas,
            "Landmark admissibility in units of the detector's map_sigma — a vertex whose shorter adjacent wall is below this is not a feature the traced layout…");
    reader.opt<bool>("RoomConcept.CornerStatsCsv", room_concept.params.corner_stats_csv,
            "evidence budget in 'uninformative votes', not a time delay");
    reader.opt<float, double>("RoomConcept.CornerMinYieldMapSigmas", room_concept.params.corner_min_yield_map_sigmas,
            "Landmark retirement by observed information — a corner whose weakest-axis σ never gets below this many map_sigmas is one the robot cannot measure…");
    reader.opt<float, double>("RoomConcept.CornerYieldLeak", room_concept.params.corner_yield_leak,
            "per matched frame (~2.5 s at 20 Hz)");
    reader.opt<int>("RoomConcept.CornerYieldWarmup", room_concept.params.corner_yield_warmup,
            "MATCHED frames tolerated before retiring — an");
    reader.opt<float, double>("RoomConcept.CornerYieldReleaseFactor", room_concept.params.corner_yield_release_factor,
            "hysteresis: release above this × the retire bar");
    reader.opt<float, double>("RoomConcept.CornerPrecisionGain", room_concept.params.corner_precision_gain,
            "global scale on corner precision vs the SDF term");
    reader.opt<float, double>("RoomConcept.CornerHuberSigma", room_concept.params.corner_huber_sigma,
            "Huber saturation in WHITENED (σ) units, not meters");
    // Gauss-Newton / Levenberg-Marquardt backend (room_gn_solver.h). All OFF by default: GnShadow
    // only LOGS, and driving the pose with it additionally requires OptimizerType = "GN".
    reader.opt<bool>("RoomConcept.GnShadow", room_concept.params.gn_shadow,
            "");
    reader.opt<bool>("RoomConcept.GnGradCheck", room_concept.params.gn_grad_check,
            "Finite-difference check of the analytic Jacobian, logged as grad_relerr in the shadow CSV");
    reader.opt<std::string>("RoomConcept.GnShadowCsv", room_concept.params.gn_shadow_csv_path,
            "");
    reader.opt<int>("RoomConcept.GnMaxIters", room_concept.params.gn_max_iters,
            "");
    reader.opt<float, double>("RoomConcept.GnLambdaInit", room_concept.params.gn_lambda_init,
            "Levenberg damping relative to diag(H)");
    reader.opt<float, double>("RoomConcept.GnStepTol", room_concept.params.gn_step_tol,
            "‖δ‖∞ (m / rad) convergence test");
    reader.opt<float, double>("RoomConcept.GnLossRelTol", room_concept.params.gn_loss_rel_tol,
            "relative loss-improvement convergence test");

    reader.opt<bool>("RoomConcept.CornerEarlyExitCheck", room_concept.params.corner_early_exit_check,
            "OFF by default — flip on to A/B");
    reader.opt<float, double>("RoomConcept.CornerEarlyExitSigma", room_concept.params.corner_early_exit_sigma,
            "per-corner whitened-residual (σ) disagreement tolerance");
    reader.opt<int>("RoomConcept.CornerEarlyExitMinBad", room_concept.params.corner_early_exit_min_bad,
            "# of disagreeing corners required to force Adam");

    // Object anchors (validated modelled objects as SE(2) pose landmarks). Loaded into BOTH the
    // shared config (read by RoomSceneGraph's graph-side gather) and the localizer params.
    reader.opt<bool>("ObjectAnchor.enable", p.OBJECT_ANCHOR_ENABLE,
            "ObjectAnchor.enable");
    reader.opt<bool>("ObjectAnchor.optimizeLandmark", p.OBJECT_ANCHOR_OPTIMIZE_LANDMARK,
            "ObjectAnchor.optimizeLandmark — p_o as a private");
    // Guarded by exists(): ConfigLoader throws on `key = []`, and a silently-empty list would disable
    // every landmark while the enable flag still read true — a confusing way to get nothing.
    {
        auto v = p.OBJECT_ANCHOR_SUBTYPES;
        reader.opt("ObjectAnchor.subtypes", v, "object subtypes accepted as pose landmarks");
        if (not v.empty()) p.OBJECT_ANCHOR_SUBTYPES = std::move(v);
    }
    reader.opt<float, double>("ObjectAnchor.weight", p.OBJECT_ANCHOR_WEIGHT,
            "ObjectAnchor.weight (keep < walls)");
    reader.opt<float, double>("ObjectAnchor.huber", p.OBJECT_ANCHOR_HUBER,
            "ObjectAnchor.huber (whitened σ units)");
    reader.opt<int>("ObjectAnchor.maxSlots", p.OBJECT_ANCHOR_MAX_SLOTS,
            "ObjectAnchor.maxSlots");
    reader.opt<float, double>("ObjectAnchor.measSigmaXY", p.OBJECT_ANCHOR_MEAS_SIG_XY,
            "ObjectAnchor.measSigmaXY (m) fallback R_o");
    reader.opt<float, double>("ObjectAnchor.measSigmaYaw", p.OBJECT_ANCHOR_MEAS_SIG_YAW,
            "ObjectAnchor.measSigmaYaw (rad) fallback R_o");
    reader.opt<float, double>("ObjectAnchor.earlyExitSigma", p.OBJECT_ANCHOR_EARLY_EXIT_SIGMA,
            "ObjectAnchor.earlyExitSigma — whitened anchor-residual");
    reader.opt<float, double>("ObjectAnchor.validateSigma", p.OBJECT_ANCHOR_VALIDATE_SIGMA,
            "ObjectAnchor.validateSigma — map-pose σ (m) below which");
    reader.opt<bool>("ObjectAnchor.freshnessEnable", p.OBJECT_ANCHOR_FRESHNESS_ENABLE,
            "ObjectAnchor.freshnessEnable — grow R_o with obs age");
    reader.opt<float, double>("ObjectAnchor.freshnessAgeScale", p.OBJECT_ANCHOR_FRESHNESS_AGE_SCALE,
            "ObjectAnchor.freshnessAgeScale — frames→σ doubles");
    room_concept.params.object_anchor.enable      = p.OBJECT_ANCHOR_ENABLE;
    // GN-only: the autograd backends have no landmark variables, so silently honouring this under
    // LBFGS/ADAM would mean the flag reads true while nothing optimises. Refuse loudly instead.
    room_concept.params.object_anchor_optimize_landmark = p.OBJECT_ANCHOR_OPTIMIZE_LANDMARK;
    if (p.OBJECT_ANCHOR_OPTIMIZE_LANDMARK and room_concept.params.optimizer_type != "GN")
    {
        qWarning() << "[room] ObjectAnchor.optimizeLandmark needs OptimizerType = \"GN\" (have"
                   << QString::fromStdString(room_concept.params.optimizer_type) << ") — landmark "
                      "optimisation DISABLED, anchors stay pinned";
        room_concept.params.object_anchor_optimize_landmark = false;
    }
    room_concept.params.object_anchor.weight      = p.OBJECT_ANCHOR_WEIGHT;
    room_concept.params.object_anchor.huber_delta = p.OBJECT_ANCHOR_HUBER;
    room_concept.params.object_anchor_max_slots   = p.OBJECT_ANCHOR_MAX_SLOTS;
    room_concept.params.object_anchor_early_exit_sigma = p.OBJECT_ANCHOR_EARLY_EXIT_SIGMA;

    // ── RGB edge alignment ────────────────────────────────────────────────────────────────────
    reader.opt<bool>("ImageEdge.enable", p.IMAGE_EDGE_ENABLE,
            "ImageEdge.enable");
    reader.opt<bool>("ImageEdge.shadow", p.IMAGE_EDGE_SHADOW,
            "ImageEdge.shadow");
    reader.opt<bool>("ImageEdge.drive", p.IMAGE_EDGE_DRIVE,
            "ImageEdge.drive (refused unless OptimizerType == 'GN')");
    reader.opt<std::string>("ImageEdge.camera", p.IMAGE_EDGE_CAMERA,
            "ImageEdge.camera — DSR node name ('zed' | 'ricoh')");
    reader.opt<bool>("ImageEdge.useWallCorners", p.IMAGE_EDGE_USE_WALL_CORNERS,
            "ImageEdge.useWallCorners");
    reader.opt<bool>("ImageEdge.useFloorJunction", p.IMAGE_EDGE_USE_FLOOR_JUNCTION,
            "ImageEdge.useFloorJunction");
    reader.opt<bool>("ImageEdge.useWallCeiling", p.IMAGE_EDGE_USE_WALL_CEILING,
            "ImageEdge.useWallCeiling");
    reader.opt<float, double>("ImageEdge.sampleSpacingM", p.IMAGE_EDGE_SAMPLE_SPACING_M,
            "ImageEdge.sampleSpacingM — arc-length in 3-D, NOT");
    reader.opt<float, double>("ImageEdge.searchSigmas", p.IMAGE_EDGE_SEARCH_SIGMAS,
            "ImageEdge.searchSigmas — Gaussian truncation point of");
    reader.opt<int>("ImageEdge.maxSearchPx", p.IMAGE_EDGE_MAX_SEARCH_PX,
            "ImageEdge.maxSearchPx — a COMPUTE bound only; the CSV");
    reader.opt<int>("ImageEdge.maxSlots", p.IMAGE_EDGE_MAX_SLOTS,
            "ImageEdge.maxSlots — newest slot only, like corners");
    reader.opt<float, double>("ImageEdge.mountPitchSigma", p.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
            "ImageEdge.mountPitchSigma (rad, ~0.2°)");
    reader.opt<float, double>("ImageEdge.mountHeightSigma", p.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
            "ImageEdge.mountHeightSigma (m)");
    reader.opt<float, double>("ImageEdge.mountYawSigma", p.IMAGE_EDGE_MOUNT_YAW_SIGMA,
            "ImageEdge.mountYawSigma (rad)");
    reader.opt<float, double>("ImageEdge.mountVertexOffsetSigmaPx", p.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX,
            "ImageEdge.mountVertexOffsetSigmaPx");
    reader.opt<float, double>("ImageEdge.mountYawCorrection", p.IMAGE_EDGE_MOUNT_YAW_CORR,
            "ImageEdge.mountYawCorrection (rad)");
    reader.opt<bool>("ImageEdge.mountApply", p.IMAGE_EDGE_MOUNT_APPLY,
            "ImageEdge.mountApply");
    reader.opt<bool>("ImageEdge.mountPublish", p.IMAGE_EDGE_MOUNT_PUBLISH,
            "ImageEdge.mountPublish");
    reader.opt<std::string>("ImageEdge.mountDescriptionFile", p.IMAGE_EDGE_MOUNT_DESCRIPTION,
            "ImageEdge.mountDescriptionFile: the robot JSON = the mount NOMINAL");
    reader.opt<float, double>("ImageEdge.wallPositionSigma", p.IMAGE_EDGE_WALL_POS_SIGMA,
            "ImageEdge.wallPositionSigma (m)");
    reader.opt<bool>("ImageEdge.useTriplePoints", p.IMAGE_EDGE_USE_TRIPLE_POINTS,
            "ImageEdge.useTriplePoints");

    // ── Platform overlays ────────────────────────────────────────────────────────────────────────
    // Parsed here because the ConfigLoader does not outlive this call; APPLIED later, once the
    // robot's own name is known from the graph. See RoomConfig::apply_platform.
    // The names come from Platform.names so the loader never has to enumerate sections — and an
    // absent key means "no overlays", which is the ordinary single-robot case rather than an error.
    // ⚠ ConfigLoader THROWS on an empty array (memory: configloader-empty-array-throws), so omit
    //   Platform.names rather than writing [].
    {
        std::vector<std::string> names;
        try { reader.opt<std::vector<std::string>>("Platform.names", names,
                ""); }
        catch (const std::exception& e) { qWarning() << "[cfg] Platform.names ignored:" << e.what(); }
        for (const auto& n : names)
        {
            RoomConfig::PlatformOverlay ov;
            // load_optional returns void and leaves the destination ALONE when the key is absent,
            // so presence is detected with a sentinel rather than a return value. NaN is the right
            // sentinel here: no legitimate config value can collide with it, whereas 0 or -1 could.
            // Presence is honest now (Reader::maybe), so no sentinel: absent stays nullopt.
            const std::string sec = "Platform." + n;
            const auto f = [&](const char* key, std::optional<float>& dst)
            {
                if (const auto v = overlay_read_impl<float, double>(cl, sec + "." + key))
                    dst = *v;
            };
            f("mountPitchSigma",    ov.mount_pitch_sigma);
            f("mountHeightSigma",   ov.mount_height_sigma);
            f("mountYawSigma",      ov.mount_yaw_sigma);
            f("mountYawCorrection", ov.mount_yaw_correction);
            f("wallPositionSigma",  ov.wall_position_sigma);
            f("CmdNoiseRot",        ov.cmd_noise_rot);
            f("CmdNoiseTrans",      ov.cmd_noise_trans);
            f("PreintZuptDensityV",     ov.zupt_density_v);
            f("PreintZuptDensityOmega", ov.zupt_density_omega);
            f("PreintOdomSigmaOmega",   ov.odom_preint_sigma_omega);
            f("PreintOdomScaleOmega",   ov.odom_preint_scale_omega);
            f("SdfSafe",                ov.sdf_safe);
            f("SdfDanger",              ov.sdf_danger);
            // An int and a bool have no spare sentinel the way a float has NaN, so each is read
            // twice from opposite seeds: only a key that is actually present makes the two agree.
            const auto i = [&](const char* key, std::optional<int>& dst)
            {
                if (const auto v = overlay_read_impl<int>(cl, sec + "." + key))
                    dst = *v;
            };
            const auto b = [&](const char* key, std::optional<bool>& dst)
            {
                if (const auto v = overlay_read_impl<bool>(cl, sec + "." + key))
                    dst = *v;
            };
            i("Period",                     ov.period_compute);
            b("UseCommandVelocityPrior",    ov.use_command_velocity_prior);
            b("CalibPivotEnabled",          ov.calib_pivot_enabled);
            b("OdomSampleLog",              ov.odom_sample_log);
            // Drift keys — see PlatformOverlay in the header for why they are here.
            f("PredictionTrustFactor",             ov.prediction_trust_factor);
            f("RotationSdfCoupling",               ov.rotation_sdf_coupling);
            f("BoundaryHessianQualityThreshold",   ov.boundary_hessian_quality_threshold);
            f("BoundaryMuQualityThreshold",        ov.boundary_mu_quality_threshold);
            f("GnLossRelTol",                      ov.gn_loss_rel_tol);
            i("GnMaxIters",                        ov.gn_max_iters);
            i("TorchNumThreads",                   ov.torch_num_threads);
            b("AdaptiveCovEnabled",                ov.adaptive_cov_enabled);
            b("WindowStrideEnabled",               ov.window_stride_enabled);
            b("BoundaryFejSchur",                  ov.boundary_fej_schur);
            b("HierPrecBoundaryEnabled",           ov.hier_prec_boundary_enabled);
            b("CornerEarlyExitCheck",              ov.corner_early_exit_check);
            f("WIor",                              ov.w_ior);
            f("BeliefForgetTime",                  ov.belief_forget_time);
            f("ObjectAnchorMeasSigmaXY",           ov.object_anchor_meas_sigma_xy);
            f("StableSdfMseMax",                   ov.stable_sdf_mse_max);
            if (auto cam = overlay_read_impl<std::string>(cl, sec + ".camera");
                cam.has_value() and not cam->empty())
                ov.image_edge_camera = *cam;
            p.platform_overlays[n] = ov;
        }
        if (not names.empty())
            qInfo() << "[cfg] platform overlays parsed for" << static_cast<int>(names.size())
                    << "robot(s); the matching one is applied once the graph names this robot";
    }
    {
        std::vector<std::string> names;
        try { reader.opt<std::vector<std::string>>("Scenario.names", names,
                ""); }
        catch (const std::exception& e) { qWarning() << "[cfg] Scenario.names ignored:" << e.what(); }
        for (const auto& n : names)
        {
            RoomConfig::ScenarioOverlay ov;
            // Presence is honest now (Reader::maybe): no NaN sentinel, no read-it-twice-from-opposite-
            // seeds trick for the bool.
            const std::string sec = "Scenario." + n;
            if (auto svg = overlay_read_impl<std::string>(cl, sec + ".RoomLayoutSvg");
                svg.has_value() and not svg->empty())
                ov.room_layout_svg = *svg;
            if (const auto h = overlay_read_impl<float, double>(cl, sec + ".RoomHeight"))
                ov.room_height = *h;
            if (const auto b = overlay_read_impl<bool>(cl, sec + ".RecenterRoomPolygon"))
                ov.recenter_room_polygon = *b;
            const auto sf = [&](const char* key, std::optional<float>& dst)
            {
                if (const auto v = overlay_read_impl<float, double>(cl, sec + "." + key))
                    dst = *v;
            };
            sf("LidarHighMaxHeight", ov.lidar_high_max_height);
            sf("TargetWallMargin",   ov.target_wall_margin);
            {   // ⚠ ConfigLoader throws on an EMPTY array, so absent is how to say "leave it".
                if (auto subs = overlay_read_impl<std::vector<std::string>>(cl, sec + ".ObjectAnchorSubtypes");
                    subs.has_value() and not subs->empty())
                    ov.object_anchor_subtypes = std::move(*subs);
            }
            p.scenario_overlays[n] = ov;
        }
        if (not names.empty())
            qInfo() << "[cfg] scenario overlays parsed for" << static_cast<int>(names.size())
                    << "scenario(s); the matching one is applied once the graph names this place";
    }
    // ConfigLoader throws on an EMPTY array (memory: configloader-empty-array-throws), so an absent
    // key is the way to say "no extra calibration cameras", not `calibCameras = []`.
    try { reader.opt<std::vector<std::string>>("ImageEdge.calibCameras", p.CALIB_CAMERAS,
            "only reading here that needs no ground truth. See camera_calibration.h"); }
    catch (const std::exception& e) { qWarning() << "[cfg] ImageEdge.calibCameras ignored:" << e.what(); }
    reader.opt<std::string>("ImageEdge.csv", p.IMAGE_EDGE_CSV,
            "ImageEdge.csv");
    reader.opt<int>("ImageEdge.minConvertIntervalMs", p.IMAGE_EDGE_MIN_CONVERT_MS,
            "ImageEdge.minConvertIntervalMs");
    reader.opt<int>("ImageEdge.calibMinConvertIntervalMs", p.IMAGE_EDGE_CALIB_MIN_CONVERT_MS,
            "ImageEdge.calibMinConvertIntervalMs");

    room_concept.params.image_edge.enable             = p.IMAGE_EDGE_ENABLE;
    room_concept.params.image_edge.drive              = p.IMAGE_EDGE_DRIVE;
    room_concept.params.image_edge.search_sigmas      = p.IMAGE_EDGE_SEARCH_SIGMAS;
    room_concept.params.image_edge.mount_pitch_sigma  = p.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    room_concept.params.image_edge.mount_height_sigma = p.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    room_concept.params.image_edge.mount_yaw_sigma    = p.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    room_concept.params.image_edge.wall_position_sigma = p.IMAGE_EDGE_WALL_POS_SIGMA;
    room_concept.params.image_edge.use_triple_points  = p.IMAGE_EDGE_USE_TRIPLE_POINTS;
    room_concept.params.image_edge_max_slots          = p.IMAGE_EDGE_MAX_SLOTS;
    room_concept.params.image_edge_shadow             = p.IMAGE_EDGE_SHADOW;
    room_concept.params.image_edge_csv                = p.IMAGE_EDGE_CSV;
    room_concept.params.calib_state_file              = p.CALIB_STATE_FILE;
    // GN-only, and refused LOUDLY rather than honoured silently: the autograd backends evaluate the
    // term (the torch mirror exists) but only the GN factor list can be driven by it, so under
    // LBFGS/ADAM the flag would read true while nothing moved. Same rule, same reason, as
    // ObjectAnchor.optimizeLandmark above.
    if (p.IMAGE_EDGE_DRIVE and room_concept.params.optimizer_type != "GN")
    {
        qWarning() << "[room] ImageEdge.drive needs OptimizerType = \"GN\" (have"
                   << QString::fromStdString(room_concept.params.optimizer_type)
                   << ") — RGB edge term DEMOTED to shadow (evaluated + logged, does not move the pose)";
        room_concept.params.image_edge.drive  = false;
        room_concept.params.image_edge_shadow = true;
    }
    if (p.IMAGE_EDGE_DRIVE and not p.IMAGE_EDGE_ENABLE)
        qWarning() << "[room] ImageEdge.drive = true but ImageEdge.enable = false — nothing is built, "
                      "the term is INERT. Set enable = true.";
    reader.opt<bool>("RoomConcept.FarPointsWeight", room_concept.params.far_points_weight,
            "enable distance-proportional weighting");
    reader.opt<float, double>("RoomConcept.FarPointsExponent", room_concept.params.far_points_exponent,
            "α — exponent of the distance weight");
    reader.opt<float, double>("RoomConcept.FarPointsMinWeight", room_concept.params.far_points_min_weight,
            "floor weight to avoid silencing near points");
    reader.opt<bool>("RoomConcept.IncidenceAngleWeight", room_concept.params.incidence_angle_weight,
            "===== Incidence-angle weighting ===== Grazing hits are more ambiguous against the nearest-face model, so reduce their contribution using the cosine…");
    reader.opt<float, double>("RoomConcept.IncidenceAngleExponent", room_concept.params.incidence_angle_exponent,
            "β — exponent of the incidence weight");
    reader.opt<float, double>("RoomConcept.IncidenceAngleMinWeight", room_concept.params.incidence_angle_min_weight,
            "floor weight for grazing rays");
    reader.opt<bool>("RoomConcept.UseCuda", room_concept.params.use_cuda,
            "GPU/CPU selection Note: For small tensors (~200 points), CPU is faster due to GPU transfer overhead");
    reader.opt<bool>("RoomConcept.DebugLog", room_concept.params.debug_log_enabled,
            "Write per-frame CSV to tmp/sdf_localizer/log.csv");
    reader.opt<bool>("RoomConcept.OptimizerTimingCsv", room_concept.params.optimizer_timing_csv,
            "lean per-update timing CSV (etc/optimizer_timing.csv)");

    reader.opt<bool>("RoomConcept.RerunEnabled", room_concept.params.rerun_enabled,
            "===== Rerun streaming =====");
    reader.opt<std::string>("RoomConcept.RerunHost", room_concept.params.rerun_host,
            "");
    reader.opt<int>("RoomConcept.RerunPort", room_concept.params.rerun_port,
            "");
    reader.opt<int>("RoomConcept.RerunSdfEveryN", room_concept.params.rerun_sdf_every_n,
            "");
    reader.opt<int>("RoomConcept.RerunSdfResolution", room_concept.params.rerun_sdf_resolution,
            "");
    reader.opt<int>("RoomConcept.RerunMaxQueue", room_concept.params.rerun_max_queue,
            "");

    // ── DSR stabilization thresholds ──────────────────────────────────────
    reader.opt<int>("DSR.StableFramesRequired", p.STABLE_FRAMES_REQUIRED,
            "DSR stabilization: this many consecutive 'stable' frames before creating the room node and re-parenting the robot under it");
    reader.opt<float, double>("DSR.StableSdfMseMax", p.STABLE_SDF_MSE_MAX,
            "unified 2026-08-29; P3Bot ran the 0.06 default only because its file was silent");
    reader.opt<float, double>("DSR.StableCovTtMax", p.STABLE_COV_TT_MAX,
            "");
    reader.opt<bool>("DSR.BootstrapTableEnabled", p.BOOTSTRAP_TABLE_ENABLED,
            "Debug bootstrap table hanging from the room node");
    reader.opt<float, double>("DSR.BootstrapTableX", p.BOOTSTRAP_TABLE_X,
            "");
    reader.opt<float, double>("DSR.BootstrapTableY", p.BOOTSTRAP_TABLE_Y,
            "");
    reader.opt<float, double>("DSR.BootstrapTableYaw", p.BOOTSTRAP_TABLE_YAW,
            "");
    reader.opt<float, double>("DSR.BootstrapTableWidth", p.BOOTSTRAP_TABLE_WIDTH,
            "");
    reader.opt<float, double>("DSR.BootstrapTableDepth", p.BOOTSTRAP_TABLE_DEPTH,
            "");
    reader.opt<float, double>("DSR.BootstrapTableHeight", p.BOOTSTRAP_TABLE_HEIGHT,
            "");

    // ── EpistemicController params ─────────────────────────────────────────
    auto& ec = epistemic.params;
    auto& ep = epistemic.epistemic_planner().params;
    reader.opt<bool>("EpistemicController.PublishAffordance", p.PUBLISH_AFFORDANCE,
            "EpistemicController.PublishAffordance — publish the room");
    reader.opt<bool>("RoomConcept.CalibPivotEnabled", p.CALIB_PIVOT_ENABLED,
            "RoomConcept.CalibPivotEnabled");
    reader.opt<float, double>("RoomConcept.CalibForcedGainNats", p.CALIB_FORCED_GAIN_NATS,
            "RoomConcept.CalibForcedGainNats");
    reader.opt<float, double>("EpistemicController.ExecStallTimeout", p.EXEC_STALL_TIMEOUT_S,
            "EpistemicController.ExecStallTimeout (s)");
    reader.opt<float, double>("EpistemicController.ExecStallProgress", p.EXEC_STALL_PROGRESS_M,
            "EpistemicController.ExecStallProgress (m)");
    reader.opt<int>("EpistemicController.NumArcCurvatures", ec.num_arc_curvatures,
            "discrete curvatures (odd ⟹ includes κ=0)");
    reader.opt<int>("EpistemicController.HorizonSteps", ec.horizon_steps,
            "---- Policy rollout & EFE ----");
    reader.opt<float, double>("EpistemicController.Dt", ec.dt,
            "Time delta");
    reader.opt<float, double>("EpistemicController.MaxAdvSpeed", ec.max_adv_speed,
            "m/s");
    reader.opt<float, double>("EpistemicController.MaxRotSpeed", ec.max_rot_speed,
            "rad/s");
    reader.opt<float, double>("EpistemicController.WEpistemic", ec.w_epistemic,
            "EFE weight: information gain");
    reader.opt<float, double>("EpistemicController.WPragmatic", ec.w_pragmatic,
            "EFE weight: goal proximity");
    reader.opt<float, double>("EpistemicController.WHeading", ec.w_heading,
            "EFE weight: heading alignment");
    reader.opt<float, double>("EpistemicController.WBoundary", ec.w_boundary,
            "EFE weight: room boundary avoidance");
    reader.opt<float, double>("EpistemicController.KRot", ec.k_rot,
            "---- Simple angle-distance controller gains ----");
    reader.opt<float, double>("EpistemicController.GaussianSigma", ec.gaussian_sigma,
            "goal proximity Gaussian σ (m)");
    reader.opt<float, double>("EpistemicController.SpeedHorizonS", ec.speed_horizon_s,
            "max time-to-target used for speed ramp (decoupled from rollout horizon)");
    reader.opt<float, double>("EpistemicController.ObstacleRadius", ec.obstacle_radius,
            "---- Obstacle avoidance (lidar floor projection) ----");
    reader.opt<float, double>("EpistemicController.ObstacleK", ec.obstacle_k,
            "obstacle repulsion gain");
    reader.opt<float, double>("EpistemicController.ObstacleStepCap", ec.obstacle_step_cap,
            "max repulsion step size");
    reader.opt<float, double>("EpistemicController.WObstacle", ec.w_obstacle,
            "EFE weight: obstacle penalty");
    reader.opt<float, double>("EpistemicController.WallFilterMargin", ec.wall_filter_margin,
            "lidar points within this margin of wall are ignored (m)");
    reader.opt<float, double>("EpistemicController.BandwidthCoupling", ec.bandwidth_coupling,
            "---- Perceptual bandwidth speed limit ---- Couples rotation and translation so that sharp turns reduce the allowed translation speed");
    reader.opt<float, double>("EpistemicController.SdfSafe", ec.sdf_safe,
            "below this SDF-MSE → full speed");
    reader.opt<float, double>("EpistemicController.SdfDanger", ec.sdf_danger,
            "at/above this → minimum speed");
    reader.opt<float, double>("EpistemicController.GovernorAlphaMin", ec.governor_alpha_min,
            "minimum speed fraction");
    reader.opt<float, double>("EpistemicController.FimCornerSigma", ec.fim_corner_sigma,
            "---- FIM scoring at final state (epistemic EFE term) ----");
    reader.opt<float, double>("EpistemicController.FimMaxRange", ec.fim_max_range,
            "max range for corner/wall visibility (m)");
    reader.opt<float, double>("EpistemicController.GridResolution", ep.grid_resolution,
            "spacing between candidate targets (m)");
    reader.opt<float, double>("EpistemicController.MinDistance", ep.min_distance,
            "ignore candidates closer than this to robot (m)");
    reader.opt<int>("EpistemicController.MaxCandidates", ep.max_candidates,
            "cap on number of evaluated candidates");
    reader.opt<float, double>("EpistemicController.TargetWallMargin", ep.target_wall_margin,
            "reject targets closer than this to walls (m)");
    reader.opt<float, double>("EpistemicController.TargetObstacleClearance", ep.target_obstacle_clearance,
            "... and to object/obstacle footprints (m)");
    reader.opt<float, double>("EpistemicController.AngularDominanceRatio", ep.angular_dominance_ratio,
            "σ²_θ / max(σ²_x, σ²_y) threshold");
    reader.opt<bool>("EpistemicController.ModeDisambiguation", ep.mode_disambiguation,
            "MODE DISAMBIGUATION (2026-09-16)");
    // WExploration (a far-is-better distance BONUS) is gone — the sign was wrong and produced a
    // corner-to-corner oscillation. Travel distance is a cost; see Params::w_travel_cost.
    reader.opt<float, double>("EpistemicController.WTravelCost", ep.w_travel_cost,
            "nats per unit (distance / room_diagonal)");
    reader.opt<float, double>("EpistemicController.IorCellSize", ep.ior_cell_size,
            "spatial resolution of the visit grid (m)");
    reader.opt<float, double>("EpistemicController.IorDecayTime", ep.ior_decay_time,
            "seconds until a visited cell is fully 'stale' again");
    reader.opt<float, double>("EpistemicController.WIor", ep.w_ior,
            "exponent for IoR suppressor: score *= staleness^w_ior");
    reader.opt<float, double>("EpistemicController.WIorDrive", ep.w_ior_drive,
            "path_interest = mean staleness of intermediate path cells ∈ [0,1] higher → prefer routes through unexplored territory ---- Inhibition-of-Return…");
    reader.opt<float, double>("EpistemicController.WPathInterest", ep.w_path_interest,
            "weight: bonus for paths that traverse unvisited cells");
    reader.opt<float, double>("EpistemicController.IorPathRadius", ep.ior_path_radius,
            "receptive-field radius for continuous path marking (m)");
    // PatrolEnabled / PatrolGainFloor / InfoExhaustedGain were the second-level "patrol mode"
    // switch. Removed: the epistemic term is now MARGINAL (it extinguishes itself when there is
    // nothing left to see) and the IoR drive is unbounded in neglect age (it never flatlines), so
    // the hand-off happens continuously and there is nothing left to trigger or floor.
    reader.opt<float, double>("EpistemicController.FimCornerSigma", ep.fim_corner_sigma,
            "---- FIM scoring at final state (epistemic EFE term) ----");
    reader.opt<float, double>("EpistemicController.FimMaxRange", ep.fim_max_range,
            "max range for corner/wall visibility (m)");
    reader.opt<float, double>("EpistemicController.FimPriorPrecisionFloor", ep.fim_prior_precision_floor,
            "floor on Y_prior eigenvalues (caps assumed max");
    reader.opt<bool>("EpistemicController.FimUseWalls", ep.fim_use_walls,
            "covariance at 1/floor) → degenerate prior gives a large-but-FINITE ΔH, no ∞/NaN ---- Wall-surface SDF observations (match the localizer's likelihood)…");
    reader.opt<float, double>("EpistemicController.FimWallSigma", ep.fim_wall_sigma,
            "SDF obs noise σ (m); ≈ RoomConcept.sigma_sdf");
    reader.opt<int>("EpistemicController.FimWallRays", ep.fim_wall_rays,
            "simulated 360° lidar rays for wall coverage");
    reader.opt<bool>("EpistemicController.FimWallIncidenceWeight", ep.fim_wall_incidence_weight,
            "down-weight grazing hits (|n·ray|)");
    reader.opt<float, double>("EpistemicController.FimWallIncidenceMin", ep.fim_wall_incidence_min,
            "floor for the incidence weight");
    reader.opt<float, double>("EpistemicController.ArrivalDistance", ep.arrival_distance,
            "target reached threshold (m)");
    reader.opt<float, double>("EpistemicController.DwellTime", ep.dwell_time,
            "seconds to stop after reaching a target");
    reader.opt<float, double>("EpistemicController.BeliefForgetTime", ep.belief_forget_time,
            "seconds; 0 disables (current self-extinguishing behaviour)");
    epistemic.set_robot_footprint(p.ROBOT_WIDTH, p.ROBOT_LENGTH);
}


std::vector<std::string> RoomConfig::apply_platform(const std::string& robot)
{
    std::vector<std::string> changed;
    const auto it = platform_overlays.find(robot);
    if (it == platform_overlays.end()) return changed;   // no section for this robot: keep the shared defaults
    const auto& ov = it->second;
    const std::string source = "Platform." + robot;
    const auto set = [&](const std::optional<float>& src, float& dst, const char* name)
    { if (src.has_value() and *src != dst) { changed.emplace_back(name); dst = *src; note_overlay_applied(name, dst, source); } };
    set(ov.mount_pitch_sigma,    IMAGE_EDGE_MOUNT_PITCH_SIGMA,  "mountPitchSigma");
    set(ov.mount_height_sigma,   IMAGE_EDGE_MOUNT_HEIGHT_SIGMA, "mountHeightSigma");
    set(ov.mount_yaw_sigma,      IMAGE_EDGE_MOUNT_YAW_SIGMA,    "mountYawSigma");
    set(ov.mount_yaw_correction, IMAGE_EDGE_MOUNT_YAW_CORR,     "mountYawCorrection");
    set(ov.wall_position_sigma,  IMAGE_EDGE_WALL_POS_SIGMA,     "wallPositionSigma");
    if (ov.image_edge_camera.has_value() and *ov.image_edge_camera != IMAGE_EDGE_CAMERA)
    { changed.emplace_back("camera"); IMAGE_EDGE_CAMERA = *ov.image_edge_camera; note_overlay_applied("camera", IMAGE_EDGE_CAMERA, source); }
    if (ov.calib_pivot_enabled.has_value() and *ov.calib_pivot_enabled != CALIB_PIVOT_ENABLED)
    { changed.emplace_back("CalibPivotEnabled"); CALIB_PIVOT_ENABLED = *ov.calib_pivot_enabled; note_overlay_applied("CalibPivotEnabled", CALIB_PIVOT_ENABLED, source); }
    if (ov.odom_sample_log.has_value() and *ov.odom_sample_log != ODOM_SAMPLE_LOG)
    { changed.emplace_back("OdomSampleLog"); ODOM_SAMPLE_LOG = *ov.odom_sample_log; note_overlay_applied("OdomSampleLog", ODOM_SAMPLE_LOG, source); }
    if (ov.object_anchor_meas_sigma_xy.has_value() and *ov.object_anchor_meas_sigma_xy != OBJECT_ANCHOR_MEAS_SIG_XY)
    { changed.emplace_back("ObjectAnchorMeasSigmaXY"); OBJECT_ANCHOR_MEAS_SIG_XY = *ov.object_anchor_meas_sigma_xy; note_overlay_applied("ObjectAnchorMeasSigmaXY", OBJECT_ANCHOR_MEAS_SIG_XY, source); }
    if (ov.stable_sdf_mse_max.has_value() and *ov.stable_sdf_mse_max != STABLE_SDF_MSE_MAX)
    { changed.emplace_back("StableSdfMseMax"); STABLE_SDF_MSE_MAX = *ov.stable_sdf_mse_max; note_overlay_applied("StableSdfMseMax", STABLE_SDF_MSE_MAX, source); }
    return changed;
}

// The overlay values whose destinations live in RoomConcept / EpistemicController rather than here.
// ⚠ CmdNoiseRot/Trans were PARSED by apply_platform and applied by nothing — harmless while each
//   robot had its own file (the top-level key did the work) and a silent regression the moment the
//   files merged. They are applied here.
std::vector<std::string> RoomConfig::apply_platform_to(const std::string& robot,
                                                       rc::RoomConcept& room_concept,
                                                       rc::EpistemicController& epistemic)
{
    std::vector<std::string> changed;
    const auto it = platform_overlays.find(robot);
    if (it == platform_overlays.end()) return changed;
    const auto& ov = it->second;
    const std::string source = "Platform." + robot;
    const auto set = [&](const auto& src, auto& dst, const char* name)
    { if (src.has_value() and *src != dst) { changed.emplace_back(name); dst = *src; note_overlay_applied(name, dst, source); } };
    auto& rp = room_concept.params;
    set(ov.cmd_noise_rot,           rp.cmd_noise_rot,                     "CmdNoiseRot");
    set(ov.cmd_noise_trans,         rp.cmd_noise_trans,                   "CmdNoiseTrans");
    set(ov.use_command_velocity_prior, rp.use_command_velocity_prior,     "UseCommandVelocityPrior");
    set(ov.zupt_density_v,          rp.odom_preint_noise.zupt_density_v,     "PreintZuptDensityV");
    set(ov.zupt_density_omega,      rp.odom_preint_noise.zupt_density_omega, "PreintZuptDensityOmega");
    set(ov.odom_preint_sigma_omega, rp.odom_preint_noise.sigma_omega,        "PreintOdomSigmaOmega");
    set(ov.odom_preint_scale_omega, rp.odom_preint_noise.scale_omega,        "PreintOdomScaleOmega");
    set(ov.sdf_safe,                epistemic.params.sdf_safe,            "SdfSafe");
    set(ov.sdf_danger,              epistemic.params.sdf_danger,          "SdfDanger");
    // Drift keys — same mechanism, different reason. See PlatformOverlay in the header.
    set(ov.prediction_trust_factor,         rp.prediction_trust_factor,               "PredictionTrustFactor");
    set(ov.rotation_sdf_coupling,           rp.rotation_sdf_coupling,                 "RotationSdfCoupling");
    set(ov.boundary_hessian_quality_threshold, rp.boundary_hessian_quality_threshold,    "BoundaryHessianQualityThreshold");
    set(ov.boundary_mu_quality_threshold,   rp.boundary_mu_quality_threshold,         "BoundaryMuQualityThreshold");
    set(ov.gn_loss_rel_tol,                 rp.gn_loss_rel_tol,                       "GnLossRelTol");
    set(ov.gn_max_iters,                    rp.gn_max_iters,                          "GnMaxIters");
    set(ov.torch_num_threads,               rp.torch_num_threads,                     "TorchNumThreads");
    set(ov.adaptive_cov_enabled,            rp.adaptive_cov_enabled,                  "AdaptiveCovEnabled");
    set(ov.window_stride_enabled,           rp.window_stride_enabled,                 "WindowStrideEnabled");
    set(ov.boundary_fej_schur,              rp.boundary_fej_schur,                    "BoundaryFejSchur");
    set(ov.hier_prec_boundary_enabled,      rp.hier_prec_boundary_enabled,            "HierPrecBoundaryEnabled");
    set(ov.corner_early_exit_check,         rp.corner_early_exit_check,               "CornerEarlyExitCheck");
    set(ov.w_ior, epistemic.epistemic_planner().params.w_ior, "WIor");
    set(ov.belief_forget_time, epistemic.epistemic_planner().params.belief_forget_time, "BeliefForgetTime");
    return changed;
}

std::vector<std::string> RoomConfig::apply_scenario_to(const std::string& scenario,
                                                       rc::EpistemicController& epistemic)
{
    std::vector<std::string> changed;
    const auto it = scenario_overlays.find(scenario);
    if (it == scenario_overlays.end()) return changed;
    const auto& ov = it->second;
    const std::string source = "Scenario." + scenario;
    if (ov.target_wall_margin.has_value() and
        *ov.target_wall_margin != epistemic.epistemic_planner().params.target_wall_margin)
    { changed.emplace_back("TargetWallMargin");
      epistemic.epistemic_planner().params.target_wall_margin = *ov.target_wall_margin; note_overlay_applied("TargetWallMargin", epistemic.epistemic_planner().params.target_wall_margin, source); }
    return changed;
}

std::vector<std::string> RoomConfig::apply_scenario(const std::string& scenario)
{
    std::vector<std::string> changed;
    const auto it = scenario_overlays.find(scenario);
    if (it == scenario_overlays.end()) return changed;
    const auto& ov = it->second;
    const std::string source = "Scenario." + scenario;
    if (ov.room_layout_svg.has_value() and *ov.room_layout_svg != ROOM_LAYOUT_SVG)
    { changed.emplace_back("RoomLayoutSvg"); ROOM_LAYOUT_SVG = *ov.room_layout_svg; note_overlay_applied("RoomLayoutSvg", ROOM_LAYOUT_SVG, source); }
    if (ov.room_height.has_value() and *ov.room_height != room_height)
    { changed.emplace_back("RoomHeight"); room_height = *ov.room_height; note_overlay_applied("RoomHeight", room_height, source); }
    if (ov.recenter_room_polygon.has_value() and *ov.recenter_room_polygon != RECENTER_ROOM_POLYGON)
    { changed.emplace_back("RecenterRoomPolygon"); RECENTER_ROOM_POLYGON = *ov.recenter_room_polygon; note_overlay_applied("RecenterRoomPolygon", RECENTER_ROOM_POLYGON, source); }
    if (ov.lidar_high_max_height.has_value() and *ov.lidar_high_max_height != LIDAR_HIGH_MAX_HEIGHT)
    { changed.emplace_back("LidarHighMaxHeight"); LIDAR_HIGH_MAX_HEIGHT = *ov.lidar_high_max_height; note_overlay_applied("LidarHighMaxHeight", LIDAR_HIGH_MAX_HEIGHT, source); }
    if (ov.object_anchor_subtypes.has_value() and *ov.object_anchor_subtypes != OBJECT_ANCHOR_SUBTYPES)
    { changed.emplace_back("ObjectAnchorSubtypes"); OBJECT_ANCHOR_SUBTYPES = *ov.object_anchor_subtypes; note_overlay_applied("ObjectAnchorSubtypes", OBJECT_ANCHOR_SUBTYPES, source); }

    // ── A WALL BAND MAY NOT REACH THE CEILING ───────────────────────────────────────────────────
    // The high band feeds a 2-D WALL polygon SDF. A ceiling return sits at an arbitrary INTERIOR
    // xy, so its distance to the nearest wall is large by construction: let the ceiling plane into
    // the band and the localiser is fitting a wall model to points that are not on a wall.
    // ★ MEASURED 2026-08-29: with LidarHighMaxHeight raised to 3.0 in a room whose RoomHeight is
    //   3.0, the startup check found a 51450-point z-peak at 3.01 m and the SDF residual sat at
    //   0.164 m RMS after optimisation, gate pinned, 0% early exit — while the pose stayed
    //   confident and corner association healthy, because the pose was not the thing that was wrong.
    // ★ NOT a tuned threshold: it is the same LIDAR_CEILING_MARGIN the startup ceiling check already
    //   applies, enforced against the ceiling the SCENARIO states rather than only against the one
    //   the check manages to detect. The check can be fooled — it was, here — and a stated ceiling
    //   cannot be. Announced, never silent: a value that is quietly reduced is a value nobody
    //   revisits.
    if (const float cap = room_height - LIDAR_CEILING_MARGIN;
        room_height > 0.f and LIDAR_HIGH_MAX_HEIGHT > cap)
    {
        qWarning() << "[cfg] LidarHighMaxHeight" << LIDAR_HIGH_MAX_HEIGHT
                   << "m reaches the stated" << room_height << "m ceiling; the high band feeds a 2-D"
                   << "WALL SDF and a ceiling return has no wall to be near. Clamped to" << cap
                   << "m (ceiling -" << LIDAR_CEILING_MARGIN << "m).";
        LIDAR_HIGH_MAX_HEIGHT = cap;
        changed.emplace_back("LidarHighMaxHeight(clamped below the ceiling)");
    }
    return changed;
}

}  // namespace rc
