/*
 * bottle_existence.cpp — see bottle_existence.h. Evidence-based bottle removal (existence log-odds + debounce).
 */

#include "bottle_existence.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <locale>
#include <print>
#include <span>
#include <vector>

#include "bottle_fitter.h"                                     // rc::BottleFitter (instances)
#include "../../common/existence_belief/existence_belief.h"    // rc::exist:: carve_box / ExistenceBelief
#include "../../common/nbv/viewpoint_score.h"                  // rc::nbv:: Target / fill / visible_fraction
#include "../../common/nbv/graph_obstacles.h"                  // rc::nbv:: sensor_from_graph / obstacles

namespace rc {

namespace {

// The bottle as the NBV sees it: an upright cylinder, so the footprint is a square of side 2r and there is no
// yaw. Same Target the epistemic planner scores candidate viewpoints with — which is the whole point: absence
// must be weighted by the model the planner maximises, or the two disagree about what a good look even is.
rc::nbv::Target target_of(const BottleInstance& inst)
{
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    rc::nbv::Target t;
    t.cx = s.cx; t.cy = s.cy; t.yaw = 0.0f;
    t.w  = 2.0f * s.radius; t.h = 2.0f * s.radius;
    t.z0 = s.cz - 0.5f * s.height;
    t.z1 = s.cz + 0.5f * s.height;
    t.sigma_pos_m    = std::sqrt(std::max(0.0f, 0.5f * (S(0, 0) + S(1, 1))));
    t.sigma_extent_m = std::sqrt(std::max(0.0f, S(3, 3)));      // radius σ
    return t;
}

// ★A BOTTLE STANDS INSIDE ITS TABLE'S FOOTPRINT, and rc::nbv::Obstacle is a 2-D rectangle with no height.
// So the support table — and the bottle's own node, which collect_graph_obstacles also returns — sit
// squarely on every sightline to the bottle, and visible_fraction would report it permanently occluded:
// vis → 0 ⇒ p_detect → 0 ⇒ absence never charged ⇒ every bottle immortal. The channel would have looked
// like it was working (L simply never moves), which is the failure mode hardest to notice from outside.
//
// The physical statement is "you cannot be occluded by a footprint you are STANDING INSIDE", and it needs
// no ids, no support-parent bookkeeping and no new parameter: drop the obstacles that contain the object's
// own centre. Everything else — another bottle, a chair between us — is a genuine occluder and stays.
//
// ⚠What this does NOT fix: a DIFFERENT table between camera and bottle still blocks the ray, even though a
// bottle at 0.85 m clears a 0.75 m tabletop. That is the shared obstacle model being 2-D; it errs toward
// "occluded" ⇒ toward HOLDING, which is the safe direction. Noted here rather than worked around.
bool footprint_contains(const rc::nbv::Obstacle& o, float x, float y)
{
    const float dx = x - o.cx, dy = y - o.cy;
    const float c = std::cos(o.yaw), s = std::sin(o.yaw);
    return std::abs(c * dx + s * dy) <= 0.5f * o.w and std::abs(-s * dx + c * dy) <= 0.5f * o.h;
}

}  // namespace

void BottleExistence::update_and_remove(BottleFitter& fitter, const Inputs& in,
                                        const std::function<void(std::uint64_t, BottleInstance&)>& on_remove)
{
    if (not cfg_.existence_enabled or in.G == nullptr)
        return;
    if (not in.fresh_masks and in.sweep == nullptr)
        return;                       // no new evidence on either channel this cycle ⇒ nothing to integrate

    // ── the camera, once per cycle (all instances share it) ───────────────────────────────────────
    // Pose AND heading: the fill model needs to know where the camera is, the visibility model which way it
    // points. zed's frame is x-right, y-DEPTH, z-up (see ROBOT_GEOMETRY.md), so the optical axis is column 1.
    bool  cam_ok  = false;
    Eigen::Vector2f cam_xy = Eigen::Vector2f::Zero();
    float cam_yaw = 0.0f;
    if (in.inner_eigen != nullptr)
        // ★PINNED to the mask's capture stamp (see Inputs::masks_stamp_ms). Reading ts=0 judges a frame by
        // a pose the robot may have left: while moving, the fill and the visibility would be computed for a
        // viewpoint that never took the picture, and absence would be charged for a look never made.
        if (const auto T = in.inner_eigen->get_transformation_matrix("room", "zed", in.masks_stamp_ms);
            T.has_value())
        {
            const auto& M = T.value();
            cam_xy  = Eigen::Vector2f(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)));
            cam_yaw = std::atan2(static_cast<float>(M(1, 1)), static_cast<float>(M(0, 1)));
            cam_ok  = true;
        }

    // ★An INCOMPLETE camera model must REFUSE, not guess (rc::nbv::Sensor::complete): without the vertical
    // field the fill collapses to horizontal-only, and for an object as tall-and-thin as a bottle that is the
    // difference between "unresolvable sliver" and "well framed". Guessing here would charge absence the
    // geometry never earned, so a cycle with no intrinsics simply carries no camera evidence.
    const rc::nbv::Sensor sensor = rc::nbv::sensor_from_graph(
        *in.G, in.inner_eigen,
        rc::detect::DetectorEnvelope{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft});
    const bool camera_usable = cam_ok and in.fresh_masks and sensor.complete();

    // Occluders: the furniture the graph knows about, plus the WALLS — a bottle the robot cannot see through a
    // wall must never be charged absence for it. Both come from the same collectors the planner uses.
    std::vector<rc::nbv::Obstacle> obstacles;
    if (camera_usable)
    {
        obstacles = rc::nbv::collect_graph_obstacles(*in.G, in.inner_eigen, 0);
        if (in.room_polygon != nullptr and in.room_polygon->size() >= 3)
        {
            const auto walls = rc::nbv::wall_obstacles(std::span<const Eigen::Vector2f>(*in.room_polygon), 0.10f);
            obstacles.insert(obstacles.end(), walls.begin(), walls.end());
        }
    }

    // The two sensor rates ARE the two log-likelihood ratios — no separate "confirm gain" / "absence gain"
    // knobs to tune, and no way for the two directions to drift apart into a ratchet:
    //     detected   ⇒ log(  P(detect | exists)   / P(detect | ¬exists)   ) = log(pd/pc)          > 0
    //     not        ⇒ log( P(¬detect | exists)   / P(¬detect | ¬exists)  ) = log((1−pd)/(1−pc))  < 0
    rc::exist::SensorModel sm;
    sm.sensor_sigma_m = cfg_.existence_sensor_sigma_m;
    sm.detection_prob = cfg_.existence_detection_prob;
    sm.clutter_prob   = cfg_.existence_clutter_prob;
    const float pd = std::clamp(sm.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(sm.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_detect = std::log(pd / pc);
    const float llr_absent = std::log((1.0f - pd) / (1.0f - pc));

    // One tick per CALL, not per instance: incrementing inside the loop would make N bottles log on
    // ALTERNATE cycles instead of together, so no two rows could be compared at the same instant —
    // which is precisely what you need when asking why one bottle held and its neighbour did not.
    static int ex_cycle = 0;
    const bool log_now = (++ex_cycle % 60 == 0);

    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter.instances())
    {
        if (not inst.ai2_initialized)
            continue;                 // no belief yet ⇒ no geometry to project, nothing to carve against
        if (not inst.existence_seeded)
        {
            inst.existence.set_max(cfg_.existence_logodds_max);
            inst.existence.set(cfg_.existence_birth_logodds);
            inst.existence_seeded = true;
        }
        inst.existence.set_max(cfg_.existence_logodds_max);
        inst.existence.set_frame_correlation(cfg_.existence_frame_correlation);

        const auto  t  = target_of(inst);
        float fill_min_dbg = std::numeric_limits<float>::quiet_NaN();
        int   dropped_dbg  = 0;   // occluders discarded because the bottle STANDS INSIDE them
        // How resolving THIS cycle's look was, in units of one ideal observation. Drives the debounce below.
        float cycle_p_detect = 0.0f;

        // ── CAMERA channel (mask clock) ───────────────────────────────────────────────────────────
        if (camera_usable)
        {
            // Occluders MINUS the ones this bottle stands inside (its support table, and its own node).
            std::vector<rc::nbv::Obstacle> occluders;
            occluders.reserve(obstacles.size());
            for (const auto& o : obstacles)
                if (not footprint_contains(o, t.cx, t.cy))
                    occluders.push_back(o);
            dropped_dbg = static_cast<int>(obstacles.size() - occluders.size());

            const auto [fill_max, fill_min] = rc::nbv::predicted_fill_axes(t, cam_xy, sensor, cam_yaw);
            const float vis   = rc::nbv::visible_fraction(t, cam_xy, cam_yaw, sensor, occluders);
            const float f_sig = rc::nbv::predicted_fill_sigma(t, cam_xy, sensor);
            // Marginalised over the belief's own σ: an uncertain bottle is one we are correspondingly less
            // sure we should have resolved, so its absence is charged less. That is the σ→precision route,
            // not a gate — nothing here can hard-refuse a look.
            const float p_det = rc::nbv::expected_p_detect(fill_max, fill_min, f_sig, vis, sensor.env);

            inst.dbg_ex_p_detect = p_det;
            fill_min_dbg = fill_min;
            inst.dbg_ex_fill     = fill_max;
            inst.dbg_ex_vis      = vis;
            cycle_p_detect       = p_det;

            const bool detected = (inst.frames_since_detection == 0);   // a bottle mask reached this instance
            if (detected)
            {
                // A mask ARRIVED, so this probe demonstrably resolved the object: p_vis = 1, and the evidence
                // it carries is scaled by how much of it the model actually explains — a blob that fell almost
                // entirely to the clutter component is a weak confirmation, not a full one.
                const float quality = std::clamp(1.0f - inst.last_clutter_frac, 0.0f, 1.0f);
                inst.existence.integrate(1.0f, quality * llr_detect);
            }
            else
                inst.existence.integrate(p_det, llr_absent);
        }

        // ── LiDAR channel (sweep clock) — the shared carve, unchanged ─────────────────────────────
        if (in.sweep != nullptr and not in.sweep->empty())
        {
            const auto& s = inst.ai2_belief.state();
            const rc::exist::Evidence ev = rc::exist::carve_box(
                in.origin, *in.sweep, s.cx, s.cy, 0.0f, 2.0f * s.radius, 2.0f * s.radius,
                s.cz - 0.5f * s.height, s.cz + 0.5f * s.height, t.sigma_pos_m, sm);
            inst.dbg_ex_lidar_occ  = ev.e_occ;
            inst.dbg_ex_lidar_free = ev.e_free;
            inst.dbg_ex_lidar_n    = ev.n_reached;

            // ★A SWEEP CANNOT JUDGE THE EXISTENCE OF AN OBJECT IT CANNOT LOCALISE BETTER THAN THAT OBJECT'S
            // OWN SIZE. carve_box blurs its surface test by sigma_surf = hypot(sensor_sigma, position_sigma).
            // MEASURED on the live bottle: sigma_surf 0.0369 m against a fitted radius of 0.0250 m — the blur
            // is 1.5x the whole object. Every beam crossing that 5 cm box then reports "passed through"
            // whether the bottle is absent or merely 3 cm from where we think, and the two are exactly what
            // the channel is supposed to distinguish. It answered with confident absence: free 22.4 vs occ
            // 8.9, L walked +4 -> -4 in 120 cycles, and the removal debounce reached 13.4 of 15 — for a
            // bottle YOLO was detecting on EVERY ONE of those cycles (detected=1, p_detect ~0.25).
            //
            // The honest weight is the same shape as the camera's p_detect, and it costs no new parameter
            // because both terms already exist in the belief: the share of the beam-placement distribution
            // that actually lands on the object.
            //
            //     p_resolve = r / (r + sigma_surf)
            //
            // -> 1 for an object far larger than the registration blur (a fridge: unchanged), -> 0 for one
            // far smaller (a bottle at arm's length: the sweep holds instead of voting). It is a continuous
            // covariate ratio, not a size gate: as the fit tightens, sigma_pos falls and the SAME bottle
            // earns its LiDAR vote back. For the numbers above it reads 0.40, which turns a net -0.28
            // nats/cycle (deletion in ~120 cycles) into a net +1.4 (the camera's confirmation wins, correctly).
            const float sigma_surf = std::hypot(cfg_.existence_sensor_sigma_m, t.sigma_pos_m);
            const float r_eff      = std::max(1e-3f, s.radius);
            const float p_resolve  = r_eff / (r_eff + sigma_surf);
            inst.dbg_ex_lidar_pres = p_resolve;
            inst.existence.integrate(ev, p_resolve);
        }

        inst.exist_logodds = inst.existence.logodds();

        // ── removal: a decision on L, debounced by LOOKS rather than cycles ───────────────────────
        // ★Counting CYCLES would let a bottle be removed by idling next to it: the streak would advance on
        // frames that resolved nothing. Accumulating p_detect instead means the debounce measures how much
        // genuine looking has happened, so `existence_remove_frames` is a number of IDEAL observations.
        // The LiDAR carve moves L but does not advance the streak — at 7 cm across, a bottle is at the very
        // edge of what a sweep resolves, and the camera is the modality that has to see it.
        if (inst.existence.should_remove(cfg_.existence_removal_prob))
            inst.existence_remove_streak += cycle_p_detect;
        else
            inst.existence_remove_streak = 0.0f;

        // ── per-cycle existence trace ─────────────────────────────────────────────────────────────
        // ★WITHOUT THIS THE CHANNEL IS UNOBSERVABLE UNTIL IT DELETES SOMETHING, which is exactly backwards
        // for a channel whose documented failure mode is going SILENTLY INERT (vis = 0 ⇒ p_detect = 0 ⇒ L
        // never moves ⇒ every bottle immortal, while everything looks healthy). `vis` and `occl_dropped` are
        // the two columns that catch it: occl_dropped counts the obstacles discarded because the bottle stands
        // INSIDE them, so occl_dropped = 0 together with vis = 0 means the support table is still eating the
        // sightline. Locale-pinned on purpose (CLAUDE.md): under es_ES a stream would emit decimal COMMAS into
        // a comma-separated file, and the reader would silently truncate every value it parsed back.
        if (log_now)
        {
            static std::ofstream ex_csv = []{
                std::ofstream f("etc/bottle_existence_log.csv", std::ios::trunc);
                f.imbue(std::locale::classic());
                f << "cycle,node,L,p_exists,cx,cy,detected,since_det,p_detect,fill_max,fill_min,"
                     "vis,occl_total,occl_dropped,cam_usable,lidar_occ,lidar_free,lidar_n,lidar_p_resolve,"
                     "remove_streak\n";
                return f; }();
            if (ex_csv)
            {
                const auto& st = inst.ai2_belief.state();
                ex_csv << ex_cycle << ',' << inst.node_name << ',' << inst.existence.logodds() << ','
                       << inst.existence.p_exists() << ',' << st.cx << ',' << st.cy << ','
                       << (inst.frames_since_detection == 0 ? 1 : 0) << ',' << inst.frames_since_detection << ','
                       << inst.dbg_ex_p_detect << ',' << inst.dbg_ex_fill << ',' << fill_min_dbg << ','
                       << inst.dbg_ex_vis << ',' << obstacles.size() << ',' << dropped_dbg << ','
                       << (camera_usable ? 1 : 0) << ',' << inst.dbg_ex_lidar_occ << ','
                       << inst.dbg_ex_lidar_free << ',' << inst.dbg_ex_lidar_n << ','
                       << inst.dbg_ex_lidar_pres << ','
                       << inst.existence_remove_streak << '\n';
                ex_csv.flush();
            }
        }

        if (inst.existence_remove_streak >= static_cast<float>(cfg_.existence_remove_frames))
            doomed.push_back(id);
    }

    for (const std::uint64_t id : doomed)
    {
        auto& insts = fitter.instances();
        if (auto it = insts.find(id); it != insts.end())
        {
            std::print("bottle_concept: [existence] REMOVE id={} (log-odds {:.2f} < {:.2f}; "
                       "p_detect {:.2f}, fill {:.3f}, vis {:.2f})\n",
                       id, it->second.exist_logodds,
                       std::log(cfg_.existence_removal_prob / (1.0f - cfg_.existence_removal_prob)),
                       it->second.dbg_ex_p_detect, it->second.dbg_ex_fill, it->second.dbg_ex_vis);
            on_remove(id, it->second);
        }
    }
}

}  // namespace rc
