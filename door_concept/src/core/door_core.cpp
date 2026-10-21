// door_core.cpp — see door_core.h. The tracker slice below was MOVED from specificworker.cpp (run_instance_tracker,
// merge_overlapping_instances) and door_fitter.cpp (frame_admissible, update_ego_motion, point_in_room) with its
// comments; what changed is only where each value comes from (the input / the TrackerWorld instead of the graph,
// the ingestors and the fitter) and where each decision goes (the sinks instead of std::print / log_tracker_event).
#include "door_core.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <limits>
#include <locale>
#include <numbers>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "../../../common/instance_tracker/birth_evidence.h"   // rc::birth::evidence (SHARED)
#include "../../../common/track/merge_sweep.h"                 // rc::track::merge_overlapping_sweep (SHARED, no Qt)
#include "../../../common/footprint/footprint.h"               // rc::geom::Footprint, overlap_ratio
#include "../../../common/nbv/identified_obstacle.h"           // rc::nbv::identified_obstacle (SHARED, pure)

namespace rc::door::core
{
namespace
{
using K = rc::door::DoorDecisionKind;

// ─── Footprint geometry: the plane maths is SHARED (common/footprint) ────────────────────────────
// ★A DOOR'S FOOTPRINT IS ITS APERTURE, NOT ITS LEAF — (ap_cx, ap_cy, ap_yaw) with the panel width across
// and the panel THICKNESS as the second extent. Two doors are the same door when their apertures coincide;
// a swung leaf moves while the aperture does not, so merging on the leaf would split one door in two every
// time it opened.
rc::geom::Footprint footprint_of(const rc::DoorState& s)
{ return { s.ap_cx, s.ap_cy, s.w, s.thickness, s.ap_yaw }; }

float footprint_overlap_ratio(const rc::DoorState& a, const rc::DoorState& b)
{ return rc::geom::overlap_ratio(footprint_of(a), footprint_of(b)); }

// Point-in-convex-quad, by consistent sign of the cross product around the loop. The quads here come
// straight from door_geometry (leaf or aperture), so they are convex and correctly ordered by
// construction — no general polygon test is warranted.
bool point_in_quad(const std::array<Eigen::Vector2f, 4>& q, const Eigen::Vector2f& p)
{
    int pos = 0, neg = 0;
    for (int i = 0; i < 4; ++i)
    {
        const Eigen::Vector2f& a = q[static_cast<std::size_t>(i)];
        const Eigen::Vector2f& b = q[static_cast<std::size_t>((i + 1) % 4)];
        const float cr = (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x());
        if (cr > 0.f) ++pos; else if (cr < 0.f) ++neg;
    }
    return pos == 0 or neg == 0;
}

// The live instances keyed in the WORLD's iteration order, with the find()/iteration the shared merge sweep needs.
// ★Order is the agent's unordered_map order: the sweep's pair order and its ties depend on it.
struct DoorViews
{
    using value_type = std::pair<std::uint64_t, TrackedDoor>;
    std::vector<value_type> v;
    explicit DoorViews(const std::vector<TrackedDoor>& doors)
    { v.reserve(doors.size()); for (const auto& d : doors) v.emplace_back(d.id, d); }
    auto begin() { return v.begin(); }
    auto end()   { return v.end(); }
    [[nodiscard]] std::size_t size() const { return v.size(); }
    auto find(std::uint64_t id) { return std::ranges::find(v, id, &value_type::first); }
};

const TrackedDoor* find_door(const std::vector<TrackedDoor>& doors, std::uint64_t id)
{
    const auto it = std::ranges::find(doors, id, &TrackedDoor::id);
    return it == doors.end() ? nullptr : &*it;
}
}  // namespace

DoorCore::DoorCore(const DoorCoreConfig& cfg) : cfg_(cfg)
{
    // The fit reads the core's views of the input: the mask packet, the low-LiDAR sweep, and the ONE claim set per cycle.
    fit_.set_sources(&masks_, &leaf_sweep_, &claims_);
}

void DoorCore::connect(TrackerWorld* world, DoorCoreSinks sinks)
{
    world_ = world;
    sinks_ = std::move(sinks);
    // The fit's stdout lines and CSV rows reach the adapter through the same diag sink (not into out.diag: the fit can
    // run outside step(), for a restored or carried door, where there is no cycle output to append to).
    if (sinks_.diag) fit_.set_diag([this](const DiagRow& r) { sinks_.diag(r); });
    else             fit_.set_diag({});
}

TrackedDoor tracked_view(std::uint64_t id, const rc::DoorFitState& inst)
{
    TrackedDoor d;
    d.id = id;
    d.name = inst.node_name;
    d.state = inst.model.state();
    d.aperture = inst.aperture;
    d.leaf_pose = inst.leaf_pose;
    d.phi_est = inst.phi_est;
    d.is_bearing_hypothesis = inst.is_bearing_hypothesis;
    d.ai2_initialized = inst.ai2_initialized;
    d.belief = inst.ai2_initialized ? &inst.ai2_belief : nullptr;
    d.chain_cov_xx = inst.chain_cov_xx;
    d.chain_cov_yy = inst.chain_cov_yy;
    d.roi_valid = inst.roi_valid;
    d.matched_frames = inst.matched_frames;
    d.assigned_mask_idx = inst.assigned_mask_idx;
    return d;
}

std::vector<TrackedDoor> TrackerWorld::doors() const
{
    std::vector<TrackedDoor> out;
    const auto states = fit_states();
    out.reserve(states.size());
    for (const auto& [id, inst] : states)   // ★the world's own order: association ties resolve by it
        out.push_back(tracked_view(id, *inst));
    return out;
}

void DoorCore::set_cycle_views(const DoorCycleInput& in)
{
    fit_.set_clock(in.t_steady_ns, in.t_wall_ms);
    fit_.set_camera(CameraModel::room_T_zed(in.robot), in.camera);
}

void DoorCore::record(K k, std::uint64_t id, float x, float y, std::string note, std::string name, DoorCycleOutput& out)
{
    // The same record log_tracker_event wrote: a = x, b = y (the floats, widened), c = d = 0.
    rc::door::DoorDecision d{out.cycle, out.t_ms, static_cast<std::uint16_t>(k), id, std::move(name),
                             static_cast<double>(x), static_cast<double>(y), 0.0, 0.0, std::move(note)};
    if (sinks_.decision) sinks_.decision(d);
    out.decisions.push_back(std::move(d));
}

void DoorCore::diag(std::string stream, std::string line, DoorCycleOutput& out)
{
    DiagRow r{std::move(stream), std::move(line)};
    if (sinks_.diag) sinks_.diag(r);
    out.diag.push_back(std::move(r));
}

void DoorCore::observe_media(const DoorCycleInput& in)
{
    masks_.apply(in.masks);
    leaf_sweep_.apply(in.bpearl);
    set_cycle_views(in);
}

bool DoorCore::frame_admissible(const MaskSlice& sl, const EgoMotion& ego, const rc::DoorConfig& cfg)
{
    if (sl.trunc_frac > cfg.ai2_trunc_gate_frac)
        return false;
    if (not cfg.ai2_motion_confirm_only)
        return true;
    const bool moving = ego.lin_mps   > cfg.ai2_still_lin_mps
                     or ego.ang_radps > cfg.ai2_still_ang_radps
                     or std::abs(sl.motion_dotd) > cfg.ai2_still_dotd;
    if (not moving)
        return true;
    return cfg.ai2_moving_update_center_radius >= 0.0f
       and sl.centroid_radius <= cfg.ai2_moving_update_center_radius;
}

std::vector<rc::exclusion::Claim> DoorCore::claims_from(const std::vector<ObjectInfo>& objects)
{
    // rc::exclusion::foreign_claims(G, inner, "door") = collect_graph_obstacles_identified over the object/box nodes
    // (skip the passage anchor; true extents only; the ts=0 room-frame pose) then drop our own prefix. The snapshot
    // recorded exactly those attributes and that pose for every foreign object/box node, in the same type order.
    std::vector<rc::nbv::IdentifiedObstacle> obs;
    for (const auto& o : objects)
    {
        if (o.is_anchor) continue;
        if (not rc::nbv::has_footprint(o.width_m, o.depth_m)) continue;
        if (not o.room_T_node.has_value()) continue;   // the transform did not resolve when it was recorded
        obs.push_back(rc::nbv::identified_obstacle(o.name, o.id, o.width_m.value(), o.depth_m.value(), o.height_m,
                                                   o.room_T_node.value()));
    }
    return rc::exclusion::foreign_claims_from(obs, "door");
}

DoorCycleOutput DoorCore::step(const DoorCycleInput& in)
{
    DoorCycleOutput out;
    out.cycle = in.cycle;
    out.t_ms = in.t_ms();
    if (world_ == nullptr)
        return out;
    observe_media(in);
    out.room_id = world_->room_id();
    // robot/camera speed → "be-still-to-update" gate (once per cycle) — was fitter_->update_ego_motion()
    ego_.update(CameraModel::room_T_zed(in.robot), in.t_steady_ns);
    // ONE ego-motion: the fit's confirm-only gate reads the value the birth gate uses (was fitter_->set_ego_motion).
    fit_.set_ego_motion(ego_.lin_mps, ego_.ang_radps);
    // ★ONE claim set per cycle for the SHARED mutual-exclusion rule: who else claims room space (was the graph walk at
    // the top of compute(); now the snapshot of the same nodes taken at the top of the cycle). The birth filter AND the
    // fit's point exclusion read it (Task 4b: the worker's own second walk, foreign_claims_, is gone).
    claims_ = claims_from(in.objects);
    // The containment prior's polygon, as the world holds it (live: the fit's own copy, set by refresh_room_geometry).
    fit_.set_room_polygon(world_->room_polygon());
    info_ = {};

    merge(out);                 // enforce physical exclusion before associating/birthing this cycle
    associate(in, out);
    world_->after_association(info_);
    births(in, out);
    fit_doors(out);
    return out;
}

// The fit of every door the world offers — compute()'s door loop and process_door_node's fit, moved (Task 4b) — then
// the decision log's Phi row per door. In the old order: births first (a newborn's belief initialises on its birth
// slice this very cycle), each door published right after its own fit (after_fit), the Phi rows after the loop.
void DoorCore::fit_doors(DoorCycleOutput& out)
{
    for (const auto id : world_->fit_targets())
    {
        rc::DoorFitState* inst = world_->begin_fit(id);
        if (inst == nullptr)
            continue;
        ++inst->processed_cycles;
        const auto observation = fit_.observe(*inst);

        // Stale check: skip heavy update if data hasn't moved for too long
        if (not rc::door::fitted_this_cycle(observation.has_fresh_data, inst->matched_frames))
            continue;

        const float free_energy = fit_.run_inference(*inst, observation);
        world_->after_fit(id, observation, free_energy);
    }

    // Decision log: each instance's leaf-angle posterior as it stands after this cycle's fitting.
    for (const auto& [id, inst] : world_->fit_states())
        if (not inst->is_bearing_hypothesis)
        {
            rc::door::DoorDecision d{out.cycle, out.t_ms, static_cast<std::uint16_t>(K::Phi), id, inst->node_name,
                                     inst->phi_est, inst->phi_sigma, inst->phi_measured ? 1.0 : 0.0, inst->phi_support, {}};
            if (sinks_.decision) sinks_.decision(d);
            out.decisions.push_back(std::move(d));
        }
}

// Collapse instances whose seat footprints overlap (same physical door fitted twice): keep the one with
// more integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a
// duplicate is gone before it is fed a mask. Mirrors table_concept::merge_overlapping_instances.
void DoorCore::merge(DoorCycleOutput& out)
{
    if (cfg_.door.tracker_merge_overlap <= 0.0f)
        return;

    // The sweep is SHARED (common/track); only the two per-object questions stay here.
    // ★A DOOR'S FOOTPRINT IS ITS APERTURE, NOT ITS LEAF — see footprint_of above: two doors are the same
    // door when their apertures coincide, and a swung leaf moves while the aperture does not.
    DoorViews views(world_->doors());
    info_.merges = rc::track::merge_overlapping_sweep(
        views,
        [&](const TrackedDoor& a, const TrackedDoor& b) -> std::optional<float>
        {
            const float ratio = footprint_overlap_ratio(a.state, b.state);
            return ratio >= cfg_.door.tracker_merge_overlap ? std::optional{ratio} : std::nullopt;
        },
        [](const TrackedDoor& in) { return in.matched_frames; },      // keep the more-observed instance
        [&](std::uint64_t keep, std::uint64_t drop, const TrackedDoor&, const TrackedDoor& dropped, float ratio)
        {
            diag("stdout", std::format("door_concept: [tracker] MERGE id={} into id={} (footprint overlap {:.2f})\n",
                                       drop, keep, ratio), out);
            record(K::Merge, drop, dropped.state.cx, dropped.state.cy,
                 std::format("into {} overlap {:.2f}", keep, ratio), dropped.name, out);
            out.deletes.push_back({drop, "merge"});
            world_->retire(drop);
        });
}

// Data-driven multi-instance lifecycle (mirrors table_concept). Doors are persistent furniture, so
// death is OFF by default (removed only by MERGE); birth_min_sep is wide. The only instance-lifecycle path.
void DoorCore::associate(const DoorCycleInput& in, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg.tracker_detection_noise_m;
    tp.birth_frames     = cfg.tracker_birth_frames;
    // ★Invariant 5: removal is a Bayesian decision on the existence log-odds, NEVER a miss counter. An
    //    armed death counter beside a live existence channel is a SECOND removal authority, and it is the
    //    one that carries no evidence and leaves no attributable record — a phantom analysis cannot tell a
    //    reasoned removal from a timeout. Tying it to the existence flag makes the two mutually exclusive
    //    by construction, and keeps them A/B-able: turn the channel off and the counter comes back exactly.
    //    door removes on `existence.should_remove(exist_removal_prob)` (Existence.Enabled, default true).
    tp.death_frames     = std::numeric_limits<int>::max();   // never retire by occlusion:
                                                            // removal is the existence channel's
    tp.birth_min_sep_m  = cfg.tracker_birth_min_sep_m;
    tp.nll_cost         = cfg.tracker_nll_cost;
    tracker_.set_params(tp);

    // Tracks ← live instances: centre from the fit, XY cov from the belief's position covariance Σ.
    const std::vector<TrackedDoor> doors = world_->doors();
    std::vector<rc::TrackView> tracks;
    tracks.reserve(doors.size());
    for (const auto& inst : doors)
    {
        rc::TrackView t;
        t.id = inst.id;
        const auto& s = inst.state;
        // Task 11 fix round 1: associate on the APERTURE (what the "door" mask labels — the doorway frame, door_fitter.cpp),
        // not the leaf centre, which a hinge flip moves by up to a width.
        t.xy = {s.ap_cx, s.ap_cy};
        // Is this instance EXPECTED to be seen this cycle — its model projects into the camera FoV.
        // Read by the tracker's association bookkeeping; out-of-view is a HOLD, never negative evidence.
        // (It used to feed the miss-counter death path too; that path is gone — removal is the
        // existence channel's, which weights absence by p_detect rather than by a frame count.)
        // roi_valid is a cycle stale (set in run_inference) but that is fine for a frustum test.
        t.expected_visible = inst.roi_valid;
        if (inst.ai2_initialized and inst.belief != nullptr)
        {
            // Gate association on the belief's position cov (+ chain) → Mahalanobis S = P + R²I.
            const auto& S = inst.belief->covariance();
            t.cov = Eigen::Matrix2f::Zero();
            t.cov(0, 0) = S(0, 0) + inst.chain_cov_xx;
            t.cov(1, 1) = S(1, 1) + inst.chain_cov_yy;
            t.has_cov = true;
        }
        tracks.push_back(t);
        world_->set_assigned_mask(inst.id, -1);   // cleared; re-set below only if associated this cycle
    }

    // Detections ← this frame's "door" mask slices (carry the slice index for the assignment).
    dets_.clear();
    const auto& pkt = masks_.packet();

    // ★BIRTH EVIDENCE — the shared CREATE policy (common/instance_tracker/birth_evidence.h). This agent fed
    // the tracker `birth_evidence = 1.0` on EVERY compute cycle, so `birth_frames` counted cycles rather than
    // observations: at ~10 Hz compute against a ~9.5 Hz mask stream one mask frame was counted several times,
    // and "N frames" became well under a second of a single unchanging view — which is how a YOLO false
    // positive on a wall panel becomes furniture. Three rules, none of them a threshold: an OBSERVATION not a
    // cycle; birth admitted by the UPDATE rule (frame_admissible — a frame the fit would refuse may not create
    // an object); and an admissible observation still worth only its reliability (confidence x range).
    // ★A BACKWARD JUMP MEANS THE PRODUCER RESTARTED, NOT THAT THE FRAME IS STALE. mask_frame_id is a
    // PER-PROCESS publish counter: when retina restarts it resets to ~0 while this agent still holds the
    // pre-restart high-water mark, so a plain `>` is false on every subsequent frame — birth_evidence
    // becomes 0 for ever and NO DOOR CAN EVER BE BORN AGAIN in this process. Silently: the candidate
    // simply never matures, so none of the three birth-suppression logs fire and the detections keep
    // arriving normally. Observed live 2026-09-09: retina restarted 11:37:41, a door sat in plain view at
    // 4.97 m with conf 0.67 and 3094 support points, and door_concept produced no birth for minutes with
    // nothing in any log to say why. MaskIngestor::refresh already adopts the new stream on exactly this
    // condition (see its "producer restarted; adopting new stream" note) — this counter is a second,
    // private copy of the same state that was never given the same rule.
    const long pkt_frame = pkt.valid ? static_cast<long>(pkt.frame_id) : 0;
    const bool producer_restarted = pkt.valid and pkt_frame < last_birth_mask_frame_;
    if (producer_restarted)
        diag("stdout", std::format("door_concept: [birth] mask_frame_id {} < last-seen {} — producer restarted; "
                                   "adopting the new stream\n", pkt_frame, last_birth_mask_frame_), out);
    const bool birth_new_obs = pkt.valid and (pkt_frame > last_birth_mask_frame_ or producer_restarted);
    if (birth_new_obs)
        last_birth_mask_frame_ = pkt_frame;
    const rc::birth::Detectability birth_detect{cfg.ai2_periph_ref, 2.5f, 2.0f};

    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& sl = pkt.slices[i];
                // ★★ONLY THE FRONT RGB-D CAMERA MAY CREATE OR UPDATE AN OBJECT. `has_depth` is NOT that
                // question: once the producer began depth-filling ricoh masks from reprojected LiDAR it
                // publishes them as full 3D slices with has_depth = 1, so a 360° detection from BEHIND the
                // robot passed every guard written as `if (has_depth)`. Reported live on bottle_concept —
                // moving and cloning with the robot facing away, 3 m off. mask_source says which camera,
                // unambiguously, and the retina has been publishing it all along. A ricoh slice may
                // still CONFIRM a live instance (common/peripheral_channel) or raise a proto-object to go and look
                // at; it may not move one. See MaskIngestor::MaskSlice::may_fit_geometry.
            if (sl.label != "door" or sl.support_end <= sl.support_begin
                or not may_fit_geometry(sl)) continue;

            // ★A `door` LABEL IS NOT A DOOR-SIZED THING. YOLO-sem sometimes returns a positive door mask
            // delimiting a much smaller region — a jamb, a panel edge, a fragment of the leaf. It is a
            // TRUE positive for the class and a false one for the object, and accepting it is worse than
            // dropping it: assigned to a live door it covers a fraction of the predicted silhouette, so
            // occ comes out small and free large, and the door is REMOVED on the strength of having been
            // detected. Observed live 2026-09-09.
            //
            // The prior is already here and already strong — DoorModel prior_h 2.00 +-0.08 m — it was
            // just never used as a test on the observation, only inside the fit.
            //
            // ★ONLY THE "TOO SMALL" DIRECTION, and only on an UNTRUNCATED view. A mask clipped by the
            // image border, or partly occluded, has a height that is a LOWER BOUND, so judging it would
            // reject the real door exactly when the robot is close enough for it to overflow the frame.
            // Nothing is rejected for being too LARGE either: that is a merge/clutter question, not a
            // size one, and the fit already handles it.
            // ⚠MEASURED, NOT APPLIED. Two attempts at a size prior on the MASK have now been backed out:
            //   1. `if (h < prior_h - 3*sigma) continue;` — a magic cutoff, against the modelling rule.
            //   2. the same thing as a Gaussian in the state prior's own sigma — which merely HID the
            //      cutoff: at sigma 0.08 m a 1.80 m mask scores 0.044 and a 1.60 m mask 0.000, so any
            //      real door slightly clipped at the floor datum or partly occluded stopped being born
            //      at all. Observed immediately: births blocked.
            // The error was using the prior's uncertainty about the DOOR'S TRUE HEIGHT as the noise model
            // for a MASK'S MEASURED height. They are different quantities: a mask height is noisy and
            // systematically UNDER-estimates (partial segmentation, occlusion, the floor datum), so its
            // likelihood is wide and skewed, and I have not measured it.
            // So: log mask_h, decide the observation model from the distribution, apply nothing until
            // then. The reported failure — a small positive mask ending up REMOVING the door — also has
            // no established mechanism yet; a size prior may not even be the right instrument for it.
            // ★A LIKELIHOOD, NOT A CUTOFF. The first version of this was `if (h < prior_h - 3*sigma)
            // continue;` — a magic threshold built out of a prior that is already a distribution, which
            // is precisely the shape this project's modelling rule says not to add. The prior IS the
            // model; the observation's weight should fall out of it continuously.
            //
            // One-sided on purpose: a mask TALLER than the prior is not penalised (that is a merge or
            // clutter question the fit already owns), and a TRUNCATED mask is not penalised at all,
            // because a clipped or occluded height is a LOWER BOUND — judging it would discount the real
            // door exactly when the robot is close enough for it to overflow the frame. So the weight
            // only expresses "this is too short to be a whole door, seen whole".
            //
            // Nothing is rejected. A 0.7 m fragment simply carries a small fraction of an observation's
            // worth, in the same units as every other factor in birth_evidence (confidence x range x
            // unclaimed), and a full-height mask carries all of it.
            float size_w = 1.0f;
            {
                const float untrunc = 1.0f - std::clamp(sl.trunc_frac, 0.0f, 1.0f);
                const float mask_h  = sl.bbox_max.z() - sl.bbox_min.z();
                if (sl.has_depth and sl.bbox_max.allFinite() and sl.bbox_min.allFinite()
                    and untrunc > 0.5f and mask_h > 0.0f and mask_h < cfg.door_prior_h_m)
                {
                    const float z = (cfg.door_prior_h_m - mask_h) / std::max(1e-3f, cfg.door_prior_h_std);
                    size_w = std::exp(-0.5f * z * z);
                }
            }
            rc::DetectionView dv;
            dv.xy = Eigen::Vector2f(sl.centroid.x(), sl.centroid.y());
            dv.slice_index = i;
            // ZED-only BIRTH: only a ZED slice (per-pixel depth ⇒ depth_var==0) may SPAWN a door. A ricoh
            // LiDAR-reprojected-depth slice (depth_var>0) has unreliable depth/extent, so it may ASSOCIATE to /
            // confirm an existing door but must NOT birth a phantom — this is exactly what created door_3 (a
            // wrong ricoh detection at 8.8 m, outside the room). A confident-ricoh escape hatch is OFF by default.
            const bool is_zed = sl.depth_var == 0.0f;
            dv.birthable = is_zed or (cfg.ricoh_birth_enabled
                                      and sl.confidence >= cfg.ricoh_birth_conf
                                      and sl.depth_var  <= cfg.ricoh_birth_max_var);
            // One admissible, reliable observation — never a cycle. See birth_evidence.h.
            dv.birth_evidence = rc::birth::evidence({sl.confidence, sl.range}, birth_detect,
                                                    birth_new_obs, frame_admissible(sl, ego_, cfg));
            // ⚠size_w deliberately NOT multiplied in — see the note above. Carried only to be logged.
            dv.dbg_size_w = size_w;

            // ★MUTUAL EXCLUSION — no two objects occupy the same space (SHARED, common/exclusion).
            // A continuous support multiplied into the birth evidence exactly like the others above, so a
            // candidate condensing onto ANOTHER CONCEPT's object never accrues enough to mature: it is not
            // vetoed, it is unsupported. Every agent already refused to fit two of its OWN instances to one
            // object; none ever asked what a different concept had claimed, which is how a refrigerator was
            // created on top of door_3 (16 cm apart, same width, same yaw) and then could not die.
            if (not claims_.empty())
            {
                const rc::exclusion::Claim* who = nullptr;
                // A door leaf runs floor to lintel: band [0, prior height]. It is the tallest thing any of
                // these concepts publishes, so in practice it overlaps everything its footprint touches —
                // which is the honest answer, and the reason door_3 correctly claimed the phantom fridge.
                const float unclaimed = rc::exclusion::p_unclaimed(
                    {dv.xy.x(), dv.xy.y(), cfg.door_prior_w_m, 0.10f /*leaf thickness*/, 0.0f}, claims_, &who,
                    0.0f, cfg.door_prior_h_m);
                dv.birth_evidence *= unclaimed;
                if (unclaimed < 0.99f)
                    diag("stdout", std::format("[door] birth cand CLAIMED by '{}' ({:.0f}%): birth_ev x{:.2f}\n",
                                               who ? who->node : "?", 100.0f * (1.0f - unclaimed), unclaimed), out);
            }
            dets_.push_back(dv);
        }
    // The remembered-door look reads this frame's detections as "seen there" (door_look.h, step_remembered_looks).
    info_.new_frame = birth_new_obs;
    info_.dets_xy.clear();
    for (const auto& d : dets_) info_.dets_xy.push_back(d.xy);

    // DIAGNOSTIC (merged-vs-single mask): one CSV row per "door" slice per cycle — count, size, centroid,
    // range. If a cluttered scene collapses to ONE big slice (npts ≫ a clean single door) the extra doors
    // never get born because no separate detection ever arrives — the failure is upstream, not in birth. The
    // adapter writes the rows to etc/door_dets_log.csv (header: cycle,n_door_slices,slice_idx,npts,conf,cx,cy,range,
    // trunc_frac,motion_var,mask_h,size_w,mask_z0,mask_z1). Formatted through the classic locale, as the old ofstream
    // (global C++ locale "C") did.
    {
        ++dets_csv_rows_;
        std::ostringstream row;
        row.imbue(std::locale::classic());
        if (dets_.empty())
            row << dets_csv_rows_ << ",0,-1,0,0,0,0,0,0,0\n";
        for (const auto& d : dets_)
        {
            const auto& sl = pkt.slices[d.slice_index];
            const std::size_t n = (sl.support_end > sl.support_begin) ? (sl.support_end - sl.support_begin) : 0;
            row << dets_csv_rows_ << ',' << dets_.size() << ',' << d.slice_index << ',' << n << ',' << sl.confidence
                << ',' << sl.centroid.x() << ',' << sl.centroid.y() << ',' << sl.range << ','
                << sl.trunc_frac << ',' << sl.motion_var << ','
                << (sl.bbox_max.z() - sl.bbox_min.z()) << ',' << d.dbg_size_w << ','
                << sl.bbox_min.z() << ',' << sl.bbox_max.z() << '\n';
        }
        diag("door_dets_log.csv", row.str(), out);
    }

    res_ = tracker_.update(tracks, dets_);

    const int n_assigned = static_cast<int>(std::count_if(res_.assignment.begin(), res_.assignment.end(),
                                                          [](int a){ return a >= 0; }));

    // Evidence-pipeline counters (section 1 of the dashboard). Same fields every concept agent fills.
    info_.mask_frame_id  = pkt.valid ? pkt.frame_id : -1;
    info_.total_slices   = pkt.valid ? static_cast<int>(pkt.slices.size()) : 0;
    info_.class_dets     = static_cast<int>(dets_.size());
    info_.assigned       = n_assigned;
    info_.tracker_births = static_cast<int>(res_.births.size());
    info_.tracker_deaths = static_cast<int>(res_.deaths.size());
    if (++dbg_cycles_ % 30 == 0 or not res_.births.empty() or not res_.deaths.empty())
        diag("stdout", std::format("[tracker] instances={} door_dets={} assigned={} unassigned={} births={} deaths={}\n",
                                   tracks.size(), dets_.size(), n_assigned,
                                   static_cast<int>(dets_.size()) - n_assigned, res_.births.size(), res_.deaths.size()),
             out);

    // ASSOCIATE: route each detection's mask slice to its instance (read in observe()).
    {
        // §3.1 (Fable, PERCEPTION_ASSOCIATION_PLAN.md): associate by MODEL EVIDENCE, not centroid. The
        // teleport was: a merged mask's centroid lands MIDWAY between two doors → falls in the wrong
        // track's Mahalanobis gate → greedy flip. Instead score each (initialised instance × door slice)
        // by the instance's belief NLL on the slice's POINTS — a merged mask fits NO single-door model, and
        // each instance keeps the slice that best matches ITS geometry. Greedy lowest-nll, 1-to-1, no gate.
        // ★The evidence is `association_nll` = the mixture NLL, NOT `mean_energy`: mean_energy weights the SDF
        // by responsibility, so a FAR slice (all points → clutter) scored ~0 = a PERFECT match → a distant
        // instance mis-claimed the 3rd door's slice and SUPPRESSED its birth (door "never seen"). The
        // mixture NLL counts the clutter cost, so a far/clutter'd slice scores HIGH → unclaimed → it births.
        struct EvPair { float e; std::uint64_t id; int slice; };
        std::vector<EvPair> pairs;
        const float R = cfg.ai2_sigma_base_m * cfg.ai2_sigma_base_m;
        const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
        for (const auto& inst : doors)
        {
            if (not inst.ai2_initialized or inst.belief == nullptr) continue;
            const Eigen::Vector2f bc = inst.belief->center_xy();   // room-frame panel centre
            const auto& BS = inst.belief->covariance();
            // Σ(0,0)=σ_s² is the along-wall position variance; use it isotropically for the position gate.
            const float pos_var = BS(0, 0);
            const float sxx = pos_var + inst.chain_cov_xx + rn2;   // innovation cov diag S = P + R²I
            const float syy = pos_var + inst.chain_cov_yy + rn2;
            for (const auto& d : dets_)
            {
                // POSITION GATE (mirrors the tracker's S=P+R²I Mahalanobis): an instance may claim a slice
                // ONLY if the slice centroid is within its gate. Without it the evidence greedy assigns EVERY
                // instance some slice, so an instance whose own door is occluded this frame claims a FAR
                // door's slice (high nll, but the least-bad available) → suppresses THAT door's birth AND
                // teleports the instance (the "3rd door never seen" + wrong-pose bug). Gate by position,
                // rank by shape-evidence = the correct combination.
                const float ex = d.xy.x() - bc.x(), ey = d.xy.y() - bc.y();
                const float m2 = ex * ex / std::max(1e-9f, sxx) + ey * ey / std::max(1e-9f, syy);
                if (m2 > cfg.tracker_gate_mahalanobis) continue;   // outside the gate → not claimable
                const auto pts_span = masks_.support(pkt.slices[d.slice_index]);
                if (pts_span.empty()) continue;
                const std::vector<Eigen::Vector3f> pts(pts_span.begin(), pts_span.end());
                pairs.push_back({inst.belief->association_nll(pts, R), inst.id, d.slice_index});
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](const EvPair& a, const EvPair& b) { return a.e < b.e; });
        std::unordered_set<std::uint64_t> used_inst; std::unordered_set<int> used_slice;
        for (const auto& p : pairs)
        {
            if (used_inst.count(p.id) or used_slice.count(p.slice)) continue;
            world_->set_assigned_mask(p.id, p.slice);
            used_inst.insert(p.id); used_slice.insert(p.slice);
        }

        // Pass 2 — INITIALISATION: a freshly-born instance has no belief yet (Pass 1 skips it), so it has no
        // way to earn its first mask. Assign it the nearest UNUSED slice within the metric fallback gate of
        // its BIRTH position, so it initialises AT the door it was born from — NOT the nearest door (that
        // was the teleport: a far-born instance grabbed a near door's mask when its own was occluded, then
        // merged, so the far door never persisted). If no slice sits at its birth spot this frame it stays
        // unassigned (frozen) — no teleport.
        const float fb2 = cfg.tracker_gate_fallback_m * cfg.tracker_gate_fallback_m;
        for (const auto& inst : doors)
        {
            if (inst.ai2_initialized or used_inst.count(inst.id)) continue;
            const auto& ms = inst.state;
            int best = -1; float best_r2 = fb2;
            for (const auto& d : dets_)
            {
                if (used_slice.count(d.slice_index)) continue;
                const float ex = d.xy.x() - ms.cx, ey = d.xy.y() - ms.cy, r2 = ex * ex + ey * ey;
                if (r2 < best_r2) { best_r2 = r2; best = d.slice_index; }
            }
            if (best >= 0) { world_->set_assigned_mask(inst.id, best); used_slice.insert(best); used_inst.insert(inst.id); }
        }
    }
    // ★(The peripheral ricoh-360 confirmation block that followed here in run_instance_tracker was DEAD CODE: it set
    // expected_visible on the LOCAL `tracks` vector after its last use by tracker_.update, so nothing ever read it.
    // Dropped with the move; no decision depended on it.)
}

// BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
// fitter with the detection XY so the model starts AT the door (not the 0,0 RT-read default).
void DoorCore::births(const DoorCycleInput& in, DoorCycleOutput& out)
{
    (void) in;
    const auto& cfg = cfg_.door;
    const auto& pkt = masks_.packet();
    for (const int d : res_.births)
    {
        const int slice = dets_[d].slice_index;
        // §3.1: under AI2 the assignment is by belief evidence, not the tracker's centroid — so a slice an
        // existing instance already claimed by mean_energy must NOT also spawn a phantom (the tracker's
        // centroid birth and the evidence assignment can disagree). Full birth-validity (P(v=clean) — don't
        // birth from a merged/contaminated mask at all) is §2, still to come.
        const Eigen::Vector3f& c = pkt.slices[slice].centroid;
        // ★Re-read every birth: an earlier birth this cycle is already an instance (a newborn's swept arc can explain
        // the next detection), and the existence update may have retired doors since the association.
        const std::vector<TrackedDoor> doors = world_->doors();

        // Room-containment pose prior at BIRTH: never spawn a door outside the walls (a mislocalized frame while
        // the robot is lost drops a detection beyond a wall). Zero prior mass outside the room → suppress.
        if (const auto poly = world_->room_polygon();
            cfg.exist_room_prior and poly.size() >= 3
            and not point_in_polygon(poly, Eigen::Vector2f(c.x(), c.y()), cfg.exist_room_margin_m))
        {
            diag("stdout", std::format("door_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — OUTSIDE room\n",
                                       slice, c.x(), c.y()), out);
            record(K::Suppress, 0, c.x(), c.y(), "outside room", {}, out);
            continue;
        }

        // MINIMUM-HEIGHT prior at BIRTH: a door is an aperture a person walks THROUGH, so a blob that
        // demonstrably tops out below MinHeightM is not one — it is a counter, a radiator, a window sill,
        // a low cabinet. Zero prior mass ⇒ suppress, exactly like the room-containment prior above.
        //
        // Judged on the SUPPORT bbox top, never on the fitted h: the template anchor pins h at 2.0 m
        // regardless of evidence, so by the time an instance exists it always claims to be door-height.
        // (Live: the phantom at (−4.22,−2.16) reported h ≡ 2.000 for its whole life.)
        //
        // "demonstrably" is doing real work — a mask clipped by the image border has an UNOBSERVED top, so
        // its bbox top is only a lower bound. The real door is routinely clipped, so suppressing on a
        // truncated view would refuse to ever birth it. Require the view to be substantially untruncated.
        if (cfg.exist_min_height_prior and pkt.slices[slice].has_depth and pkt.slices[slice].bbox_max.allFinite())
        {
            const float top    = pkt.slices[slice].bbox_max.z();
            const float untrunc = 1.0f - std::clamp(pkt.slices[slice].trunc_frac, 0.0f, 1.0f);
            if (untrunc > 0.5f and top < cfg.exist_min_height_m)
            {
                diag("stdout", std::format("door_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — support tops "
                                           "at {:.2f} m < {:.2f} m (untruncated view: it is NOT door-height)\n",
                                           slice, c.x(), c.y(), top, cfg.exist_min_height_m), out);
                record(K::Suppress, 0, c.x(), c.y(), std::format("top {:.2f}m", top), {}, out);
                continue;
            }
        }
        // ★AN OPEN LEAF EXPLAINS ITS OWN DETECTION. Once phi is estimated (M1) the model knows where the
        // panel physically is — swung out of the doorway, often near-perpendicular to the robot — and a
        // `door` mask landing there is the SAME door, seen in its new pose. Without this the leaf reads
        // as an unexplained detection half a width from the aperture, and a second door is born from the
        // first one obeying an order this agent itself gave. The aperture is tested too: a mask in the
        // now-empty doorway (the frame, validated by the RGB branch) is likewise already accounted for.
        //
        // ★THIS IS EXPLAINING-AWAY, NOT A VETO. It suppresses a BIRTH only where the existing belief
        // already predicts an object; it never suppresses evidence reaching an instance, and it cannot
        // hide a genuinely new door that is anywhere else. A phi estimate that is wrong stops explaining
        // and the birth goes ahead — which is the right failure, and is visible in the log line.
        {
            const Eigen::Vector2f cxy(c.x(), c.y());
            std::uint64_t explainer = 0;
            const char* by = "";
            for (const auto& inst : doors)
            {
                if (inst.is_bearing_hypothesis) continue;
                // ★TEST THE SWEPT REGION, NOT THE CURRENT POSE. Testing only the leaf quad at the
                // CURRENT phi has a chicken-and-egg failure that was observed live: both doors are born
                // in the same cycle, so when the leaf detection arrives the first instance has not yet
                // estimated phi, its quad is still in the doorway, and the second door is created
                // anyway — after which the APERTURE instance is the one that dies and the spurious leaf
                // instance survives. Backwards.
                //
                // A hinged leaf of width w sweeps a disc of radius w about its hinge. Any `door` mask
                // inside that disc is a POSSIBLE POSE OF THIS DOOR at some angle — that is kinematics,
                // known from the model, independent of the current estimate and available from the
                // first cycle. Two distinct doors sharing a hinge within one leaf-width is not a real
                // configuration, so the cost of the wider test is negligible against what it prevents.
                // ★HINGE-AGNOSTIC, because M0 does not know the hinge. door_geometry.h pins HingeSide
                // to Near and leaves the real choice to M2, so a disc centred on the ASSUMED hinge is
                // centred on the wrong end of the door whenever that pin is wrong — and misses by up to
                // a full leaf width. Measured 2026-09-09: the spurious instance sat 1.306 m from the
                // aperture centre while the from-the-hinge radius was 0.865 m, so it was never tested.
                //
                // The leaf hinges on ONE of the two vertical edges, each half_w from the aperture
                // centre, and reaches w from its hinge. So the union of everything it can occupy under
                // either choice is a disc about the APERTURE CENTRE of radius half_w + w — exact, not a
                // margin, and it is what the model genuinely knows while the hinge is unresolved.
                const rc::door::LeafPose& L = inst.leaf_pose;
                const float leaf_w  = 2.0f * L.half_w;
                const float swept_r = L.half_w + leaf_w + L.half_t;
                if ((cxy - inst.aperture.centre_xy()).norm() <= swept_r)
                { explainer = inst.id; by = "leaf swept arc (hinge unresolved)"; break; }
                if (point_in_quad(rc::door::footprint(inst.aperture), cxy)) { explainer = inst.id; by = "aperture"; break; }
            }
            if (explainer != 0)
            {
                const TrackedDoor* it = find_door(doors, explainer);
                const float phi_deg = (it != nullptr) ? it->phi_est * 180.0f / std::numbers::pi_v<float> : 0.0f;
                diag("stdout", std::format("door_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — "
                                           "explained by {}'s {} (phi {:.0f} deg)\n",
                                           slice, c.x(), c.y(), it != nullptr ? it->name : std::string{"?"},
                                           by, phi_deg), out);
                record(K::Suppress, explainer, c.x(), c.y(), std::format("{} phi {:.0f}deg", by, phi_deg),
                     it != nullptr ? it->name : std::string{}, out);
                continue;
            }
        }

        {
            std::uint64_t claimer = 0;
            for (const auto& inst : doors)
                if (inst.assigned_mask_idx == slice) { claimer = inst.id; break; }
            if (claimer != 0)
            {
                // DIAGNOSTIC (missing-door): a persistently-detected cluster that never instantiates is
                // usually a birth SUPPRESSED here — a distant instance's belief mis-claimed this slice by
                // mean_energy, stealing it from birth (and likely teleporting toward it). Surface which
                // instance claimed it and how far it sits, so the failure is visible, not silent.
                float dist = -1.0f;
                const TrackedDoor* it = find_door(doors, claimer);
                if (it != nullptr)
                    dist = std::hypot(it->state.cx - c.x(), it->state.cy - c.y());
                diag("stdout", std::format("door_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — claimed by "
                                           "id={} ({:.2f} m away)\n", slice, c.x(), c.y(), claimer, dist), out);
                record(K::Suppress, claimer, c.x(), c.y(), std::format("claimer {:.2f}m", dist),
                     it != nullptr ? it->name : std::string{}, out);
                continue;
            }
        }
        // RE-ACQUISITION: is this detection a door that was removed and has come back at the same place? If so
        // it is not a new object — it takes its old name and resumes the belief it had converged to, instead of
        // becoming door_N+1 with template priors. (Live run: one real door burned three identities this way.)
        std::optional<GhostMatch> revive;
        std::string preferred_name;
        if (auto g = world_->match_ghost(Eigen::Vector2f(c.x(), c.y())); g.has_value())
        {
            preferred_name = g->name;
            diag("stdout", std::format("door_concept: [tracker] RE-ACQUIRE '{}' at ({:.2f},{:.2f}) — {:.2f} m from where it was "
                                       "removed after {} cycles; resuming its belief\n",
                                       g->name, c.x(), c.y(), (g->xy - Eigen::Vector2f(c.x(), c.y())).norm(),
                                       g->lived_cycles), out);
            revive = std::move(g);
        }
        // No ghost from THIS run, but the place may still be a door we have met before. The persistent table
        // returns the NAME only: geometry has to be re-earned from evidence (a remembered belief would be
        // asserting a shape nothing in this session has seen), while the identity is a fact about the
        // apartment, not about the session. This is what stops door_1/door_2 from trading doors on restart.
        else if (const auto idn = world_->match_identity(Eigen::Vector2f(c.x(), c.y())); idn.has_value())
        {
            preferred_name = idn->name;
            diag("stdout", std::format("door_concept: [tracker] IDENTITY '{}' at ({:.2f},{:.2f}) — {:.2f} m from the aperture "
                                       "remembered in etc/door_identities.csv; same door, name restored (belief re-earned)\n",
                                       idn->name, c.x(), c.y(), (idn->xy - Eigen::Vector2f(c.x(), c.y())).norm()), out);
        }
        BirthRequest req;
        req.centroid = c;
        req.room_id = world_->room_id();
        req.preferred_name = preferred_name;
        req.reserved = world_->reserved_names();   // a number a ghost or a stored identity holds is not free
        req.slice = slice;
        const auto new_id = world_->create_door(req);
        if (new_id != 0)
        {
            // Room-frame XY the newborn's model was MEANT to cold-start at (consumed once by init_instance): the room→door
            // RT written at birth is not reliably composable the same cycle (was fitter_->note_birth inside create_door).
            // ⚠DEAD IN PRACTICE, KEPT (behaviour-preserving slice; found while replaying the fit, task-4b-report.md): live,
            // the node insert inside create_door fires modify_node_slot SYNCHRONOUSLY (same thread ⇒ a direct call), so
            // ensure_instance has already built the instance — before the RT edge and before this seed — with its
            // pre-belief box at (0, 0). This seed is never consumed; the birth cycle's first estimate_phi selects its
            // rays around (0, 0) (tourR1 door_1: 14686 rays against 114 around the seed). Parked for the defect phase.
            fit_.note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
            out.creates.push_back({++create_seq_, CreateKind::DoorFromDetection, req.room_id, {}, preferred_name,
                                   Mat4d::Identity(), cfg.door_prior_w_m, cfg.door_thickness_m, cfg.door_prior_h_m,
                                   std::nullopt, std::nullopt, 0, std::format("slice {}", slice)});
            out.creates.back().pose.block<3, 1>(0, 3) = c.cast<double>();
            if (revive.has_value())
            {
                world_->reacquire(new_id, revive.value());
                record(K::Reacquire, new_id, c.x(), c.y(), preferred_name, preferred_name, out);
                world_->forget_ghost(Eigen::Vector2f(c.x(), c.y()));   // consumed — the door is alive again
            }
            // Materialise the DoorInstance NOW, at birth — do not wait for the freshly inserted DSR node
            // to surface in get_nodes_by_type (it is not reliably visible the same cycle). Birth was
            // decoupled across three async steps (create node → ensure_instance when the node appears →
            // belief init when a mask associates), so a born instance was never registered as a track:
            // the tracker kept seeing its detection unexplained and RE-BIRTHED it every cycle (the
            // 403-birth / 401-merge churn, duplicates collapsing at 0,0). Creating the instance here
            // registers it as a track immediately and hands it its birth slice so observe()/run_inference
            // initialise the belief AT the detection centroid this cycle — birth, track, and init are now atomic.
            const auto name = world_->adopt_birth(new_id, slice, c);
            out.creates.back().name = name.value_or(std::string{});
            record(K::Birth, new_id, c.x(), c.y(), {}, name.value_or(std::string{}), out);
        }
    }
}

std::string allocate_door_name(std::string_view preferred_name, const std::vector<std::string>& graph_door_names,
                               std::span<const std::string> reserved_names)
{
    // Auto-name: the LOWEST "door_<N>" (N >= 1) that is neither LIVE nor REMEMBERED (2026-10-12, was one past the
    // highest — names only ever grew: door_40 after a few crossings).
    std::set<int> taken;
    const auto take = [&](std::string_view nm)
    {
        if (not nm.starts_with("door_")) return;
        int v = 0;
        const std::string_view digits = nm.substr(5);   // "door_" = 5 chars
        if (const auto [p, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), v);
            ec == std::errc{} and p == digits.data() + digits.size())
            taken.insert(v);
    };
    bool preferred_free = not preferred_name.empty();
    for (const auto& nm : graph_door_names)
        if (nm.rfind("door_", 0) == 0)
        {
            if (nm == preferred_name) preferred_free = false;   // still occupied → don't collide
            take(nm);
        }
    // A number a GHOST or a stored identity still holds is not free either. Recycling it was the identity defect:
    // `door_1` died, its number went straight to a different door across the room, and when THAT one died it
    // overwrote the original's ghost — so the real door_1 came back as door_3. Reserved names never mark
    // `preferred_name` occupied, since that ghost is precisely the one being consumed here.
    for (const auto& nm : reserved_names)
        take(nm);
    // RE-ACQUISITION: a door that flickered out and came back keeps its name, so downstream consumers see the
    // same object rather than a fresh one. (The DSR id necessarily changes — the old node was deleted.)
    if (preferred_free)
        return std::string(preferred_name);
    int n = 1;
    while (taken.contains(n)) ++n;
    return "door_" + std::to_string(n);
}

bool point_in_polygon(const std::vector<Vec2f>& poly, const Vec2f& q, float margin_m)
{
    const std::size_t n = poly.size();
    if (n < 3)
        return true;                       // no trusted polygon → unknown room → impose no prior

    // Ray-cast parity test (even-odd rule) for strict interior.
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f& b = poly[j];
        if (((a.y() > q.y()) != (b.y() > q.y())) and
            (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y() + 1e-12f) + a.x()))
            inside = not inside;
    }
    if (inside)
        return true;
    if (margin_m <= 0.0f)
        return false;

    // Outside the polygon: accept only if within margin_m of the boundary (wall-hugging door, centroid noise).
    float best2 = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f ab = poly[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        const float t = (len2 > 1e-8f) ? std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        best2 = std::min(best2, (q - (a + t * ab)).squaredNorm());
    }
    return best2 <= margin_m * margin_m;
}

}  // namespace rc::door::core
