/*
 * chair_fitter.cpp — the active-inference fit core for chair_concept (AI2 recursive-Laplace belief).
 */

#include "chair_fitter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <unordered_map>
#include <utility>

namespace rc {

ChairFitter::ChairFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         ChairConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         ChairSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph)
{}

void ChairFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = enabled and (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void ChairFitter::compute_chain_cov(ChairInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the chair centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution,
    // pinned to the mask capture stamp.
    const auto& s = inst.model.state();
    const Mat::Vector3d centre(s.cx, s.cy, s.cz);
    const auto c_src = inner_eigen_->transform(chain_src_frame_, centre, "room", inst.last_mask_timestamp_ms);
    if (not c_src.has_value())
        return;
    DSR::GaussianPoint3D gp;
    gp.mean = c_src.value();
    gp.covariance = DSR::Cov3d::Zero();
    const auto g = gaussian_->transform_point("room", gp, chain_src_frame_, inst.last_mask_timestamp_ms);
    if (not g.has_value())
        return;
    inst.chain_cov_xx = static_cast<float>(g->covariance(0, 0));
    inst.chain_cov_yy = static_cast<float>(g->covariance(1, 1));
}

// ─── Instance lifecycle ──────────────────────────────────────────────────────

bool ChairFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    ChairState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.seat_w = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.seat_d = v.value();
    // height_m carries the OVERALL height (seat_h + back_h, see ChairSceneGraph); recover seat_h by
    // subtracting back_h (default) — else seat_h reads back as the full 0.9 m and the model fits a giant chair.
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value())
        init_state.seat_h = std::max(0.10f, v.value() - init_state.back_h);

    // Read RT pose from room→chair edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G_->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G_->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2) { init_state.cx = tvec[0]; init_state.cy = tvec[1]; }
            }
            if (const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3) init_state.yaw = rvec[2];
            }
        }
    }

    // Tracker birth seed: a freshly born node's room→chair RT may not compose this cycle, so prefer the
    // detection XY the tracker handed us (consumed once) over a possibly-0,0 RT read.
    if (auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        birth_seeds_.erase(it);
    }

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF; replace any non-finite field
    // with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* name)
        {
            if (!std::isfinite(v))
            {
                std::print("chair_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.cz, 0.0f, "cz");
        fix(init_state.yaw, 0.0f, "yaw");
        fix(init_state.seat_w, 0.45f, "seat_w");
        fix(init_state.seat_d, 0.45f, "seat_d");
        fix(init_state.seat_h, 0.45f, "seat_h");
        fix(init_state.back_h, 0.45f, "back_h");
    }

    ChairInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = ChairModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name());

    instances_.emplace(node.id(), std::move(inst));
    std::print("chair_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ─────────────────────────────────────────────────────────────

ChairFitter::ChairObservation ChairFitter::observe(ChairInstance& inst, const DSR::Node& node)
{
    ChairObservation observation;

    // Detection-aliveness ages every cycle; a fresh chair mask below resets it to 0.
    if (inst.frames_since_detection < 1000000) ++inst.frames_since_detection;

    // Primary path: YOLO "masks" (room frame), masks-only. Classify-don't-destroy SDF split keeps
    // inliers as candidates and the rest as residuals that drive model expansion.
    const auto& masks_packet = mask_ingestor_->packet();
    if (masks_packet.valid && masks_packet.frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance: the tracker's gated assignment when present (multi-instance safe),
        // else the greedy nearest-to-our-centre.
        std::optional<MaskIngestor::MaskSlice> selected_mask;
        if (const auto& sl = masks_packet.slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[inst.assigned_mask_idx];
        else
            selected_mask = mask_ingestor_->select_nearest(Eigen::Vector3f(inst.model.state().cx, inst.model.state().cy, inst.model.state().seat_h), "chair");
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this chair on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence = slice.confidence;
            inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
            inst.last_motion_var  = slice.motion_var;     // AI2 ego-motion / range / truncation channels
            inst.last_motion_dotd = slice.motion_dotd;
            inst.last_trunc_frac  = slice.trunc_frac;
            inst.last_range       = slice.range;
            const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
            const std::size_t end = std::min(slice.support_end, masks_packet.support_points.size());

            std::vector<Eigen::Vector3f> candidate_pts;
            std::vector<Eigen::Vector3f> residual_pts;
            candidate_pts.reserve(end > begin ? end - begin : 0);
            residual_pts.reserve(end > begin ? end - begin : 0);

            for (std::size_t i = begin; i < end; ++i)
            {
                const auto& p = masks_packet.support_points[i];
                const float sdf = inst.model.sdf_point(p);
                if (std::abs(sdf) < cfg_.sdf_threshold_for_storage)
                    candidate_pts.push_back(p);
                else
                    residual_pts.push_back(p);
            }

            observation.has_fresh_data = true;
            observation.candidate_pts = std::move(candidate_pts);
            observation.residual_pts = std::move(residual_pts);

            if (!observation.candidate_pts.empty() || !observation.residual_pts.empty())
            {
                const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
                observation.explanation_ratio = total > 0.0f
                    ? static_cast<float>(observation.candidate_pts.size()) / total : 0.0f;

                inst.last_masks_frame_seen = masks_packet.frame_id;

                if (should_log(inst))
                    std::print("[{}] masks={} label='{}' conf={:.2f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f})\n",
                               inst.node_name, masks_packet.frame_id, slice.label, slice.confidence,
                               end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                               slice.centroid.x(), slice.centroid.y(), slice.centroid.z());
                return observation;
            }
        }
    }

    // Fallback: candidate/residual point attributes written directly on the node.
    int last_frame = -1;
    if (const auto v = G_->get_attrib_by_name<last_sensing_frame_att>(node); v.has_value())
        last_frame = v.value();

    observation.has_fresh_data = (last_frame > inst.last_frame_seen);
    if (not observation.has_fresh_data)
        return observation;

    inst.last_frame_seen = last_frame;
    observation.candidate_pts = mask_ingestor_->read_pts_attrib(node, "candidate_pts_att");
    observation.residual_pts  = mask_ingestor_->read_pts_attrib(node, "residual_pts_att");

    if (const auto v = G_->get_attrib_by_name<explanation_ratio_att>(node); v.has_value())
        observation.explanation_ratio = v.value();

    if (should_log(inst))
        std::print("[{}] ↓ frame={} cands={} resid={} expl={:.2f}\n",
                   inst.node_name, last_frame,
                   observation.candidate_pts.size(), observation.residual_pts.size(),
                   observation.explanation_ratio);
    return observation;
}

bool ChairFitter::should_log(const ChairInstance& inst) const
{
    const int period = std::max(1, cfg_.chair_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── AI2 inference (shared recursive-Laplace belief; mirrors table run_inference) ───────────────
float ChairFitter::run_inference(ChairInstance& inst, const ChairObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy belief init from the model state; on the FIRST frame snap cx,cy→centroid and cz/seat_h→z-pctiles
    // (the belief is a LOCAL filter — the tracker birth provides the coarse yaw + size seed).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        ChairBeliefState s0{m.cx, m.cy, m.cz, m.yaw, m.seat_w, m.seat_d, m.seat_h, m.back_h};
        if (npts > 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            std::vector<float> zs; zs.reserve(static_cast<std::size_t>(npts));
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v)
            { for (const auto& p : v) { sum += p; zs.push_back(p.z()); } };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            s0.cx = sum.x() / npts; s0.cy = sum.y() / npts;
            std::sort(zs.begin(), zs.end());
            const float zmid = zs[static_cast<std::size_t>(0.55f * (zs.size() - 1))];   // ~seat-top band
            s0.cz = cfg_.ai2_floor_z;                                        // PIN to the shared room floor
            s0.seat_h = std::clamp(zmid - cfg_.ai2_floor_z, 0.20f, 0.80f);   // seat top above the floor

            // Coarse yaw from the backrest offset: the backrest sits on the −local_y seat edge, so the
            // vector from the seat centroid to the upper (backrest) points runs along −local_y ⇒
            // yaw = atan2(dx, −dy). Seeds the (weakly-observed, near-square-seat) yaw near the answer so
            // the belief doesn't dwell at the wrong seed for hundreds of frames before the backrest wins.
            Eigen::Vector2f seat_c = Eigen::Vector2f::Zero(), back_c = Eigen::Vector2f::Zero();
            int n_seat = 0, n_back = 0;
            const float back_z = zmid + 0.10f, seat_lo = zmid - 0.10f;
            const auto split = [&](const std::vector<Eigen::Vector3f>& v)
            {
                for (const auto& p : v)
                {
                    if (p.z() > back_z)       { back_c += p.head<2>(); ++n_back; }
                    else if (p.z() > seat_lo) { seat_c += p.head<2>(); ++n_seat; }
                }
            };
            split(observation.candidate_pts); split(observation.residual_pts);
            if (n_back > 20 and n_seat > 20)
            {
                seat_c /= static_cast<float>(n_seat); back_c /= static_cast<float>(n_back);
                const Eigen::Vector2f d = back_c - seat_c;
                if (d.norm() > 0.05f)   // enough backrest offset to be a meaningful heading
                    s0.yaw = std::atan2(d.x(), -d.y());
            }
        }
        ChairBeliefParams p;
        p.sigma_base_m         = cfg_.ai2_sigma_base_m;
        p.clutter_frac         = cfg_.ai2_clutter_frac;
        p.clutter_scale_m      = cfg_.ai2_clutter_scale_m;
        p.clutter_structure_gain = cfg_.ai2_clutter_structure_gain;
        p.prior_size_std       = cfg_.ai2_prior_size_std;
        p.process_std_m        = cfg_.ai2_process_std_m;
        p.process_std_yaw      = cfg_.ai2_process_std_yaw;
        p.process_std_size     = cfg_.ai2_process_std_size;
        p.floor_z              = cfg_.ai2_floor_z;
        p.floor_std            = cfg_.ai2_floor_std;
        p.seat_anchor_std      = cfg_.ai2_seat_anchor_std;
        p.seat_anchor_band     = cfg_.ai2_seat_anchor_band;
        p.seat_extent_std      = cfg_.ai2_seat_extent_std;
        p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        p.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        p.common_mode_yaw_std  = cfg_.ai2_common_mode_yaw_std;
        p.gn_iters             = cfg_.ai2_gn_iters;
        inst.ai2_belief = ChairBelief(s0, p);
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);

    if (not observation.has_fresh_data)   // freeze-on-stale
    {
        compute_projected_roi(inst);
        return 0.0f;
    }

    // Static range weighting + ego-motion downweight (mirror table): a far view widens the common-mode
    // (binding yaw cap) so it can't rotate the chair; continuous, no gate.
    const float range         = std::max(0.0f, inst.last_range);
    const float range_lat_var = (cfg_.ai2_range_noise_lat_per_m * range) * (cfg_.ai2_range_noise_lat_per_m * range);
    const float range_yaw_var = (cfg_.ai2_range_noise_yaw_per_m * range) * (cfg_.ai2_range_noise_yaw_per_m * range);
    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m + std::max(0.0f, inst.last_motion_var) + range_lat_var;

    const bool gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac;
    compute_chain_cov(inst);

    float energy = 0.0f;
    if (gated)
        inst.ai2_belief.predict();
    else
    {
        ChairFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var;
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var;
        frame.chain_cov_yaw = range_yaw_var;
        energy = inst.ai2_belief.update(frame);
        // NOTE: refine_extent (coverage/extent likelihood) DISABLED — coverage without a free-space
        // counter-force is positive feedback: it inflated the footprint to cover contamination/neighbours
        // (seat → 2–5.8 m) and spawned phantom instances. Kept in the belief for reference; see Fable review.

        // 180° yaw-flip: a chair's only front/back asymmetry is the backrest, so the fit can lock 180° off
        // when the backrest was weakly seen at seed time, and no gradient step makes the π jump once yaw is
        // confidently converged. resolve_orientation() runs a SEQUENTIAL test (EMA of the yaw+π energy
        // advantage) — it corrects a sustained-wrong orientation but ignores single partial-view frames.
        if (inst.ai2_belief.resolve_orientation(frame.points, R) and should_log(inst))
            std::print("[{}] AI2 yaw 180° FLIP corrected (sustained backrest evidence)\n", inst.node_name);

        // Clutter diagnostic: how much of the assigned mask the model can't explain (off-model points —
        // e.g. the table bleeding into a chair mask, which drags the centroid). High + a position jump ⇒
        // contamination the clutter component didn't fully reject.
        inst.last_clutter_frac = inst.ai2_belief.clutter_fraction(frame.points, R);
    }

    // Write belief → legacy ChairState so downstream publish/viewer/RT code is unchanged.
    const auto& bs = inst.ai2_belief.state();
    ChairState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.cz = bs.cz; ms.yaw = bs.yaw;
    ms.seat_w = bs.seat_w; ms.seat_d = bs.seat_d; ms.seat_h = bs.seat_h; ms.back_h = bs.back_h;
    inst.model.set_state(ms);

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;
    compute_projected_roi(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} clutter={:.0f}% R={:.4f} range={:.2f} trunc={:.2f}{} | cx={:.2f} cy={:.2f} cz={:.2f} ψ={:.2f} | sw={:.2f} sd={:.2f} sh={:.2f} bh={:.2f}\n",
                   inst.node_name, npts, 100.0f * inst.last_clutter_frac, R, range, inst.last_trunc_frac, gated ? " GATED" : "",
                   bs.cx, bs.cy, bs.cz, bs.yaw, bs.seat_w, bs.seat_d, bs.seat_h, bs.back_h);

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

void ChairFitter::log_ai2_csv(const ChairInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,npts,gated,energy,R,motion_var,trunc_frac,range,clutter_frac,"
                 << "cx,cy,cz,yaw,seat_w,seat_d,seat_h,back_h,"
                 << "std_cx,std_cy,std_cz,std_yaw,std_sw,std_sd,std_sh,std_bh\n";
    }
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_trunc_frac << ',' << inst.last_range << ',' << inst.last_clutter_frac << ','
             << s.cx << ',' << s.cy << ',' << s.cz << ',' << s.yaw << ','
             << s.seat_w << ',' << s.seat_d << ',' << s.seat_h << ',' << s.back_h << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << sd(3) << ',' << sd(4) << ',' << sd(5) << ',' << sd(6) << ',' << sd(7) << '\n';
    ai2_csv_.flush();
}

// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────

std::optional<Eigen::Matrix4d> ChairFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", pose_ts_ms);
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", 0);
    if (not (rtb.has_value() and btz.has_value()))
        return std::nullopt;
    const auto to_mat4 = [](const Mat::RTMat& T)
    {
        Eigen::Matrix4d m;
        const auto& s = T.matrix();
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);   // no aligned load
        return m;
    };
    return to_mat4(rtb.value()) * to_mat4(btz.value());
}

void ChairFitter::compute_projected_roi(ChairInstance& inst)
{
    inst.roi_valid = false;
    if (not inner_eigen_)
        return;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return;
    }

    const auto Mopt = room_T_zed_matrix();   // room_T_zed (camera→room)
    if (not Mopt.has_value())
        return;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();   // room point → camera frame

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float H  = static_cast<float>(camera_api_->get_height());
    if (fx <= 0.f || fy <= 0.f || W <= 0.f || H <= 0.f)
        return;
    const float cx_px = W * 0.5f, cy_px = H * 0.5f;

    // Project the 8 box corners (top slab + footprint at floor) into the image. Camera convention
    // matches the producer: X=right, Y=forward(depth), Z=up ⇒ col=cx+X/Y·fx, row=cy−Z/Y·fy.
    const auto& s = inst.model.state();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float hw = s.seat_w * 0.5f, hh = s.seat_d * 0.5f;
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    for (const int ix : {-1, 1})
        for (const int iy : {-1, 1})
            for (const float z : {s.cz, s.cz + s.seat_h + s.back_h})
            {
                const float lx = static_cast<float>(ix) * hw, ly = static_cast<float>(iy) * hh;
                const Eigen::Vector4d Pr(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, z, 1.0);
                const Eigen::Vector4d Pc = zed_T_room * Pr;
                const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
                if (Y <= 0.20) continue;   // skip corners at/near the image plane: X/Y explodes there
                ++in_front;
                const float col = cx_px + static_cast<float>(X / Y) * fx;
                const float row = cy_px - static_cast<float>(Z / Y) * fy;
                min_col = std::min(min_col, col); max_col = std::max(max_col, col);
                min_row = std::min(min_row, row); max_row = std::max(max_row, row);
            }

    if (in_front < 4)   // need most of the box in front of the camera to trust the ROI
        return;

    const float roi_cx = 0.5f * (min_col + max_col);
    const float roi_cy = 0.5f * (min_row + max_row);
    const float off_x = (roi_cx - cx_px) / (0.5f * W);   // [-1,1], 0 = centred
    const float off_y = (roi_cy - cy_px) / (0.5f * H);
    const float fill  = std::max((max_col - min_col) / W, (max_row - min_row) / H);
    // Reject degenerate projections (robot too close / a corner grazing the image plane → the bbox
    // explodes). Mark invalid and clamp so consumers/logs never see garbage.
    const bool sane = std::isfinite(off_x) && std::isfinite(off_y) && std::isfinite(fill)
                      && std::abs(off_x) < 3.0f && std::abs(off_y) < 3.0f && fill < 4.0f;
    inst.roi_offset_x = std::clamp(off_x, -3.0f, 3.0f);
    inst.roi_offset_y = std::clamp(off_y, -3.0f, 3.0f);
    inst.roi_fill     = std::clamp(fill, 0.0f, 4.0f);
    inst.roi_valid    = sane;
}

// ─── Voxel bank (chair-owned historical memory) ──────────────────────────────

std::uint64_t ChairFitter::voxel_key(const Eigen::Vector3f& point, float quantization_m)
{
    const float q = std::max(1e-4f, quantization_m);
    const int ix = static_cast<int>(std::floor(point.x() / q));
    const int iy = static_cast<int>(std::floor(point.y() / q));
    const int iz = static_cast<int>(std::floor(point.z() / q));

    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    mix(static_cast<std::uint64_t>(ix));
    mix(static_cast<std::uint64_t>(iy));
    mix(static_cast<std::uint64_t>(iz));
    return h;
}

void ChairFitter::ingest_observation_voxels(ChairInstance& inst, const ChairObservation& observation)
{
    std::size_t inserted = 0;
    std::size_t rejected_foreign = 0;
    const auto max_points = static_cast<std::size_t>(std::max(1, cfg_.voxel_bank_max_points));

    auto ingest = [&](const std::vector<Eigen::Vector3f>& src)
    {
        for (const auto& p : src)
        {
            if (inst.voxel_bank_pts.size() >= max_points)
                break;
            if (not is_voxel_owned_by_chair(inst, p))
            {
                ++rejected_foreign;
                continue;
            }
            const auto key = voxel_key(p, cfg_.voxel_bank_quantization_m);
            if (inst.voxel_bank_keys.insert(key).second)
            {
                inst.voxel_bank_pts.push_back(p);
                ++inserted;
            }
        }
    };

    ingest(observation.candidate_pts);
    ingest(observation.residual_pts);

    if (inserted > 0 && should_log(inst))
        std::print("[{}] voxel-bank: +{} total={} (cap={}) reject_foreign={}\n",
                   inst.node_name, inserted, inst.voxel_bank_pts.size(), max_points, rejected_foreign);
}

bool ChairFitter::is_voxel_owned_by_chair(const ChairInstance& inst, const Eigen::Vector3f& point) const
{
    const auto& s = inst.model.state();

    // XY ownership gate: chair-centered radius with a configurable margin.
    const float half_diag = 0.5f * std::sqrt(s.seat_w * s.seat_w + s.seat_d * s.seat_d);
    const float gate_radius = std::max(1.0f, half_diag + cfg_.voxel_select_radius_margin_m);
    const float dx = point.x() - s.cx;
    const float dy = point.y() - s.cy;
    if (std::hypot(dx, dy) > gate_radius)
        return false;

    // Height gate to reject floor / distant clutter points in mixed scenes.
    const float z_min = -0.05f;
    const float z_max = s.cz + s.seat_h + s.back_h + cfg_.voxel_select_height_margin_m;
    return point.z() >= z_min && point.z() <= z_max;
}

// ─── Factory helpers ─────────────────────────────────────────────────────────

ChairModelParams ChairFitter::make_model_params() const
{
    ChairModelParams p;
    p.sigma_obs = cfg_.sigma_obs;
    return p;
}

}  // namespace rc
