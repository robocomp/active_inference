/*
 * residual_fitter.cpp — see residual_fitter.h
 */

#include "residual_fitter.h"

#include <algorithm>
#include <cmath>
#include <print>

namespace rc
{

ResidualFitter::ResidualFitter(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                               ResidualConfig& cfg)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg)
{
    if (not cfg_.ai2_csv_path.empty())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (ai2_csv_.is_open())
            ai2_csv_ << "cycle,node,pts,zed_pts,R,energy,cx,cy,yaw,w,d,s_cx,s_cy,s_yaw,s_w,s_d\n";
    }
}

bool ResidualFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_node_id)
{
    room_node_id_ = room_node_id;
    if (instances_.count(node.id()))
        return false;

    ResidualInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();

    ResidualState st;
    st.w      = G_->get_attrib_by_name<width_m_att> (node).value_or(0.40f);
    st.d      = G_->get_attrib_by_name<depth_m_att> (node).value_or(0.40f);
    st.height = G_->get_attrib_by_name<height_m_att>(node).value_or(0.80f);
    if (inner_eigen_)
        if (const auto t = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), node.name(), 0);
            t.has_value())
        { st.cx = static_cast<float>(t->x()); st.cy = static_cast<float>(t->y()); st.cz = static_cast<float>(t->z()); }

    if (const auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        st.cx = it->second.x(); st.cy = it->second.y();
        birth_seeds_.erase(it);
    }

    ResidualModelParams mp{cfg_.belief.min_size, cfg_.belief.max_size};
    inst.model = ResidualModel(mp);
    inst.model.set_state(st);
    instances_.emplace(node.id(), std::move(inst));
    return true;
}

ResidualFitter::ResidualObservation ResidualFitter::observe(ResidualInstance& inst)
{
    ResidualObservation obs;
    if (not clusters_ or inst.assigned_cluster_idx < 0 or
        inst.assigned_cluster_idx >= static_cast<int>(clusters_->size()))
        return obs;
    obs.cluster = &(*clusters_)[inst.assigned_cluster_idx];
    obs.has_fresh_data = not obs.cluster->points.empty();
    if (obs.has_fresh_data)
        ++inst.matched_frames;
    return obs;
}

float ResidualFitter::run_inference(ResidualInstance& inst, const ResidualObservation& observation)
{
    if (not observation.has_fresh_data or observation.cluster == nullptr)
    {
        if (inst.belief_initialized)
            inst.belief.predict_stale();
        return inst.prev_free_energy;
    }
    const ResidualCluster& cl = *observation.cluster;

    if (not inst.belief_initialized)
    {
        ResidualBeliefState s0;
        s0.cx = cl.centroid.x(); s0.cy = cl.centroid.y();
        s0.yaw = cl.yaw_seed;    s0.w = cl.w_seed;  s0.d = cl.d_seed;
        inst.belief = ResidualBelief(s0, cfg_.belief);
        inst.belief_initialized = true;
    }

    ResidualFrame frame;
    frame.points = cl.points;                       // room-frame 3D; the box SDF uses only xy
    // Per-point reliability: R_i = σ² + (k_range·range)² + (lag·rot_rate·range)². Far points and points
    // captured while rotating get high R → the box widens Σ instead of chasing sparse/smeared returns.
    const float s2 = cfg_.belief.sigma_base_m * cfg_.belief.sigma_base_m;
    const float kr = cfg_.range_noise_coeff;
    const float ke = cfg_.ego_motion_lag_s * rot_rate_;   // tangential smear scale per metre of range
    const Eigen::Vector2f o = sensor_origin_.head<2>();
    frame.R.resize(frame.points.size());
    for (std::size_t i = 0; i < frame.points.size(); ++i)
    {
        const float range = (frame.points[i].head<2>() - o).norm();
        const float sr = kr * range, se = ke * range;
        frame.R[i] = s2 + sr * sr + se * se;
    }
    // ZED dense-depth boost: append the backprojected depth points (set by the worker) with their own
    // (crisper) obs noise, so the box is refined against dense depth where the camera sees it.
    if (not inst.zed_points.empty())
    {
        const float Rz = cfg_.zed_sigma_m * cfg_.zed_sigma_m;
        frame.points.insert(frame.points.end(), inst.zed_points.begin(), inst.zed_points.end());
        frame.R.insert(frame.R.end(), inst.zed_points.size(), Rz);
    }
    frame.chain_cov_xx = inst.chain_cov_xx;         // localization common-mode (set by the last write)
    frame.chain_cov_yy = inst.chain_cov_yy;

    const float energy = inst.belief.update(frame);

    // Copy the footprint posterior into the model; store the hull. HEIGHT: a mostly-horizontal LiDAR grazes an
    // obstacle in a THIN band at beam height, so the cluster z-SPAN (z_max−z_min) badly under-states the real
    // height (a flat-slab box). An obstacle occupies from the SURFACE IT RESTS ON up to the highest point seen,
    // so anchor the box bottom to the inferred support (the floor, or a tabletop when it sits on one — set by
    // the worker as inst.support_z): height = z_max − support_z, centre = ½(support_z + z_max). Conservative +
    // physical (the true top may be higher under occlusion, but never lower than the highest return).
    const float support_z = std::min(inst.support_z, cl.z_max - 0.05f);   // guard: support below the top
    ResidualState ms = inst.model.state();
    const auto& bs = inst.belief.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.yaw = bs.yaw; ms.w = bs.w; ms.d = bs.d;
    ms.cz     = 0.5f * (support_z + cl.z_max);
    ms.height = std::max(0.05f, cl.z_max - support_z);
    inst.model.set_state(ms);

    // Observed near-surface anchor: along the sensor→object axis, the closest cluster returns ARE the object's
    // visible front. Take a low percentile of the projections (outlier-robust) so the scene-graph can pin the
    // published box's near face here — the box then extends into the occlusion, never poking toward the sensor.
    const Eigen::Vector2f so = sensor_origin_.head<2>();
    Eigen::Vector2f nfn = Eigen::Vector2f(ms.cx, ms.cy) - so;
    if (const float nn = nfn.norm(); nn > 1e-3f and not cl.points.empty())
    {
        nfn /= nn;
        std::vector<float> proj; proj.reserve(cl.points.size());
        for (const auto& p : cl.points) proj.push_back((p.head<2>() - so).dot(nfn));
        const std::size_t k = proj.size() / 20;                 // 5th-percentile nearest (reject stray close points)
        std::nth_element(proj.begin(), proj.begin() + k, proj.end());
        const float near_d = proj[k];
        inst.nf_x = so.x() + near_d * nfn.x(); inst.nf_y = so.y() + near_d * nfn.y();
        inst.nf_valid = true;
    }
    inst.model.set_hull(cl.hull);

    // Convergence / divergence bookkeeping.
    const bool explained = std::isfinite(energy) and energy > 0.0f;
    inst.frames_diverged = explained ? 0 : inst.frames_diverged + 1;
    const float fe = std::isfinite(energy) ? energy : inst.prev_free_energy;
    inst.frames_converged = (std::abs(fe - inst.prev_free_energy) < cfg_.fe_eps) ? inst.frames_converged + 1 : 0;

    log_ai2_csv(inst, static_cast<int>(frame.points.size()), s2, fe);
    return fe;
}

bool ResidualFitter::should_log(const ResidualInstance& inst) const
{
    return cfg_.log_period_frames > 0 and (inst.processed_cycles % cfg_.log_period_frames) == 0;
}

void ResidualFitter::log_ai2_csv(const ResidualInstance& inst, int point_count, float R, float energy)
{
    if (not ai2_csv_.is_open())
        return;
    const auto& s = inst.belief.state();
    const auto& S = inst.belief.covariance();
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << point_count << ','
             << inst.zed_points.size() << ',' << R << ','
             << energy << ',' << s.cx << ',' << s.cy << ',' << s.yaw << ',' << s.w << ',' << s.d << ','
             << std::sqrt(S(0,0)) << ',' << std::sqrt(S(1,1)) << ',' << std::sqrt(S(2,2)) << ','
             << std::sqrt(S(3,3)) << ',' << std::sqrt(S(4,4)) << '\n';
}

}  // namespace rc
