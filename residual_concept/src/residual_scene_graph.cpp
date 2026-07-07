/*
 * residual_scene_graph.cpp — DSR node/RT I/O for residual_concept.
 */

#include "residual_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <print>
#include <utility>

namespace rc
{

ResidualSceneGraph::ResidualSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                       DSR::RT_API* rt_api,
                                       DSR::InnerEigenAPI* inner_eigen,
                                       ResidualConfig& cfg,
                                       std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), inner_eigen_(inner_eigen), cfg_(cfg),
      relayout_(std::move(relayout))
{}

void ResidualSceneGraph::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = enabled and (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

std::uint64_t ResidualSceneGraph::create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                                 float yaw, float w, float d, float height,
                                                                 std::uint64_t room_node_id,
                                                                 std::uint64_t timestamp_ms)
{
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value() or not rt_api_)
        return 0;

    // Auto-name: one past the highest existing "residual_<N>" obstacle node this agent owns.
    int max_n = 0;
    for (const auto& n : G_->get_nodes_by_type("obstacle"))
        if (n.name().rfind("residual_", 0) == 0)
            try { max_n = std::max(max_n, std::stoi(n.name().substr(9))); } catch (...) {}
    const std::string name = "residual_" + std::to_string(max_n + 1);

    DSR::Node node = DSR::Node::create<obstacle_node_type>(name);
    G_->add_or_modify_attrib_local<width_m_att> (node, std::max(0.05f, w));
    G_->add_or_modify_attrib_local<depth_m_att> (node, std::max(0.05f, d));
    G_->add_or_modify_attrib_local<height_m_att>(node, std::max(0.05f, height));
    {
        const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
        const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
        G_->add_or_modify_attrib_local<pos_x_att>(node, rpx + 140.f + 18.f * static_cast<float>(max_n % 6));
        G_->add_or_modify_attrib_local<pos_y_att>(node, rpy +  40.f + 18.f * static_cast<float>(max_n / 6));
    }

    const auto id_opt = G_->insert_node(node);
    if (not id_opt.has_value())
        return 0;

    const float cz = centroid_room.z() > 1e-3f ? centroid_room.z() : 0.5f * std::max(0.05f, height);
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                      {centroid_room.x(), centroid_room.y(), cz},
                                      {0.0f, 0.0f, yaw}, timestamp_ms);

    std::print("residual_concept: [tracker] BIRTH '{}' id={} at room ({:.2f},{:.2f},{:.2f}) w={:.2f} d={:.2f}\n",
               name, id_opt.value(), centroid_room.x(), centroid_room.y(), cz, w, d);
    if (relayout_)
        relayout_();
    return id_opt.value();
}

void ResidualSceneGraph::step_write_model(ResidualInstance& inst, DSR::Node& node, float free_energy,
                                          std::uint64_t room_node_id, std::uint64_t timestamp_ms)
{
    const auto& s = inst.model.state();

    // Publish only when the footprint meaningfully changed (FE jitter never fully settles → not a trigger).
    const float pe = cfg_.write_threshold_m;
    const bool changed =
        std::abs(s.cx  - inst.last_pub_cx)  > pe or
        std::abs(s.cy  - inst.last_pub_cy)  > pe or
        std::abs(s.w   - inst.last_pub_w)   > pe or
        std::abs(s.d   - inst.last_pub_d)   > pe or
        std::abs(s.yaw - inst.last_pub_yaw) > 0.05f;
    if (not changed)
        return;

    inst.last_pub_cx = s.cx; inst.last_pub_cy = s.cy;
    inst.last_pub_w  = s.w;  inst.last_pub_d  = s.d;  inst.last_pub_yaw = s.yaw;

    // Conservative publish: the fit UNDER-sizes the obstacle (only near faces seen), so grow the published
    // footprint by k·σ of the extent (occlusion inflates σ → grows on the unobserved side) + a clearance
    // margin. Internal fit/carve keep the raw estimate; only what the planner consumes is inflated.
    float infl_w = 0.0f, infl_d = 0.0f;
    if (inst.belief_initialized)
    {
        const auto& S = inst.belief.covariance();
        infl_w = cfg_.publish_extent_sigma_k * std::sqrt(std::max(0.0f, S(3, 3)));
        infl_d = cfg_.publish_extent_sigma_k * std::sqrt(std::max(0.0f, S(4, 4)));
    }
    const float grow = 2.0f * cfg_.publish_safety_margin_m;   // margin is per-SIDE → doubles the full extent
    const float w_pub = std::max(0.05f, s.w + infl_w + grow);
    const float d_pub = std::max(0.05f, s.d + infl_d + grow);

    // NEAR-FACE anchoring: the box's near (sensor-facing) face is pinned to the OBSERVED near surface (nf,
    // the closest cluster returns — set by the fitter), and ALL depth (raw + inflation) extends into the
    // occluded FAR side. This removes the "closer than it should be" bias: a symmetric box centred on the
    // near-face point cloud otherwise pokes half its depth toward the sensor into verified-free space.
    // Internal fit/carve keep s.cx/s.cy.
    Eigen::Vector2f n = Eigen::Vector2f(s.cx, s.cy) - sensor_xy_;   // sensor → box (xy)
    const float nn = n.norm();
    inst.pub_cx = s.cx; inst.pub_cy = s.cy;
    if (nn > 1e-3f and inst.nf_valid)
    {
        n /= nn;
        const Eigen::Vector2f u1(std::cos(s.yaw), std::sin(s.yaw)), u2(-std::sin(s.yaw), std::cos(s.yaw));
        const float a1 = std::abs(n.dot(u1)), a2 = std::abs(n.dot(u2));
        const float sup_pub = 0.5f * (w_pub * a1 + d_pub * a2);     // box half-support along n (inflated)
        // Put the centre one half-support BEHIND the observed near surface → near face lands exactly on nf.
        inst.pub_cx = inst.nf_x + sup_pub * n.x();
        inst.pub_cy = inst.nf_y + sup_pub * n.y();
    }
    {   // DIAGNOSTIC: is the published centre actually pushed AWAY from the sensor? (fit→nf→pub ranges)
        static int dbg = 0;
        if ((dbg++ % 40) == 0)
        {
            const float rf = (Eigen::Vector2f(s.cx, s.cy)         - sensor_xy_).norm();
            const float rn = (Eigen::Vector2f(inst.nf_x, inst.nf_y) - sensor_xy_).norm();
            const float rp = (Eigen::Vector2f(inst.pub_cx, inst.pub_cy) - sensor_xy_).norm();
            std::println("[nf-anchor] {} sensor=({:.2f},{:.2f}) range fit={:.2f} nf={:.2f} pub={:.2f} (w {:.2f}->{:.2f} d {:.2f}->{:.2f}) nf_valid={}",
                         inst.node_name, sensor_xy_.x(), sensor_xy_.y(), rf, rn, rp, s.w, w_pub, s.d, d_pub, inst.nf_valid);
        }
    }

    G_->add_or_modify_attrib_local<width_m_att> (node, w_pub);
    G_->add_or_modify_attrib_local<depth_m_att> (node, d_pub);
    // Height is already floor-anchored (bottom at the floor, top at the highest return); add the same
    // clearance margin at the top for occlusion above the highest beam.
    G_->add_or_modify_attrib_local<height_m_att>(node, std::max(0.05f, s.height + grow));
    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);
    G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);

    // Layer B: the convex-hull footprint (room frame) as flat XY pairs — the tighter footprint a future
    // planner can consume. Dilated OUTWARD by the same per-side clearance as the box (safety margin + the
    // occlusion-aware k·σ) so it too errs LARGER, never undersizing the true obstacle.
    {
        const auto& hull = inst.model.hull();
        const float per_side = cfg_.publish_safety_margin_m + 0.25f * (infl_w + infl_d);
        Eigen::Vector2f c = Eigen::Vector2f::Zero();
        for (const auto& p : hull) c += p;
        if (not hull.empty()) c /= static_cast<float>(hull.size());
        std::vector<float> hull_flat;
        hull_flat.reserve(hull.size() * 2);
        for (const auto& p : hull)
        {
            const Eigen::Vector2f r = p - c;
            const Eigen::Vector2f q = (r.norm() > 1e-4f) ? (p + per_side * r.normalized()) : p;
            hull_flat.push_back(q.x()); hull_flat.push_back(q.y());
        }
        G_->runtime_checked_add_or_modify_attrib_local(node, "footprint_hull", hull_flat);
    }

    G_->update_node(node);
    write_rt_pose(inst, room_node_id, timestamp_ms);
}

void ResidualSceneGraph::write_rt_pose(ResidualInstance& inst, std::uint64_t room_node_id, std::uint64_t timestamp_ms)
{
    if (not rt_api_)
        return;
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value())
        return;

    const auto& s = inst.model.state();
    // Publish the DIRECTIONALLY-shifted centre (near face pinned to the observed surface), not the raw fit
    // centre — see step_write_model. pub_cx/pub_cy default to s.cx/s.cy until the first conservative publish.
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), inst.node_id,
                                      {inst.pub_cx, inst.pub_cy, s.cz}, {0.0f, 0.0f, s.yaw}, timestamp_ms);

    // Attach the Laplace covariance on the RT edge (6×6 SE3, row-major [x,y,z,rx,ry,rz]). Belief Σ is 5×5
    // over [cx,cy,yaw,w,d]; map cx,cy → translation xy, yaw → rz. cz anchored (small var); roll/pitch (rx,ry)
    // unobservable for a footprint box. The controller reads this for its look-up rate.
    if (auto edge = G_->get_edge(room_node_id, inst.node_id, "RT"); edge.has_value())
    {
        float sxx = 0.01f, syy = 0.01f, sxy = 0.0f, syaw = cfg_.unobservable_var;
        if (inst.belief_initialized)
        {
            const auto& S = inst.belief.covariance();   // [cx,cy,yaw,w,d]
            sxx = S(0, 0); syy = S(1, 1); sxy = S(0, 1); syaw = S(2, 2);
        }

        // Chain/localization cov J·Σ_chain·Jᵀ (source frame → room), added to the xy translation block.
        inst.chain_cov_xx = 0.0f; inst.chain_cov_yy = 0.0f;
        if (chain_cov_enabled_ and gaussian_ and inner_eigen_)
        {
            const Mat::Vector3d centre_room(s.cx, s.cy, s.cz);
            if (const auto c_src = inner_eigen_->transform(chain_src_frame_, centre_room, "room", timestamp_ms);
                c_src.has_value())
            {
                DSR::GaussianPoint3D gp;
                gp.mean = c_src.value();
                gp.covariance = DSR::Cov3d::Zero();
                if (const auto g_fit = gaussian_->transform_point("room", gp, chain_src_frame_, timestamp_ms);
                    g_fit.has_value())
                {
                    const auto& C = g_fit.value().covariance;
                    inst.chain_cov_xx = static_cast<float>(C(0, 0));
                    inst.chain_cov_yy = static_cast<float>(C(1, 1));
                    sxx += inst.chain_cov_xx;
                    syy += inst.chain_cov_yy;
                }
            }
        }

        std::vector<float> cov_flat(36, 0.0f);
        cov_flat[0 * 6 + 0] = sxx;
        cov_flat[1 * 6 + 1] = syy;
        cov_flat[0 * 6 + 1] = sxy;
        cov_flat[1 * 6 + 0] = sxy;
        cov_flat[2 * 6 + 2] = 0.01f;                  // cz (anchored, small)
        cov_flat[3 * 6 + 3] = cfg_.unobservable_var;  // rx
        cov_flat[4 * 6 + 4] = cfg_.unobservable_var;  // ry
        cov_flat[5 * 6 + 5] = syaw;                   // rz (yaw)
        G_->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov_flat);
        G_->insert_or_assign_edge(edge.value());
    }
}

}  // namespace rc
