/*
 * human_scene_graph.cpp — DSR node/RT I/O for human_concept.
 */

#include "human_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>

#include <QDebug>

#include "body18.h"   // rc::human::KP

namespace rc {

HumanSceneGraph::HumanSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                 DSR::RT_API* rt_api,
                                 DSR::InnerEigenAPI* inner_eigen,
                                 HumanConfig& cfg,
                                 std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), inner_eigen_(inner_eigen), cfg_(cfg),
      relayout_(std::move(relayout))
{}

Eigen::Vector3f HumanSceneGraph::pelvis_of(const human::KpArray& kp)
{
    using namespace human;
    const Eigen::Vector3f lh = kp.row(KP::L_HIP).transpose();
    const Eigen::Vector3f rh = kp.row(KP::R_HIP).transpose();
    if (lh.allFinite() and rh.allFinite())
        return 0.5f * (lh + rh);
    if (lh.allFinite()) return lh;
    if (rh.allFinite()) return rh;
    // Fall back to the neck if both hips are missing.
    const Eigen::Vector3f neck = kp.row(KP::NECK).transpose();
    return neck;
}

std::vector<float> HumanSceneGraph::skeleton_mesh(const human::KpArray& kp)
{
    std::vector<float> out;
    out.reserve(human::NUM_KP * 3);
    for (int i = 0; i < human::NUM_KP; ++i)
        for (int c = 0; c < 3; ++c)
            out.push_back(kp(i, c));
    return out;
}

void HumanSceneGraph::scaffold_missing_person_nodes(const std::vector<SkeletonBody>& bodies,
                                                    std::uint64_t room_node_id)
{
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value())
        return;

    for (const auto& b : bodies)
    {
        const std::string name = "person_" + std::to_string(b.id);
        if (G_->get_node(name).has_value())
            continue;

        const Eigen::Vector3f pelvis = pelvis_of(b.kp);
        if (not pelvis.allFinite())
            continue;   // can't place it yet

        DSR::Node person_node = DSR::Node::create<person_node_type>(name);
        G_->add_or_modify_attrib_local<person_id_att>(person_node, b.id);
        {
            const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
            const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
            G_->add_or_modify_attrib_local<pos_x_att>(person_node, rpx + 180.f);
            G_->add_or_modify_attrib_local<pos_y_att>(person_node, rpy +  80.f);
        }

        const auto id_opt = G_->insert_node(person_node);
        if (not id_opt.has_value())
        {
            qWarning() << "human_concept: failed to insert node" << name.c_str();
            continue;
        }
        rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                          {pelvis.x(), pelvis.y(), pelvis.z()}, {0.0f, 0.0f, 0.0f});

        std::print("human_concept: created node '{}' id={} at room ({:.2f}, {:.2f}, {:.2f})\n",
                   name, id_opt.value(), pelvis.x(), pelvis.y(), pelvis.z());
        if (relayout_)
            relayout_();
    }
}

void HumanSceneGraph::step_write_model(HumanInstance& inst, DSR::Node& node, float free_energy)
{
    if (not inst.has_result)
        return;
    const auto& r = inst.last_result;

    // Active-perception feedback for the affordance contract (dead-banded).
    if (inst.detection_alive != inst.last_pub_detection_alive or
        std::abs(inst.last_mask_confidence - inst.last_pub_detection_conf) > 1.0f)
    {
        G_->runtime_checked_add_or_modify_attrib_local(node, "human_detection_alive",
                                                       inst.detection_alive ? 1 : 0);
        G_->runtime_checked_add_or_modify_attrib_local(node, "human_detection_confidence",
                                                       inst.last_mask_confidence);
        inst.last_pub_detection_alive = inst.detection_alive;
        inst.last_pub_detection_conf  = inst.last_mask_confidence;
    }

    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);
    G_->add_or_modify_attrib_local<model_uncertainty_att>(node, r.uncertainty_trace);
    G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
    // Publish the SMOOTHED command pose (controller output) when available, else the raw fit.
    const human::KpArray& pub_kp = inst.has_cmd ? inst.cmd_kp : r.kp_pred_aligned;
    G_->add_or_modify_attrib_local<mesh_vertices_att>(node, skeleton_mesh(pub_kp));
    G_->update_node(node);

    write_rt_pose(inst);
}

void HumanSceneGraph::write_rt_pose(HumanInstance& inst)
{
    if (inst.parent_id == 0 or not rt_api_ or not inst.has_result)
        return;

    const Eigen::Vector3f pelvis = pelvis_of(inst.has_cmd ? inst.cmd_kp : inst.last_result.kp_pred_aligned);
    if (not pelvis.allFinite())
        return;

    // Dead-band: suppress RT churn below ~1 cm.
    constexpr float kMinWriteDistSq = 0.01f * 0.01f;
    const float dx = pelvis.x() - inst.last_written_x;
    const float dy = pelvis.y() - inst.last_written_y;
    if (inst.last_written_x != std::numeric_limits<float>::max() and
        dx * dx + dy * dy < kMinWriteDistSq)
        return;

    auto parent_opt = G_->get_node(inst.parent_id);
    if (not parent_opt.has_value())
        return;

    // Body orientation from the Kabsch rotation (canonical→room): euler XYZ.
    const Eigen::Vector3f euler = inst.last_result.R.cast<float>().eulerAngles(0, 1, 2);

    rt_api_->insert_or_assign_edge_RT(parent_opt.value(), inst.node_id,
                                      {pelvis.x(), pelvis.y(), pelvis.z()},
                                      {euler.x(), euler.y(), euler.z()});

    // Covariance proxy on the RT edge: a 6×6 diagonal scaled by tr(cov). The estimator's Lambda is
    // over joint angles, not the global pose, so this is a coarse uncertainty signal until a free-flyer
    // reformulation folds the pose into the belief (see plan / Pinocchio note).
    if (auto edge = G_->get_edge(inst.parent_id, inst.node_id, "RT"); edge.has_value())
    {
        const float tr = std::isfinite(inst.last_result.uncertainty_trace)
                             ? inst.last_result.uncertainty_trace : 1.0f;
        const float pos_var = std::clamp(cfg_.pose_cov_scale * tr, 1e-4f, 1.0f);
        std::vector<float> cov_flat(36, 0.0f);
        for (int d = 0; d < 3; ++d) cov_flat[d * 6 + d] = pos_var;       // x,y,z
        for (int d = 3; d < 6; ++d) cov_flat[d * 6 + d] = 1.0f;          // rx,ry,rz (loosely constrained)
        G_->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov_flat);
        G_->insert_or_assign_edge(edge.value());
    }

    inst.last_written_x = pelvis.x();
    inst.last_written_y = pelvis.y();
}

}  // namespace rc
