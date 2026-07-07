/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */

#include "room_scene_graph.h"

#include <cmath>
#include <string>
#include <vector>

#include <QString>
#include <QtCore/qdebug.h>

namespace rc
{

void RoomSceneGraph::monitor_affordance()
{
    if (G_)
        affordance_manager_.monitor_execution(G_);
}

void RoomSceneGraph::on_controller_lost()
{
    if (!G_)
        return;
    if (affordance_manager_.release_execution_claim(G_))
    {
        if (epistemic_)
            epistemic_->epistemic_planner().clear_target();
        qWarning() << "[Presence] released stale afford_room execution claim after controller loss";
    }
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::update(const rc::RoomConcept::UpdateResult& res, float adv, float side, float rot,
                           bool write_rt)
{
    last_adv_  = adv;
    last_side_ = side;
    last_rot_  = rot;

    const float sdf_mse = res.sdf_mse;
    const float cov_tt  = (res.covariance.rows() > 2 && res.covariance.cols() > 2)
                          ? res.covariance(2, 2) : 1.f;
    const bool stable   = (res.iterations_used == 0)
                          && sdf_mse < params_->STABLE_SDF_MSE_MAX
                          && cov_tt  < params_->STABLE_COV_TT_MAX;

    if (!room_node_created_)
    {
        stable_frames_ = stable ? stable_frames_ + 1 : 0;
        if (stable_frames_ >= params_->STABLE_FRAMES_REQUIRED)
            dsr_create_room_and_reparent(res);
        else if (write_rt)
            dsr_update_pose(res);   // world->robot RT while waiting for stable room creation
    }
    else
    {
        if (write_rt)
            dsr_update_pose(res);   // robot->room RT (skipped when the odometry publisher owns it)
        if (params_->PUBLISH_AFFORDANCE)
            dsr_update_affordance(res); // publish epistemic target affordance (off ⇒ room never competes)
    }
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_update_pose(const rc::RoomConcept::UpdateResult& res)
{
    write_robot_room_rt(res.robot_pose, res.covariance, static_cast<std::uint64_t>(res.timestamp_ms));
}

// High-rate predicted pose (dead-reckoned between lidar corrections): same RT edge, same frame/cov
// conversion as a corrected pose — only the source and the (grown) covariance + validity stamp differ.
void RoomSceneGraph::dsr_publish_predicted_pose(const Eigen::Affine2f& robot_pose,
                                                const Eigen::Matrix3f& covariance,
                                                std::uint64_t timestamp_ms)
{
    write_robot_room_rt(robot_pose, covariance, timestamp_ms);
}

// Shared writer for the robot↔room RT edge. `robot_pose`/`covariance` are the room←robot estimate
// (room frame); when the room is a child of the robot we invert to robot→room (pose AND cov Jacobian)
// before the timestamped ring-buffer write. Called by both the corrected and predicted paths.
void RoomSceneGraph::write_robot_room_rt(const Eigen::Affine2f& robot_pose,
                                         const Eigen::Matrix3f& covariance,
                                         std::uint64_t timestamp_ms)
{
    if (!G_ || !rt_api_) return;

    const auto describe_node = [this](uint64_t id) -> QString
    {
        if (id == 0)
            return "0:<unset>";
        if (const auto node = G_->get_node(id); node.has_value())
            return QString::number(id) + ":" + QString::fromStdString(node->name());
        return QString::number(id) + ":<missing>";
    };

    const auto log_cached_ids = [this, &describe_node](const char *reason, uint64_t missing_child_id)
    {
        const char *missing_child_kind = "other";
        if (missing_child_id == dsr_room_id_)
            missing_child_kind = "room";
        else if (missing_child_id == dsr_robot_id_)
            missing_child_kind = "robot";
        else if (missing_child_id == dsr_world_id_)
            missing_child_kind = "world";

        qWarning() << "dsr_update_pose:" << reason
                   << "missing_child_kind=" << missing_child_kind
                   << "missing_child_id=" << missing_child_id
                   << "world=" << describe_node(dsr_world_id_)
                   << "robot=" << describe_node(dsr_robot_id_)
                   << "room=" << describe_node(dsr_room_id_);
    };

    if ((dsr_world_id_ == 0 || !G_->get_node(dsr_world_id_).has_value()) ||
        (dsr_robot_id_ == 0 || !G_->get_node(dsr_robot_id_).has_value()))
    {
        check_init_graph_is_valid();
    }

    if (room_node_created_ && !G_->get_node(dsr_room_id_).has_value())
    {
        qWarning() << "dsr_update_pose: cached room node missing, resetting room state"
                   << "room_id=" << dsr_room_id_;
        log_cached_ids("cached room node missing", dsr_room_id_);
        room_node_created_ = false;
        dsr_room_id_ = 0;
        affordance_manager_.reset();
        stable_frames_ = 0;
    }

    const Eigen::Matrix2f R = robot_pose.linear();
    const Eigen::Vector2f t = robot_pose.translation();
    const float theta_room_to_robot = std::atan2(R(1, 0), R(0, 0));

    // Convert room->robot estimate into robot->room when the room is a child of the robot.
    const Eigen::Vector2f t_robot_to_room = -(R.transpose() * t);
    const float theta_robot_to_room = -theta_room_to_robot;

    const uint64_t parent_id = room_node_created_ ? dsr_robot_id_ : dsr_world_id_;
    const uint64_t child_id  = room_node_created_ ? dsr_room_id_  : dsr_robot_id_;

    auto parent_opt = G_->get_node(parent_id);
    if (!parent_opt.has_value()) return;

    if (!G_->get_node(child_id).has_value())
    {
        qWarning() << "dsr_update_pose: destination node missing, skipping RT update"
                   << "parent_id=" << parent_id
                   << "child_id=" << child_id
                   << "room_node_created=" << room_node_created_;
        log_cached_ids("destination node missing", child_id);
        if (room_node_created_ && child_id == dsr_room_id_)
        {
            room_node_created_ = false;
            dsr_room_id_ = 0;
            affordance_manager_.reset();
            stable_frames_ = 0;
        }
        return;
    }

    const float x     = room_node_created_ ? t_robot_to_room.x() : t.x();
    const float y     = room_node_created_ ? t_robot_to_room.y() : t.y();
    const float theta = room_node_created_ ? theta_robot_to_room : theta_room_to_robot;

    // ── Covariance (SE2 3×3 packed into 6×6 flat row-major) ───────────────
    Eigen::Matrix3f cov_se2 = Eigen::Matrix3f::Identity();
    if (covariance.rows() >= 3 && covariance.cols() >= 3)
        cov_se2 = covariance.topLeftCorner<3, 3>();

    if (room_node_created_)
    {
        const float c = std::cos(theta_room_to_robot);
        const float s = std::sin(theta_room_to_robot);
        Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
        J(0, 0) = -c;  J(0, 1) = -s;  J(0, 2) =  s * t.x() - c * t.y();
        J(1, 0) =  s;  J(1, 1) = -c;  J(1, 2) =  c * t.x() + s * t.y();
        J(2, 2) = -1.f;
        cov_se2 = J * cov_se2 * J.transpose();
    }

    std::vector<float> cov_flat(36, 0.f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cov_flat[r * 6 + c] = cov_se2(r, c);

    // ── Timestamped RT write ────────────────────────────────────────────────
    // Write via the RT_API covariance+timestamp overload so the edge keeps a proper TIMESTAMPED
    // HISTORY ring buffer (rt_timestamps / rt_head_index). The previous code overwrote a single
    // untimestamped block, so consumers' InterpolatedRT could never interpolate to a requested
    // time — it always returned the latest pose (RTdelta=0), freezing the controller's lidar
    // overlay between the ~5 Hz pose publishes. res.timestamp_ms is the pose's validity time (the
    // localization stamp). DSR interpolates pose between blocks; it does NOT extrapolate the velocity
    // attrs (written below) — those are for consumers to read directly.
    try
    {
        rt_api_->insert_or_assign_edge_RT(parent_opt.value(), child_id,
                                          std::vector<float>{x, y, 0.f},
                                          std::vector<float>{0.f, 0.f, theta},
                                          cov_flat,
                                          timestamp_ms);
    }
    catch (const std::exception &error)
    {
        qWarning() << "dsr_update_pose: insert_or_assign_edge_RT failed:" << error.what();
        log_cached_ids("insert_or_assign_edge_RT failed", child_id);
        if (room_node_created_ && child_id == dsr_room_id_)
        {
            room_node_created_ = false;
            dsr_room_id_ = 0;
            affordance_manager_.reset();
            stable_frames_ = 0;
        }
        return;
    }

    // Publish the robot's BODY-FRAME twist + covariance on the SAME RT edge, so the controller reads
    // velocity DIRECTLY (no pose differentiation → no correction-induced velocity spikes). Convention:
    // rt_translation_velocity = [adv(fwd,+x), side(lat,+y), 0], rt_rotation_euler_xyz_velocity = [0,0,rot].
    // last_adv_/side_/rot_ are the measured odometry (set in update()). Covariance = diagonal (config).
    // Re-fetch the edge FRESH from the graph (NOT from the stale parent_opt node captured before the RT
    // write) — otherwise we'd write back the old pose ring buffer and freeze the position.
    if (auto edge = G_->get_edge(parent_opt.value().id(), child_id, "RT"); edge.has_value())
    {
        G_->add_or_modify_attrib_local<rt_translation_velocity_att>(
            edge.value(), std::vector<float>{last_adv_, last_side_, 0.f});
        G_->add_or_modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(
            edge.value(), std::vector<float>{0.f, 0.f, last_rot_});
        std::vector<float> vel_cov(36, 0.f);   // 6×6 row-major; SE2 diag at [0],[7],[14] (matches pose cov)
        if (params_)
        {
            vel_cov[0]  = params_->ROBOT_VEL_COV_ADV;
            vel_cov[7]  = params_->ROBOT_VEL_COV_SIDE;
            vel_cov[14] = params_->ROBOT_VEL_COV_ROT;
        }
        G_->add_or_modify_attrib_local<rt_se2_covariance_velocity_att>(edge.value(), vel_cov);
        G_->insert_or_assign_edge(edge.value());
    }
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res)
{
    if (!G_) return;

    const auto room_polygon = room_concept_->nominal_room_polygon();
    std::vector<float> polygon_x;
    std::vector<float> polygon_y;
    polygon_x.reserve(room_polygon.size());
    polygon_y.reserve(room_polygon.size());
    for (const auto& vertex : room_polygon)
    {
        polygon_x.push_back(vertex.x());
        polygon_y.push_back(vertex.y());
    }

    if (const auto room_nodes = G_->get_nodes_by_type("room"); !room_nodes.empty())
    {
        dsr_room_id_ = room_nodes.front().id();
        room_node_created_ = true;
        stable_frames_ = 0;
        dsr_update_pose(res);
        return;
    }

    DSR::Node room_node = DSR::Node::create<room_node_type>("room");
    room_node.attrs()[delimiting_polygon_x_str.data()] = DSR::Attribute{polygon_x, 0, 0};
    room_node.attrs()[delimiting_polygon_y_str.data()] = DSR::Attribute{polygon_y, 0, 0};
    room_node.attrs()[room_height_str.data()] = DSR::Attribute{params_->room_height, 0, 0};
    // TODO: change to add_or_modify_attrib_local once available

    const auto room_id_opt = G_->insert_node(room_node);
    if (!room_id_opt.has_value())
    {
        qWarning() << "DSR: failed to create room node";
        return;
    }

    dsr_room_id_ = room_id_opt.value();
    room_node_created_ = true;
    stable_frames_ = 0;
    trigger_layout_();

    dsr_update_pose(res);
    dsr_create_wall_nodes();

    // Seed the epistemic planner with room geometry so it can generate candidates.
    if (!room_polygon.empty() && epistemic_)
    {
        Eigen::Vector2f pmin = room_polygon.front();
        Eigen::Vector2f pmax = room_polygon.front();
        for (const auto& v : room_polygon)
        {
            pmin = pmin.cwiseMin(v);
            pmax = pmax.cwiseMax(v);
        }
        epistemic_->set_room_bounds(pmin, pmax);
        epistemic_->set_room_polygon(room_polygon);
    }
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_update_affordance(const rc::RoomConcept::UpdateResult& res)
{
    if (!G_ || !room_node_created_ || !epistemic_) return;

    auto& planner = epistemic_->epistemic_planner();

    // Always update robot state so mark_and_refresh uses the correct position.
    epistemic_->set_robot_state(res.robot_pose, res.covariance);

    if (affordance_manager_.consume_completion_event())
    {
        planner.clear_target();
        planner.mark_and_refresh();   // keep path trail live in viewer
        planner.refresh_belief();     // just finished exploring → belief fresh (restart forget clock)
        return;
    }

    if (affordance_manager_.is_executing(G_))
    {
        planner.mark_and_refresh();   // stamp path + refresh IoR overlay during navigation
        planner.refresh_belief();     // actively exploring → hold belief fresh; forgetting only when idle
        return;
    }

    // Refresh obstacle exclusion zones from DSR graph before selecting the target.
    update_planner_obstacle_footprints();

    // Ask the planner for the current best target (handles dwell / arrival internally)
    const auto target_opt = planner.update_target();

    if (!target_opt.has_value()) return;

    const float tx   = target_opt->position.x();
    const float ty   = target_opt->position.y();
    // Advertise the GROUNDED epistemic value in nats: the raw FIM D-optimality gain
    // ΔH = ½·log det(I + Y_prior⁻¹·I_pred) = expected pose-entropy reduction from observing
    // corners/walls at this viewpoint. NOT target.score (which folds in IoR-staleness × path
    // heuristics used only for the room's own target ranking). This puts afford_room's gain in the
    // same currency as afford_table's ΔH so the controller can compare them as one EFE term.
    //
    // Recompute LIVE against the current pose precision (do not reuse the value frozen at target
    // selection): as the robot drives in and localization tightens, Y_prior grows ⇒ the pose-FIM
    // part decays toward 0. We publish the TOTAL gain (pose-FIM ΔH + IoR patrol-staleness drive):
    // the IoR term is NON-saturating, so once the pose is localized the advertised gain stays
    // positive on long-unvisited cells instead of falling through the consumer's withdrawal
    // threshold and stopping exploration after a single affordance.
    // The recompute at target_opt->position also gives the rotate-in-place recovery a real gain.
    const float gain = planner.live_total_epistemic_gain(target_opt->position);

    // Heading: face toward room centre so the robot maximises wall/corner visibility
    const float cx  = (planner.room_min().x() + planner.room_max().x()) * 0.5f;
    const float cy  = (planner.room_min().y() + planner.room_max().y()) * 0.5f;
    const float yaw = std::atan2(cy - ty, cx - tx);

    affordance_manager_.publish_target(
        G_,
        dsr_room_id_,
        tx,
        ty,
        yaw,
        gain,
        [this]() { trigger_layout_(); },
        [this]() { trigger_layout_(); });
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::update_planner_obstacle_footprints()
{
    if (!G_ || !rt_api_ || !room_node_created_ || !epistemic_) return;

    std::vector<rc::EpistemicPlanner::ObstacleFootprint> footprints;

    auto collect = [&](const std::string& node_type)
    {
        for (const auto& node : G_->get_nodes_by_type(node_type))
        {
            const auto w_opt = G_->get_attrib_by_name<width_m_att>(node);
            const auto d_opt = G_->get_attrib_by_name<depth_m_att>(node);
            if (!w_opt.has_value() || !d_opt.has_value()) continue;
            const float half_w = w_opt.value() * 0.5f;
            const float half_d = d_opt.value() * 0.5f;
            if (half_w <= 0.f || half_d <= 0.f) continue;

            const auto rt_opt = rt_api_->get_RT_pose_from_parent(node);
            if (!rt_opt.has_value()) continue;

            const Eigen::Vector3d t = rt_opt->translation();
            const float yaw = static_cast<float>(
                std::atan2(rt_opt->linear()(1, 0), rt_opt->linear()(0, 0)));

            footprints.push_back({
                .center = {static_cast<float>(t.x()), static_cast<float>(t.y())},
                .half_w = half_w,
                .half_d = half_d,
                .yaw    = yaw
            });
        }
    };

    collect("object");
    collect("obstacle");
    collect("table");   // tables (table_concept) are type "table", not "object" — avoid them too

    epistemic_->epistemic_planner().set_obstacle_footprints(std::move(footprints));
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::load_robot_body_dimensions_from_graph()
{
    if (!G_)
        return;

    std::optional<DSR::Node> body_node = std::nullopt;

    const auto body_nodes = G_->get_nodes_by_type("body");
    if (!body_nodes.empty())
        body_node = body_nodes.front();
    else
        body_node = G_->get_node("body");

    if (!body_node.has_value())
    {
        qWarning() << "dsr_init_graph: no 'body' node found; keeping default robot dimensions"
                   << params_->ROBOT_WIDTH << params_->ROBOT_LENGTH << params_->ROBOT_HEIGHT;
        return;
    }

    dsr_body_id_ = body_node->id();

    if (const auto width_value = G_->get_attrib_by_name<width_m_att>(body_node.value()); width_value.has_value())
        params_->ROBOT_WIDTH = width_value.value();
    if (const auto depth_value = G_->get_attrib_by_name<depth_m_att>(body_node.value()); depth_value.has_value())
        params_->ROBOT_LENGTH = depth_value.value();
    if (const auto height_value = G_->get_attrib_by_name<height_m_att>(body_node.value()); height_value.has_value())
        params_->ROBOT_HEIGHT = height_value.value();

    if (epistemic_)
        epistemic_->set_robot_footprint(params_->ROBOT_WIDTH, params_->ROBOT_LENGTH);

    qInfo() << "Robot dimensions from body node: width depth height ="
            << params_->ROBOT_WIDTH << params_->ROBOT_LENGTH << params_->ROBOT_HEIGHT;
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_create_wall_nodes()
{
    if (!G_ || !rt_api_) return;

    auto room_node_opt = G_->get_node(dsr_room_id_);
    if (!room_node_opt.has_value()) { qWarning() << "dsr_create_wall_nodes: room node missing"; return; }

    // Guard — idempotent: if wall nodes already exist under this room, skip.
    if (!G_->get_nodes_by_type("wall").empty())
        return;

    const auto polygon = room_concept_->nominal_room_polygon();
    const int n = static_cast<int>(polygon.size());
    if (n < 3) { qWarning() << "dsr_create_wall_nodes: polygon has fewer than 3 vertices"; return; }

    const float half_h = params_->room_height * 0.5f;

    // ── Walls ────────────────────────────────────────────────────────────────
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& p0 = polygon[i];
        const Eigen::Vector2f& p1 = polygon[(i + 1) % n];
        const float L = (p1 - p0).norm();
        if (L < 0.1f)
        {
            qWarning() << "dsr_create_wall_nodes: skipping degenerate wall" << i << "(length" << L << "m)";
            continue;
        }

        const Eigen::Vector2f dir = (p1 - p0) / L;
        const float yaw = std::atan2(dir.y(), dir.x());
        const Eigen::Vector2f mid = (p0 + p1) * 0.5f;

        DSR::Node wall_node = DSR::Node::create<wall_node_type>("wall_" + std::to_string(i));
        G_->add_or_modify_attrib_local<width_m_att>(wall_node, L);
        G_->add_or_modify_attrib_local<height_m_att>(wall_node, params_->room_height);
        G_->add_or_modify_attrib_local<parent_att>(wall_node, dsr_room_id_);
        G_->add_or_modify_attrib_local<level_att>(wall_node, 4);

        const auto wall_id = G_->insert_node(wall_node);
        if (!wall_id.has_value())
        {
            qWarning() << "dsr_create_wall_nodes: failed to insert wall_" + QString::number(i);
            continue;
        }

        rt_api_->insert_or_assign_edge_RT(room_node_opt.value(),
                                          wall_id.value(),
                                          {mid.x(), mid.y(), half_h},
                                          {0.f, 0.f, yaw});
    }

    // ── Floor ─────────────────────────────────────────────────────────────────
    // Purely semantic parent for floor-attached objects; placed at room origin.
    DSR::Node floor_node = DSR::Node::create<floor_node_type>("floor");
    G_->add_or_modify_attrib_local<parent_att>(floor_node, dsr_room_id_);
    G_->add_or_modify_attrib_local<level_att>(floor_node, 4);

    const auto floor_id = G_->insert_node(floor_node);
    if (!floor_id.has_value())
        qWarning() << "dsr_create_wall_nodes: failed to insert floor node";
    else
        rt_api_->insert_or_assign_edge_RT(room_node_opt.value(),
                                          floor_id.value(),
                                          {0.f, 0.f, 0.f},
                                          {0.f, 0.f, 0.f});

    trigger_layout_();
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::cleanup_room_graph_nodes()
{
    if (!G_) return;
    // Delete affordance nodes hanging from room via "has_intention" edges,
    // then delete the room nodes themselves.
    for (const auto& room_node : G_->get_nodes_by_type("room"))
    {
        for (const auto& edge : G_->get_node_edges_by_type(room_node, "has_intention"))
            if (G_->get_node(edge.to()).has_value())
                G_->delete_node(edge.to());
        G_->delete_node(room_node);
    }
    // Delete wall and floor nodes owned by this agent.
    for (const auto& n : G_->get_nodes_by_type("wall"))
        G_->delete_node(n);
    if (auto n = G_->get_node("floor"); n.has_value())
        G_->delete_node(n.value());
    // Fallback: delete the affordance node by its known name in case it is orphaned.
    if (auto n = G_->get_node("afford_room"); n.has_value())
        G_->delete_node(n.value());
    room_node_created_ = false;
    dsr_room_id_ = 0;
    affordance_manager_.reset();
    stable_frames_ = 0;
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::check_init_graph_is_valid()
{
    if (!G_) { qWarning() << "dsr_init_graph: DSR graph not available"; return; }

    // Resolve the root/world node by type (name may vary, e.g. "root", "world")
    const auto root_nodes = G_->get_nodes_by_type("root");
    if (!root_nodes.empty())
        dsr_world_id_ = root_nodes.front().id();
    else { qWarning() << "dsr_init_graph: no 'root' type node found in graph"; return; }

    // Resolve the robot node by type
    const auto robot_nodes = G_->get_nodes_by_type("robot");
    if (!robot_nodes.empty())
        dsr_robot_id_ = robot_nodes.front().id();
    else { qWarning() << "dsr_init_graph: no 'robot' type node found in graph"; return; }

    load_robot_body_dimensions_from_graph();
}

}  // namespace rc
