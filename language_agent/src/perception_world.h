#pragma once
/*
 * perception_world.h — fills a SeeScene (can_see.h) from the graph, on the MAIN thread, for the can_see tool.
 *
 *   room polygon     delimiting_polygon_x/_y of the current room node (a proto-room's polygon is fine)
 *   doors            centre via room->door, width_m, door_phi_rad, door_open_prob (the gap is cut in can_see.h)
 *   objects          the digest's current-room objects: pose (incl. its z) via room->object, sizes and shape from the digest
 *   ZED              cam_rgb_width/height/focalx/focaly of the `zed` node (the field of view), mount height via
 *                    room->zed
 *   helios / bpearl  rc::media::sensor_model_from_graph + the mount from the RT chain room->sensor
 * A missing model or transform leaves that channel ABSENT: the evaluator then reports it NotModelled (it is no look,
 * never a zero look). Every get_transformation_matrix optional is checked (CLAUDE.md).
 */

#include <cmath>
#include <limits>
#include <numbers>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/media_transport/media_transport.h"
#include "approach.h"
#include "can_see.h"
#include "world_digest.h"

namespace rc::chat
{

namespace see_detail
{
/// body <- sensor: the mount in the ROBOT BASE frame (`body`, as canvas_lidar.cpp), NOT in the room: the evaluator
/// applies robot(vp) * mount itself, and a room-frame mount would carry the robot's pose a second time.
inline std::optional<Eigen::Isometry3f> mount_of(DSR::InnerEigenAPI& inner, const std::string& robot_frame, const std::string& node)
{
    const auto T = inner.get_transformation_matrix(robot_frame, node);
    if (not T.has_value()) return std::nullopt;
    Eigen::Isometry3f m = Eigen::Isometry3f::Identity();
    m.matrix() = T->matrix().cast<float>();
    return m;
}
}   // namespace see_detail

/// The sensors of the scene: the ZED (field of view from the intrinsics, mount height above the robot's floor node) and the two LiDARs
/// (body <- sensor). The imagined thing in free space in front of the robot still needs its camera, room or not.
inline void fill_sensors(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner, SeeScene& s, const std::string& robot_frame)
{
    // The ZED's height is read from the robot-type node (Shadow, at the floor), the same on every path whether or not a room
    // is known; room frames sit at z = 0, so it equals the room-relative height. No robot node, or no chain to the camera:
    // the camera is left out and the answer says so (never silently dropped).
    const auto robots = G.get_nodes_by_type("robot");
    const std::string ref_frame = robots.empty() ? std::string{} : robots.front().name();
    // ZED: the field of view from the intrinsics (pinhole), the mount height from the RT chain.
    if (const auto zn = G.get_node("zed"); zn.has_value())
    {
        const auto w = G.get_attrib_by_name<cam_rgb_width_att>(zn.value());
        const auto h = G.get_attrib_by_name<cam_rgb_height_att>(zn.value());
        const auto fx = G.get_attrib_by_name<cam_rgb_focalx_att>(zn.value());
        const auto fy = G.get_attrib_by_name<cam_rgb_focaly_att>(zn.value());
        const auto Tz = ref_frame.empty() ? decltype(inner.get_transformation_matrix(ref_frame, "zed")){}
                                          : inner.get_transformation_matrix(ref_frame, "zed");
        if (w and h and fx and fy and Tz.has_value() and *w > 0 and *h > 0 and *fx > 0 and *fy > 0)
        {
            rc::nbv::Sensor z;
            z.hfov_rad = 2.0f * std::atan(static_cast<float>(*w) / (2.0f * static_cast<float>(*fx)));
            z.vfov_rad = 2.0f * std::atan(static_cast<float>(*h) / (2.0f * static_cast<float>(*fy)));
            z.height_m = static_cast<float>(Tz->translation().z());
            s.zed = z;
            s.zed_mount_z = z.height_m;
        }
        else s.notes.push_back("I could not read my camera's mount or intrinsics, so I left it out");
    }

    // LiDARs: model from the media descriptor, mount from the RT chain; either missing ⇒ the channel is absent.
    const auto rig = [&](const char* name) -> std::optional<rc::card::LidarRig>
    {
        const auto m = rc::media::sensor_model_from_graph(G, name);
        const auto mt = see_detail::mount_of(inner, robot_frame, name);
        if (not m.has_value() or not mt.has_value()) return std::nullopt;
        return lidar_rig_from_body_mount(*m, *mt);
    };
    s.helios = rig("helios");
    s.bpearl = rig("bpearl");
}

[[nodiscard]] inline SeeScene build_see_scene(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner, const Digest& digest,
                                              const std::string& robot_frame = "body")
{
    SeeScene s;
    s.room = digest.facts.room;
    if (s.room.empty() or digest.room_id == 0) { s.room.clear(); fill_sensors(G, inner, s, robot_frame); return s; }

    if (const auto rn = G.get_node(digest.room_id); rn.has_value())
    {
        const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(rn.value());
        const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(rn.value());
        if (px.has_value() and py.has_value())
        {
            const auto& xs = px.value().get();
            const auto& ys = py.value().get();
            for (std::size_t i = 0; i < std::min(xs.size(), ys.size()); ++i) s.polygon.emplace_back(xs[i], ys[i]);
        }
        // The wall height the renderers use (imagine_view); absent or implausible = the 2.5 m default, flagged as assumed.
        if (const auto rh = G.get_attrib_by_name<room_height_att>(rn.value()); rh.has_value() and std::isfinite(*rh) and *rh > 0.0f)
        {
            s.room_height = *rh;
            s.room_height_known = true;
        }
    }

    // robot pose: room <- body, the SAME base frame the LiDAR mounts below are expressed in (body <- sensor).
    // robot_yaw is the body x axis' bearing; the robot FACES body +y (ROBOT_GEOMETRY.md), so its heading is
    // robot_yaw + pi/2 (can_see.h resolve_from).
    if (const auto T = inner.get_transformation_matrix(s.room, robot_frame); T.has_value())
    {
        s.robot_xy = Eigen::Vector2f(static_cast<float>(T->translation().x()), static_cast<float>(T->translation().y()));
        s.robot_yaw = static_cast<float>(std::atan2(T->matrix()(1, 0), T->matrix()(0, 0)));
    }

    for (const auto& name : digest.facts.doors)
    {
        const auto T = inner.get_transformation_matrix(s.room, name);
        const auto dn = G.get_node(name);
        if (not T.has_value() or not dn.has_value()) continue;
        SeeDoor d;
        d.name = name;
        d.xy = {static_cast<float>(T->translation().x()), static_cast<float>(T->translation().y())};
        d.width_m = G.get_attrib_by_name<width_m_att>(dn.value()).value_or(0.0f);
        d.phi_rad = G.get_attrib_by_name<door_phi_rad_att>(dn.value());
        d.p_open = G.get_attrib_by_name<door_open_prob_att>(dn.value());
        s.doors.push_back(std::move(d));
    }

    for (const auto& o : digest.objects)
    {
        if (not o.in_current_room) continue;
        const auto T = inner.get_transformation_matrix(s.room, o.name);
        if (not T.has_value()) continue;
        SeeObject so;
        so.name = o.name;
        so.subtype = o.subtype;
        so.x = static_cast<float>(T->translation().x());
        so.y = static_cast<float>(T->translation().y());
        so.yaw = static_cast<float>(std::atan2(T->matrix()(1, 0), T->matrix()(0, 0)));
        // Its real height: the RT z is the node origin, the BASE for every concept but the bottle, whose origin is its
        // centre (bottle_scene_graph.cpp:293; can_see.h object_base_z).
        so.z_origin = static_cast<float>(T->translation().z());
        so.origin_is_centre = o.subtype == "bottle";
        so.shape = o.shape;
        so.w = o.width_m; so.d = o.depth_m; so.h = o.height_m;
        // The standoff: its approach affordance's side nearest the robot (room frame), else the robot-side point at the
        // collision floor + half the footprint, on the line from the object to the robot.
        if (const auto aff = G.get_node("aff_" + o.name + "_approach"); aff.has_value())
        {
            const auto names = G.get_attrib_by_name<aff_option_names_att>(*aff);
            const auto xs = G.get_attrib_by_name<aff_option_x_m_att>(*aff);
            const auto ys = G.get_attrib_by_name<aff_option_y_m_att>(*aff);
            const auto ws = G.get_attrib_by_name<aff_option_yaw_rad_att>(*aff);
            if (names.has_value() and xs.has_value() and ys.has_value() and ws.has_value())
                if (const auto sides = parse_sides(*names, xs->get(), ys->get(), ws->get()); sides.has_value())
                {
                    float best = std::numeric_limits<float>::max();
                    for (const auto& sd : *sides)
                    {
                        const Eigen::Vector2f p(sd.x, sd.y);
                        const float dist = s.robot_xy ? (p - *s.robot_xy).norm() : 0.0f;
                        if (dist < best) { best = dist; so.standoff = p; }
                    }
                }
        }
        if (not so.standoff.has_value() and s.robot_xy.has_value())
        {
            // body clearance 0.30 m = kRobotRadiusM in rc::card::where_to_stand (evaluator.h), plus half the footprint,
            // plus a viewing margin.
            constexpr float kRobotRadiusM = 0.30f, kViewMarginM = 0.5f;
            const Eigen::Vector2f c(so.x, so.y);
            const Eigen::Vector2f dir = (*s.robot_xy - c);
            if (dir.norm() > 1e-3f)
                so.standoff = c + dir.normalized() * (kRobotRadiusM + 0.5f * std::max(so.w.value_or(0.f), so.d.value_or(0.f)) + kViewMarginM);
        }
        s.objects.push_back(std::move(so));
    }

    fill_sensors(G, inner, s, robot_frame);
    return s;
}

}   // namespace rc::chat
