#pragma once
/*
 * controller_approach_view.h — what the 2-D canvas shows of an OBJECT-APPROACH affordance (2026-11-11, owner request).
 *
 * The approach node (common/pragmatic_affordance/object_approach.cpp) lists one stand-off pose per side it offers:
 * aff_option_names (comma-separated, the default first) and aff_option_x_m / _y_m / _yaw_rad (room frame; yaw = the
 * FORWARD heading, +x = 0, CCW — object_approach_geometry.h standpoint()). The consumer's claim names the side it
 * executes (executing_option on the `executing` edge into the node). This reads all of that, plus the parent object's
 * centre / yaw / footprint, into plain values for Viewer2D::draw_approach — so a stand-off that is too short, or a heading
 * that does not point at the object, is visible at a glance.
 *
 * Display only: called on the control thread at the canvas's ~5 Hz, type-attributed DSR access only, every optional
 * checked. Header-only.
 */

#include <cmath>
#include <optional>
#include <string>
#include <string_view>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "viewer_2d.h"

namespace rc::approach_view
{

/// nullopt when `node_name` is not an approach node offering options. `required_option` = the side an order asked for
/// ("" = none); `working` = the controller's own step target (x, y, yaw) when it drives this node.
[[nodiscard]] inline std::optional<Viewer2D::ApproachView>
read(DSR::DSRGraph &G, DSR::InnerEigenAPI *ie, const std::string &room_name, const std::string &node_name,
     const std::string &required_option, const std::optional<Eigen::Vector3f> &working)
{
    if (node_name.empty()) return std::nullopt;
    const auto n = G.get_node(node_name);
    if (not n.has_value()) return std::nullopt;
    const auto names = G.get_attrib_by_name<aff_option_names_att>(n.value());
    const auto xs = G.get_attrib_by_name<aff_option_x_m_att>(n.value());
    const auto ys = G.get_attrib_by_name<aff_option_y_m_att>(n.value());
    const auto ws = G.get_attrib_by_name<aff_option_yaw_rad_att>(n.value());
    if (not names.has_value() or not xs.has_value() or not ys.has_value() or not ws.has_value()) return std::nullopt;
    const std::vector<float> &x = xs.value().get(), &y = ys.value().get(), &w = ws.value().get();

    std::string executing;
    for (const auto &e : G.get_edges_to_id(n->id()))
        if (e.type() == "executing")
        {
            executing = G.get_attrib_by_name<executing_option_att>(e).value_or(std::string{});
            break;
        }

    Viewer2D::ApproachView v;
    v.node = node_name;
    const std::string &s = names.value();
    std::size_t i = 0, a = 0;
    while (a <= s.size() and not s.empty())
    {
        const auto b = std::min(s.find(',', a), s.size());
        if (i < x.size() and i < y.size() and i < w.size() and std::isfinite(x[i]) and std::isfinite(y[i])
            and std::isfinite(w[i]))
            v.options.push_back({.name = s.substr(a, b - a), .p = Eigen::Vector2f(x[i], y[i]), .yaw = w[i]});
        a = b + 1;
        ++i;
    }
    if (v.options.empty()) return std::nullopt;
    // chosen: the side claimed, else the side ordered, else the default (names[0] == epistemic_target_*)
    const std::string &pick = not executing.empty() ? executing : required_option;
    bool picked = false;
    for (auto &o : v.options)
        if (not pick.empty() and o.name == pick) { o.chosen = true; o.executing = not executing.empty(); picked = true; }
    if (not picked) v.options.front().chosen = true;

    if (const auto parent = G.get_attrib_by_name<parent_att>(n.value()); parent.has_value())
        if (const auto pn = G.get_node(parent.value()); pn.has_value())
        {
            if (ie != nullptr)
                if (const auto T = ie->get_transformation_matrix(room_name, pn->name()); T.has_value())
                {
                    const auto &m = T->matrix();
                    v.obj_centre = Eigen::Vector2f(static_cast<float>(m(0, 3)), static_cast<float>(m(1, 3)));
                    v.obj_yaw = static_cast<float>(std::atan2(m(1, 0), m(0, 0)));
                }
            const float wm = G.get_attrib_by_name<width_m_att>(pn.value()).value_or(0.f);
            const float dm = G.get_attrib_by_name<depth_m_att>(pn.value()).value_or(0.f);
            if (std::isfinite(wm) and std::isfinite(dm)) { v.obj_width_m = wm; v.obj_depth_m = dm; }
        }
    v.working_target = working;
    return v;
}

}   // namespace rc::approach_view
