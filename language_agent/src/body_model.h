#pragma once
/*
 * body_model.h — the robot body the imagine tool judges a doorway with: THE CONTROLLER'S body, not a second one.
 *   half width = max(support_radius along +x, along -x) of the footprint with the controller's safety margin applied,
 *                the exact expression controller_session.cpp uses for its door fit;
 *   footprint  = the robot node's mesh, cross-checked for Shadow against the compiled hull exactly as
 *                controller_robot_body.cpp does (an old wheel-less mesh would shrink the body 4 cm per side);
 *   margin     = Controller.FootprintSafetyMarginM read from the controller's own etc/config.toml with the same
 *                ConfigLoader every agent uses; when the file does not set it the controller keeps ControllerParams'
 *                default, so that default is read from the struct, never typed here. Unreadable file = no margin,
 *                and the answer says so;
 *   height     = the mesh's z-extent (bb_max.z in the robot frame).
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>

#include <ConfigLoader/ConfigLoader.h>

#include "../../common/footprint/robot_body_check.h"
#include "../../common/robot_footprint/robot_footprint.h"
#include "../../controller/src/controller_runtime_types.h"

namespace rc::chat
{

struct BodyModel
{
    float half_width = 0.f;
    std::optional<float> height_m;
    std::string note;   ///< plain words when the width is not the controller's full one
};

/// Controller.FootprintSafetyMarginM as the controller would run with: the file's value, else ControllerParams' own
/// default. nullopt = the controller's config file cannot be read.
[[nodiscard]] inline std::optional<float> controller_safety_margin(const std::filesystem::path& controller_config)
{
    std::error_code ec;
    if (not std::filesystem::exists(controller_config, ec)) return std::nullopt;
    try
    {
        ConfigLoader cl;
        cl.load(controller_config.string());
        constexpr const char* key = "Controller.FootprintSafetyMarginM";
        if (cl.exists(key)) return static_cast<float>(cl.get<double>(key));
        return ControllerParams{}.footprint_safety_margin_m;
    }
    catch (const std::exception&) { return std::nullopt; }
}

/// `mesh` = the robot node's mesh file, `is_shadow` = the robot is Shadow (the cross-check applies), `margin` as above.
[[nodiscard]] inline std::optional<BodyModel> load_body_model(const std::filesystem::path& mesh, bool is_shadow,
                                                              const std::optional<float>& margin)
{
    rc::RobotFootprint::MeshReport rep;
    auto fp = rc::RobotFootprint::from_obj(mesh.string(), 0.f, rep);
    if (not fp.has_value()) return std::nullopt;
    rc::RobotFootprint body = *fp;
    if (is_shadow)
    {
        if (rc::shadow_mesh_disagreement(rep, mesh.string()).has_value())
            body = rc::RobotFootprint::shadow();   // the controller's fall-back: the compiled hull
    }
    BodyModel m;
    if (margin.has_value()) body.set_safety_margin(*margin);
    else m.note = "without the controller's safety margin";
    m.half_width = std::max(body.support_radius(0.f, Eigen::Vector2f(1.f, 0.f)), body.support_radius(0.f, Eigen::Vector2f(-1.f, 0.f)));
    if (rep.bb_max.z() > 0.f) m.height_m = rep.bb_max.z();
    return m;
}

}   // namespace rc::chat
