// door_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only door_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what the agent's existence P(detect) really multiplies, read off
// core/door_existence.cpp ~1260-1279. Contract: common/perception_card/card.h.
#pragma once

#include "door_config.h"
#include "../../common/perception_card/card.h"

namespace rc
{
inline card::Card make_card(const DoorConfig& cfg)
{
    card::Card c;
    c.concept_ = "door";
    c.graph_type = "door";
    c.level = card::Level::Articulated;
    // ZED mask = identity. The LiDAR channel is the low bpearl dome, NOT helios: the agent's ingestor is built with
    // LidarGates{0, 0, lidar_bpearl_precision} (specificworker.cpp:564), so helios is never pumped.
    c.channels = {{"zed", card::Role::Identity, true},
                  {"bpearl", card::Role::Occupancy, true}};
    // The leaf the fit holds: width/height priors of etc/config.toml [DoorModel] (a 1.00 x 2.10 m Webots door, thickness fixed).
    c.geometry = {cfg.door_prior_w_m, cfg.door_thickness_m, cfg.door_prior_h_m, 0.0f, cfg.door_prior_h_m};
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    // q_return (per-beam LiDAR return probability) is left at its default 1.0: UNCALIBRATED for the door. The only door
    // number nearby, Existence.DetectionProb, is a camera mask-pixel probability, a different quantity.
    // door_existence.cpp:1279: p_detect = envelope(roi_fill, visible = 1) * leaf_obliquity * resolvability * in_fov *
    // central * kappa. The envelope is the single-fill overload (axes = 1). The agent passes visible = 1 and applies
    // in_fov as its own factor; the evaluator folds visibility into the envelope's visible argument and skips in_fov,
    // so the product is the same. phi_max = Openable.PhiMaxRad: the uniform prior on the hinge angle of an unmeasured door.
    sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}, {"axes", 1.0f}}},
                {"leaf_obliquity", {{"phi_max", cfg.openable_phi_max_rad}}},
                {"resolvability", {}},
                {"in_fov", {}},
                {"central", {}},
                {"kappa", {}}};
    c.subtypes[""] = sp;
    c.blind_spots = {
        "resolvability and kappa need a measurement (silhouette cells, pose covariance): a hypothetical door takes them as 1, so P is an upper bound.",
        "An edge-on (open) leaf presents almost no area to segment: obliquity cuts both ways, it can neither refute nor defend the door.",
        "The fill envelope's near shoulder is the fleet default, not fitted to doors (the door's removal path has no near shoulder of its own)."};
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = sp.env.min_fill != fleet.min_fill or sp.env.max_fill != fleet.max_fill or sp.env.soft != fleet.soft;
    c.calibration = fitted ? "fitted" : "prior";
    return c;
}
}  // namespace rc
