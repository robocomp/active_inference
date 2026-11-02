// hood_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only hood_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what the agent's existence P(detect) really multiplies, read off
// hood_existence.cpp. Contract: common/perception_card/card.h.
#pragma once

#include "hood_config.h"
#include "../../common/perception_card/card.h"

namespace rc
{
inline card::Card make_card(const HoodConfig& cfg)
{
    card::Card c;
    c.concept_ = "hood";
    c.graph_type = "object";
    c.level = card::Level::Countable;
    c.channels = {{"zed", card::Role::Identity, true},
                  {"helios", card::Role::Occupancy, true},
                  {"ricoh", card::Role::Identity, false}};
    // etc/config.toml AI2Prior* (manifest hood.concept.toml): a HANGING box, span [H - extent, H].
    const float depth = cfg.ai2_prior_depth_m < 0.0f ? cfg.ai2_prior_footprint_m : cfg.ai2_prior_depth_m;
    c.geometry = {cfg.ai2_prior_footprint_m, depth, cfg.vertical_extent_m,
                  cfg.ai2_prior_height_m - cfg.vertical_extent_m, cfg.ai2_prior_height_m};
    // Level 1: hood_projection.cpp's walls_block(cam, sample, polygon, kHoodOwnWallSkipM): LINE walls, own-wall radius 0.70 m.
    c.walls = card::WallOcclusion{kHoodOwnWallSkipM, -1.0f, 0.0f};
    // hood_existence.cpp:185: envelope(sil.fill, visible = in_fov_frac) * central_frac. The comment there
    // names range_conf but the code does not multiply it (same as the refrigerator), so it is not listed.
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    sp.q_return = cfg.existence_detection_prob;
    sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}, {"axes", 1.0f}}},
                {"central", {}}};
    // axes = 1: the agent calls the single-fill p_detect(sil.fill = max axis, in_fov, env), with no sigma marginalisation.
    c.subtypes[""] = sp;
    c.blind_spots = {
        "The envelope is the fleet prior, never fitted on a hood: p_detect is a class-level guess.",
        "A thin canopy 2 m up is grazed by few near-horizontal LiDAR rings; the measured return rate is low, so LiDAR absence is weak.",
        "The top merges into the wall or leaves the frame at usable stand-offs, so the vertical extent is a parameter, not a DOF.",
        "Edge-on views are detectable up close, so obliquity is deliberately not a term."};
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = sp.env.min_fill != fleet.min_fill or sp.env.max_fill != fleet.max_fill or sp.env.soft != fleet.soft;
    c.calibration = fitted ? "fitted" : "prior";
    return c;
}
}  // namespace rc
