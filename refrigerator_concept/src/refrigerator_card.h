// refrigerator_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only refrigerator_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what the agent's existence P(detect) really multiplies, read off
// refrigerator_existence.cpp. Contract: common/perception_card/card.h.
#pragma once

#include "refrigerator_config.h"
#include "../../common/perception_card/card.h"

namespace rc
{
inline card::Card make_card(const RefrigeratorConfig& cfg)
{
    card::Card c;
    c.concept_ = "refrigerator";
    c.graph_type = "object";
    c.level = card::Level::Countable;
    // ricoh: peripheral attention only (never births, never fits, never charges absence).
    c.channels = {{"zed", card::Role::Identity, true},
                  {"helios", card::Role::Occupancy, true},
                  {"ricoh", card::Role::Identity, false}};
    // etc/config.toml AI2PriorFootprintM / AI2PriorHeightM (= manifest prior.footprint / prior.height): floor-anchored [0, H].
    c.geometry = {cfg.ai2_prior_footprint_m, cfg.ai2_prior_footprint_m, cfg.ai2_prior_height_m, 0.0f, cfg.ai2_prior_height_m};
    // Level 1: refrigerator_projection.cpp's walls_block(cam, sample, polygon, kRefrigeratorOwnWallSkipM): LINE walls
    // (thickness 0), own-wall radius 0.70 m.
    c.walls = card::WallOcclusion{kRefrigeratorOwnWallSkipM, -1.0f, 0.0f};
    // refrigerator_existence.cpp:170: p_detect = envelope(sil.fill, visible = in_fov_frac) * central_frac.
    // The comment there names range_conf, but the code no longer multiplies it, so it is not listed.
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    sp.q_return = cfg.existence_detection_prob;
    sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}, {"axes", 1.0f}}},
                {"central", {}}};
    // axes = 1: the agent calls the single-fill p_detect(sil.fill = max axis, in_fov, env), with no sigma marginalisation.
    c.subtypes[""] = sp;
    c.blind_spots = {
        "Nose-to-nose (fill above about 1.3 of the frame) the mask truncates; the envelope shoulder is gradual (soft 0.26).",
        "The lower shoulder is NOT measured: the tour never went far enough for the fridge to become too small to segment.",
        "Edge-on views are detectable up close, so obliquity is deliberately not a term.",
        "The back face rests against a wall and is never scored; the 180 degree front/back ambiguity is resolved only by the clearance prior and the door seam."};
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = sp.env.min_fill != fleet.min_fill or sp.env.max_fill != fleet.max_fill or sp.env.soft != fleet.soft;
    c.calibration = fitted ? "fitted" : "prior";
    // etc/config.toml: "CALIBRATED 2026-08-08 from a deliberate approach/retreat tour".
    if (fitted) c.calibration_date = "2026-08-08";
    return c;
}
}  // namespace rc
