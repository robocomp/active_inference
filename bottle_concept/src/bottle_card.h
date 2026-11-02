// bottle_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only bottle_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what the agent's existence P(detect) really multiplies, read off
// bottle_existence.cpp. Contract: common/perception_card/card.h.
#pragma once

#include "bottle_config.h"
#include "../../common/perception_card/card.h"

namespace rc
{
inline card::Card make_card(const BottleConfig& cfg)
{
    card::Card c;
    c.concept_ = "bottle";
    c.graph_type = "object";
    c.level = card::Level::Countable;
    // ricoh: no-depth bearing that HOLDS a live instance (never births, never charges absence).
    c.channels = {{"zed", card::Role::Identity, true},
                  {"helios", card::Role::Occupancy, true},
                  {"ricoh", card::Role::Identity, false}};
    // Cylinder: width = depth = 2 r, span [0, h] relative to the instance's resolved support (floor or a table).
    c.geometry = {2.0f * cfg.prior_radius, 2.0f * cfg.prior_radius, cfg.prior_height, 0.0f, cfg.prior_height};
    // Level 1: bottle_existence.cpp adds rc::nbv::wall_obstacles(polygon, 0.10) to the occluders: 0.10 m boxes, and no
    // own-wall rule at all (a bottle stands on a support, never in a wall).
    c.walls = card::WallOcclusion{0.0f, -1.0f, kBottleWallThicknessM};
    // bottle_existence.cpp:219-224: rc::nbv::expected_p_detect(fill_max, fill_min, f_sig, visible, env) -- the
    // two-axis envelope with the visible fraction, marginalised over the belief's own fill sigma (that
    // marginalisation is the evaluator's job, not a card term).
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    sp.q_return = cfg.existence_detection_prob;
    sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}, {"axes", 2.0f}}}};
    c.subtypes[""] = sp;
    c.blind_spots = {
        "A 6 cm object is where YOLO gives up first: a missing mask can be absence of the label, not of the bottle.",
        "The envelope is the fleet prior, uncalibrated for an object this small.",
        "A bottle standing inside its support table or occluded by clutter is only partly visible; absence is weighted by the visible fraction."};
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = sp.env.min_fill != fleet.min_fill or sp.env.max_fill != fleet.max_fill or sp.env.soft != fleet.soft;
    c.calibration = fitted ? "fitted" : "prior";
    return c;
}
}  // namespace rc
