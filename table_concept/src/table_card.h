// table_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only table_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what the agent's existence P(detect) really multiplies, read off
// table_existence.cpp. Contract: common/perception_card/card.h.
#pragma once

#include "table_config.h"
#include "../../common/perception_card/card.h"

namespace rc
{
// bsm_samples = how many blind-spot-map samples are loaded (0 = none, the card is a pure prior).
inline card::Card make_card(const TableConfig& cfg, std::size_t bsm_samples = 0)
{
    card::Card c;
    c.concept_ = "table";
    c.graph_type = "object";
    c.level = card::Level::Countable;
    // table_existence.cpp ~200: camera (mask) channel + LiDAR carve. ricoh only raises attention (bearings).
    c.channels = {{"zed", card::Role::Identity, true},
                  {"helios", card::Role::Occupancy, true},
                  {"ricoh", card::Role::Identity, false}};
    // Class template of table_model.h (w 1.0, h 0.6, table_height 0.75): the manifest sizes are mean-less
    // (std 0.30), so these are the belief defaults, floor-anchored span [0, H].
    c.geometry = {1.0f, 0.6f, 0.75f, 0.0f, 0.75f};
    // table_existence.cpp:200: p_detect = envelope(roi_fill, visible = in_fov_frac) * central_frac. in_fov is
    // INSIDE the envelope call (as its visible fraction), so it is not a separate term. (The one-sided
    // range_conf form there is the existence_absence_envelope=false fallback, off in etc/config.toml.)
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    sp.q_return = cfg.existence_detection_prob;
    sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}}},
                {"central", {}}};
    // round and square share one envelope (the existence code has a single det_env_); "" is the default entry.
    c.subtypes = {{"", sp}, {"round", sp}, {"square", sp}};
    // Instance terms (p_clear = YOLO score grade, and the blind-spot map) are NOT in card.h's closed term list,
    // so they cannot be named here without making the card unreadable; they are described in blind_spots and
    // the blind-spot map is carried by calibration/blind_spot_map_path.
    c.blind_spots = {
        "Close range: a table of about 0.9 m overflows the ZED frame and YOLO returns no mask, so a missing mask there is a blind spot, not absence.",
        "Per-instance score grade p_clear: a table YOLO scores just above its cutoff drops below it from a slightly different view.",
        "The solid band is an abstraction of a hollow plate with thin legs: LiDAR free space under the top is not trustworthy.",
        "Learned per-viewpoint blind spots (az, el, range) correct the prior when the blind-spot map is loaded."};
    const card::SubtypeParams& d = c.subtypes.at("");
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = d.env.min_fill != fleet.min_fill or d.env.max_fill != fleet.max_fill or d.env.soft != fleet.soft;
    c.calibration = "prior";
    if (bsm_samples > 0 and cfg.existence_blind_spot_map)
    {
        c.calibration = "blind_spot_map";
        c.blind_spot_map_path = cfg.existence_blind_spot_file;
    }
    else if (fitted)
        c.calibration = "fitted";
    return c;
}
}  // namespace rc
