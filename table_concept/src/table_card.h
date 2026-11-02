// table_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only table_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what the agent's existence P(detect) really multiplies, read off
// table_existence.cpp. Contract: common/perception_card/card.h.
#pragma once

#include "table_config.h"
#include "table_model.h"   // TableState: the class template (w, h, table_height)
#include "../../common/perception_card/card.h"
#include "../../common/occlusion/occlusion.h"   // rc::occlusion::kOwnWallSkipM (the skip table_projection.cpp passes)

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
    // Class template of table_model.h's TableState (the manifest sizes are mean-less, std 0.30); floor-anchored [0, H].
    const TableState tpl{};
    c.geometry = {tpl.w, tpl.h, tpl.table_height, 0.0f, tpl.table_height};
    // Level 1: table_projection.cpp's walls_block(cam, sample, polygon, rc::occlusion::kOwnWallSkipM): LINE walls, 0.15 m.
    c.walls = card::WallOcclusion{rc::occlusion::kOwnWallSkipM, -1.0f, 0.0f};
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    sp.q_return = cfg.existence_detection_prob;
    if (cfg.existence_absence_envelope)
        // table_existence.cpp:200: p_detect = envelope(roi_fill, visible = in_fov_frac) * central_frac. roi_fill is the
        // single-fill overload (axes = 1); in_fov is INSIDE the envelope call (as its visible fraction).
        sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}, {"axes", 1.0f}}},
                    {"central", {}}};
    else
        // existence_absence_envelope = false: absence_range_conf(range) * in_fov_frac * central_frac, no fill envelope.
        sp.terms = {{"range_conf", {{"ref_m", cfg.existence_absence_range_ref_m}, {"power", cfg.existence_absence_range_power}}},
                    {"in_fov", {}}, {"central", {}}};
    // round and square share one envelope (the existence code has a single det_env_); "" is the default entry.
    c.subtypes = {{"", sp}, {"round", sp}, {"square", sp}};
    // Instance terms, applied after the geometry in table_existence.cpp (p_clear, then the blind-spot map).
    // p_clear for a fresh instance: table_existence.cpp:205 starts `float p_clear = 1.0f` and only changes it when
    // existence_score_grade and score_n > 0, so a hypothesis (no score history) takes 1.
    if (cfg.existence_score_grade) { c.instance_terms.push_back("p_clear"); c.instance_prior["p_clear"] = 1.0f; }
    if (cfg.existence_blind_spot_map) c.instance_terms.push_back("blind_spot_map");
    c.blind_spots = {
        "Close range: a table of about 0.9 m overflows the ZED frame and YOLO returns no mask, so a missing mask there is a blind spot, not absence.",
        "The solid band is an abstraction of a hollow plate with thin legs: LiDAR free space under the top is not trustworthy."};
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
