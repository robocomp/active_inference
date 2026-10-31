// cabinet_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only cabinet_config.h / cabinet_model.h): the unit test builds it without Qt / DSR.
// Described AS IT RUNS TODAY, per cabinet_existence.cpp:128; collective (run-of-cabinets) fields come later with the
// aggregate-objects work. Contract: common/perception_card/card.h.
#pragma once

#include "cabinet_config.h"
#include "cabinet_model.h"   // CabinetState: the class template (L, d, z0, z1)
#include "../../common/perception_card/card.h"

namespace rc
{
inline card::Card make_card(const CabinetConfig& cfg)
{
    card::Card c;
    c.concept_ = "cabinet";
    c.graph_type = "box";   // cortex registers no `cabinet` type: a cabinet is a `box` named cabinet_*
    c.level = card::Level::Countable;
    // lidar_precision / free_space_precision > 0 pump helios; the bpearl gate is 0 in etc/config.toml, so it is not a channel.
    c.channels = {{"zed", card::Role::Identity, true},
                  {"helios", card::Role::Occupancy, true}};
    const CabinetState tpl{};
    c.geometry = {tpl.L, tpl.d, tpl.z1 - tpl.z0, tpl.z0, tpl.z1};
    card::SubtypeParams sp;   // no detector envelope: the removal path runs the one-sided range confidence instead
    sp.q_return = cfg.existence_detection_prob;
    // cabinet_existence.cpp:128: p_detect = absence_range_conf(mean_range) * in_fov_frac * central_frac. Deliberately NOT
    // obliquity: this robot is chronically edge-on to cabinets.
    sp.terms = {{"range_conf", {{"ref_m", cfg.existence_absence_range_ref_m}, {"power", cfg.existence_absence_range_power}}},
                {"in_fov", {}}, {"central", {}}};
    c.subtypes[""] = sp;
    c.blind_spots = {
        "Far away the cabinet subtends too few pixels: range_conf decays as (ref/range)^power beyond ref_m, so a missing mask there is weak evidence.",
        "A run is long and wall-flush: it overflows the frustum close up, so in_fov_frac, not range, limits the near view.",
        "The model is per run, not per cabinet: the collective fields (a kitchen of runs) are not on this card yet."};
    c.calibration = "prior";
    return c;
}
}  // namespace rc
