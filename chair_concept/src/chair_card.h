// chair_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only chair_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what chair_detect.h's chair_p_detect really multiplies. Contract: common/perception_card/card.h.
#pragma once

#include "chair_config.h"
#include "chair_model.h"   // ChairState: the class template sizes
#include "../../common/perception_card/card.h"

namespace rc
{
inline card::Card make_card(const ChairConfig& cfg)
{
    card::Card c;
    c.concept_ = "chair";
    c.graph_type = "object";
    c.level = card::Level::Countable;
    // ZED mask = identity and the only channel whose absence removes. The ricoh 360 slices confirm a chair but their
    // absence never removes it ("RICOH CONFIRMS, NEVER REMOVES", chair_existence.cpp), so ricoh is not modelled.
    c.channels = {{"zed", card::Role::Identity, true},
                  {"ricoh", card::Role::Identity, false}};
    const ChairState tpl{};   // class template (a floor-anchored seat + backrest)
    c.geometry = {tpl.seat_w, tpl.seat_d, tpl.seat_h + tpl.back_h, 0.0f, tpl.seat_h + tpl.back_h};
    card::SubtypeParams sp;
    sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
    // chair_detect.h: envelope(roi_fill) * in_fov * central * wall line of sight. The envelope is the single-fill
    // overload (axes = 1) with visible = 1; in_fov is its own factor. Wall line of sight is a per-view geometric
    // 0/1 on the camera -> chair segment, which depends on the room, not on the class, so it is not a card term.
    sp.terms = {{"envelope", {{"min_fill", sp.env.min_fill}, {"max_fill", sp.env.max_fill}, {"soft", sp.env.soft}, {"axes", 1.0f}}},
                {"in_fov", {}},
                {"central", {}}};
    c.subtypes[""] = sp;
    c.blind_spots = {
        "Close range: a chair that overflows the ZED frame gets no YOLO mask; the envelope's upper shoulder makes that a blind spot, not absence.",
        "The envelope is the fleet default (ChairModel.Detect* are not set in etc/config.toml), not fitted to chairs.",
        "Wall line of sight is applied per view from the room polygon; the card cannot list it, so the card's P is an upper bound in another room."};
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = sp.env.min_fill != fleet.min_fill or sp.env.max_fill != fleet.max_fill or sp.env.soft != fleet.soft;
    c.calibration = fitted ? "fitted" : "prior";
    return c;
}
}  // namespace rc
