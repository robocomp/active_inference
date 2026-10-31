// chair_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only chair_config.h): the unit test builds it without Qt / DSR.
// What the card lists is what chair_detect.h's chair_p_detect really multiplies. Contract: common/perception_card/card.h.
#pragma once

#include "chair_config.h"
#include "chair_model.h"   // the SDF constants and size bounds the generative block exports
#include "../../common/perception_card/card.h"
#include "../../common/occlusion/occlusion.h"   // rc::occlusion::kOwnWallSkipM (the skip chair_detect.h passes)

namespace rc
{
// The chair's generative block: the SAME compound the SDF in chair_model.cpp evaluates (seat slab + backrest + 4 legs),
// written as closed primitives bound to the four size variables. The numbers that are NOT variables come from
// ChairModel (LEG_HALF) and the prior (seat_thick, the SDF's own: one value, see chair_shape_test). The variable
// bounds are the prior's [prior.bounds], which ChairModel::apply_constraints clamps to as well.
inline card::Generative make_generative(const ChairPrior& pr)
{
    using card::Bind;
    using card::Prim;
    const auto K = [](float b) { return Bind{"", 0.0f, b}; };
    const auto V = [](const char* v, float a = 1.0f, float b = 0.0f) { return Bind{v, a, b}; };
    const float T = pr.seat_thick;
    card::Generative g;
    // seat slab: x,y centred, z base = seat_h - T, thickness T
    g.shape.push_back({Prim::Slab, {{"w", V("seat_w")}, {"d", V("seat_d")}, {"h", K(T)},
                                    {"x", K(0.0f)}, {"y", K(0.0f)}, {"z", V("seat_h", 1.0f, -T)}}});
    // backrest panel: width = seat_w, thickness T on the -y edge (centre y = -seat_d/2 + T/2), base at the seat top
    g.shape.push_back({Prim::Panel, {{"w", V("seat_w")}, {"t", K(T)}, {"h", V("back_h")},
                                     {"x", K(0.0f)}, {"y", V("seat_d", -0.5f, 0.5f * T)}, {"z", V("seat_h")}}});
    // four legs under the seat, flush with its footprint, from the floor to the seat's underside
    g.shape.push_back({Prim::LegSet, {{"n", K(4.0f)}, {"half", K(ChairModel::LEG_HALF)},
                                      {"fw", V("seat_w")}, {"fd", V("seat_d")},
                                      {"z0", K(0.0f)}, {"z1", V("seat_h", 1.0f, -T)}}});
    g.vars = {{"seat_w", pr.seat_w, pr.size_std, pr.seat_w_b.lo, pr.seat_w_b.hi, false},
              {"seat_d", pr.seat_d, pr.size_std, pr.seat_d_b.lo, pr.seat_d_b.hi, false},
              {"seat_h", pr.seat_h, pr.size_std, pr.seat_h_b.lo, pr.seat_h_b.hi, false},
              {"back_h", pr.back_h, pr.size_std, pr.back_h_b.lo, pr.back_h_b.hi, false}};
    g.placement.support = "floor";
    g.placement.contexts = {{"free", pr.p_free}, {"against_wall", pr.p_against_wall}};
    g.placement.wall_gap_mean = pr.wall_gap_mean;
    g.placement.wall_gap_sigma = pr.wall_gap_sigma;
    g.bounds_source = "chair.concept.toml [prior.bounds]";
    return g;
}

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
    // The chair the model projects: the manifest [prior.template] (a floor-anchored seat + backrest to its top).
    const float top = cfg.prior.top();
    c.geometry = {cfg.prior.seat_w, cfg.prior.seat_d, top, 0.0f, top};
    // Level 1: chair_detect.h's walls_block(cam, chair centre, polygon, rc::occlusion::kOwnWallSkipM): LINE walls, 0.15 m.
    c.walls = card::WallOcclusion{rc::occlusion::kOwnWallSkipM, -1.0f, 0.0f};
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
        "in_fov = overlap-area fraction of the projected box with the image; central = 1 minus the norm of the normalised ROI offset (0 at the image edge).",
        "Close range: a chair that overflows the ZED frame gets no YOLO mask; the envelope's upper shoulder makes that a blind spot, not absence.",
        "The envelope is the fleet default (ChairModel.Detect* are not set in etc/config.toml), not fitted to chairs.",
        "Wall line of sight is applied per view from the room polygon; the card cannot list it, so the card's P is an upper bound in another room."};
    const rc::detect::DetectorEnvelope fleet{};
    const bool fitted = sp.env.min_fill != fleet.min_fill or sp.env.max_fill != fleet.max_fill or sp.env.soft != fleet.soft;
    c.calibration = fitted ? "fitted" : "prior";
    c.generative = make_generative(cfg.prior);
    return c;
}
}  // namespace rc
