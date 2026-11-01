#pragma once
/*
 * imagine_see.h — "would you see the fridge from there, if door_2 were open?". The EXISTING can_see machinery
 * (can_see.h: the target concept's perception card, evaluate_scene with the doors marginalised, answer_can_see's
 * wording) run on the INTERVENED copy of the scene; nothing new is modelled here. Pure values, no DSR, no Qt.
 */

#include <optional>
#include <string>
#include <string_view>

#include "imagined_world.h"

namespace rc::chat::imagine
{

/// `from` is a place name ("" / "here" = where the robot is); the robot is imagined there facing the target (best
/// heading: it can turn). `door` + `opening` (a parsed Opening) intervene on a door first; an opening without a phi
/// ("as it is now") changes nothing. Errors of the intervention and the resolver's ASK come back verbatim.
[[nodiscard]] inline std::string answer_see(const ImaginedWorld& w, const std::optional<rc::card::Card>& card,
                                            std::string_view target, std::string_view from, std::string_view door = {},
                                            const std::optional<Opening>& opening = {})
{
    ImaginedWorld iw = w;
    std::string assumption;
    if (not door.empty() and opening.has_value() and opening->phi_rad.has_value())
    {
        auto r = with_door_phi(w, door, *opening->phi_rad);
        if (not r.has_value()) return r.error();
        iw = std::move(*r);
        assumption = std::format("if {} were {}", door, opening->words);
    }
    const auto sp = resolve_standpoint(iw, from);
    if (not sp.has_value()) return sp.error();
    if (not card.has_value()) return answer_no_card(true, "");

    const std::string tgt(target);
    const auto tg = resolve_target(iw.scene, tgt, tgt, card);
    if (not tg.has_value()) return tg.error();
    const auto r = evaluate_scene(*card, iw.scene, *tg, rc::card::Viewpoint{sp->xy, std::nullopt});
    std::string out = "Imagining" + (assumption.empty() ? std::string{} : " " + assumption) + ": ";
    out += answer_can_see(card, tg->label, sp->words, r);
    if (not iw.notes.empty())
    {
        out += " (";
        for (std::size_t i = 0; i < iw.notes.size(); ++i) out += (i ? "; " : "") + iw.notes[i];
        out += ")";
    }
    return out;
}

}   // namespace rc::chat::imagine
