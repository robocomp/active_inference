// door_existence_info.h — the information a ZED look buys about a door's EXISTENCE (pure; std only).
// Existence E ~ Bernoulli(p_exists) is a variable of the door's belief like its along-wall offset or width; one detector outcome
// Y carries I(E;Y) nats about it. It is part of the epistemic gain (EpistemicPlanner) and of the remembered-door look (door_look.h):
// the value of looking is high while existence is open (a 360-only birth sits at p ~ 0.7) and ZERO once it is settled either way -
// no rule 'ricoh-born => look' exists anywhere, the offer simply has nothing left to buy when the ZED has confirmed or removed it.
// The functions now live in common/existence_belief/existence_info.h (shared by every concept agent, 2026-11-23); this
// header only forwards them so door code is unchanged.
#pragma once
#include "../../common/existence_belief/existence_info.h"

namespace rc::door
{
using rc::exist::binary_entropy_nats;
using rc::exist::existence_mutual_info;
}  // namespace rc::door
