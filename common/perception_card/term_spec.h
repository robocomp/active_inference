// term_spec.h -- the one card type the term library needs. Split out of card.h so that terms.h (which the agents'
// framework-free cores include) does not drag in the card's JSON codec, and with it media_transport/flat_json.h:
// door_concept's test_door_core_tracker forbids any media_transport include under src/core.
#pragma once

#include <map>
#include <string>

namespace rc::card
{
struct TermSpec
{
    std::string                  name;     // one of kTermNames (card.h)
    std::map<std::string, float> params;
};
}  // namespace rc::card
