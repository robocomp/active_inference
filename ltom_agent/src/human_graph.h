// The human at the chat on LTOM's memory graph (domain 5): person node `human_1` under root, `owns` edges to the
// memory nodes of the objects they own, and LTOM's answer to the chat's last claim. Also reads the chat's claim from
// the working graph. Spec: docs/superpowers/specs/2026-11-11-human-model-design.md.
#pragma once
#include "human.h"

#include <memory>
#include <optional>

#include <dsr/api/dsr_api.h>

namespace ltom
{
// The claim language_agent posts on its `chat_intention` node (human_claim_*); nullopt when absent or incomplete.
std::optional<Claim> read_claim(DSR::DSRGraph& working);

class HumanGraph
{
public:
    explicit HumanGraph(std::shared_ptr<DSR::DSRGraph> memory);
    // Repeatable; writes only on change.
    void sync(const HumanBook& book, const std::optional<Answer>& last);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
};
}
