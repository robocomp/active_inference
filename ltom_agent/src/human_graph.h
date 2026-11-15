// The human at the chat on LTOM's memory graph (domain 5): person node `human_1` under root, `owns` edges to the
// memory nodes of the objects they own, and LTOM's answer to the chat's last claim. Also reads the chat's claim from
// the working graph. Spec: docs/superpowers/specs/2026-11-11-human-model-design.md.
#pragma once
#include "human.h"
#include "repertory.h"

#include <map>

#include <memory>
#include <optional>

#include <dsr/api/dsr_api.h>

namespace ltom
{
// The claim language_agent posts on its `chat_intention` node (human_claim_*); nullopt when absent or incomplete.
std::optional<Claim> read_claim(DSR::DSRGraph& working);

// The dated evidence lines of a belief: "YYYY-MM-DD HH:MM | yes|no|forgotten | <words> | <object>".
std::string belief_evidence_text(const Belief& b, const Repertory* r);

class HumanGraph
{
public:
    explicit HumanGraph(std::shared_ptr<DSR::DSRGraph> memory);
    // Repeatable; writes only on change. `r` names the objects in the belief evidence lines (nullptr: uid only).
    void sync(const HumanBook& book, const std::optional<Answer>& last, const Repertory* r = nullptr);
    // One `ltom_question_<stamp>` metaconcept under root per open question (changed object, 2026-11-15); a closed
    // question's node is deleted. Repeatable; writes only on change.
    void sync_questions(const std::map<std::uint64_t, Question>& open);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
};
}
