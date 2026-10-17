/*
 * dsr_fact_source.h — episodic_agent's two READ adapters over the live graph (2026-10-17).
 *
 *   DsrFactSource  rc::em::FactSource over DSR: what the Builder snapshots as a record's pre/post facts.
 *   LiveDiff       the structural events the Builder attaches to open records (born/died, offered/withdrawn,
 *                  current_moved), from one id-per-type snapshot per poll.
 *   read_journal   the controller's journal wire: `em_journal` on its dedicated node (em_journal.h kJournalNodeName).
 *
 * Fact.values (ruling R2): the kFactAttrs attributes the node actually carries (type-attributed getters; a bool
 * reads 0/1), "x","y","yaw" from the RT edge parent→node (rt_translation, rt_rotation_euler_xyz), and
 * "cov_xx","cov_yy" from that edge's rt_covariance (6x6 row-major over [x,y,z,rx,ry,rz]: [0] and [7]) when present.
 * ABSENT IS NOT ZERO: whatever cannot be read is left out of the map.
 *
 * ltsm_id (doors only): the name of the `passage_N` metaconcept in ltsm memory (domain 2) whose `match` edge points
 * at the door's memory FACE, which ltsm_agent names "r<idx>_<live door name>" with idx = the live room's room_id
 * attribute, else the trailing digits of its name (ltsm_agent/src/room_eviction.cpp room_index). No face, no
 * passage, or no memory graph ⇒ "". ltsm_agent's place-based fallback for re-created doors is NOT reproduced.
 *
 * Main-thread use only (the graph API is thread-safe, but nothing here needs another thread).
 */
#pragma once

#include "../../common/episodic/em_builder.h"
#include "../../common/episodic/em_journal.h"
#include "../../common/episodic/em_schema.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <dsr/api/dsr_api.h>

namespace rc::em
{

class DsrFactSource : public FactSource
{
public:
    /// `ltsm` may be null (no memory graph declared): every ltsm_id is then "".
    DsrFactSource(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> ltsm);

    std::optional<Fact> fact(std::string_view node) const override;
    /// Name of the room the single `current` edge points at; "" when there is none or more than one.
    std::string current_room() const override;

    /// The passage_N in ltsm memory matching this live door's memory face; "" when unknown.
    std::string ltsm_id_of(const DSR::Node& door) const;

private:
    std::shared_ptr<DSR::DSRGraph> live_, ltsm_;
};

/// Structural events between two polls. The first poll only primes (it reports nothing: everything would be
/// "born"). Node ids per type in kEventNodeTypes; type "affordance" reports offered/withdrawn instead of
/// born/died; a died node is reported under the name it had when last seen. current_moved fires when the single
/// `current` edge points at a room different from the last one it pointed at (ambiguous moments are skipped).
class LiveDiff
{
public:
    std::vector<Event> poll(DSR::DSRGraph& G);

private:
    bool primed_ = false;
    std::map<std::uint64_t, std::pair<std::string, std::string>> known_;   // id -> (type, name)
    std::optional<std::uint64_t> current_;
};

/// The controller's journal wire: the em_journal attribute of the node named kJournalNodeName of type
/// kJournalNodeType. nullopt when that node is absent (or carries no journal yet).
std::optional<std::string> read_journal(DSR::DSRGraph& G);

}   // namespace rc::em
