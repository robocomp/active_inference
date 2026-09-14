/*
 * passage_harvest.h — collect what door_concept measured about a doorway, and turn it into a belief.
 *
 * The contract, the folding rules and the reasons behind each are in ../EVICTION.md; this header
 * says what the code does. Three jobs:
 *
 *   POLL   Each cycle, look at every live door node for a NEW `passage_seq`. door_concept writes one
 *          judgement episode per VISIT (enter the assessment zone, judge, leave or cross), so a new
 *          sequence number means one new episode to record. Poll-based on purpose: crossings happen
 *          without evictions — the apartment is ONE fitted room, so going through its doors changes
 *          no room and would fire no fence to harvest at.
 *   LOG    Append the episode to etc/passages.csv, always, whatever we do with it afterwards. The
 *          CSV is the durable truth; the belief is derived from it and recomputable if lost.
 *   FOLD   At eviction, fold the accumulated evidence into the passage node's two Beta posteriors.
 *
 * ★ WHAT FOLDS IS NOT WHAT HAPPENED. Three rules, each of which exists because the obvious version
 *   is wrong (see EVICTION.md for the arithmetic):
 *
 *   - `passage_llr` folds from ANY visit, through the likelihood ratio, NOT as a fractional count.
 *     `α += p` adds a full count of mass however little the visit discriminated: 100 uninformative
 *     visits give Beta(51,51) — "50/50 to ±5%", which nobody measured. The exact update is a
 *     moment-matched mixture whose weight comes from the ratio, so llr == 0 is an exact no-op.
 *   - An UNCLAIMED crossing is logged and NOT folded. The robot only walks through doorways that
 *     are already passable, so folding those makes α climb while β never moves: a door shut 90% of
 *     the time would read "reliably passable".
 *   - Of the seven `aff_outcome` words only `Timeout` is negative evidence about the APERTURE —
 *     the robot got there, tried, and did not get through. `Unreachable`/`Infeasible`/`Refused`/
 *     `Abandoned`/`OutsideRoom` are facts about the APPROACH; folding them would teach "this
 *     doorway is blocked" when the truth is "we could not get to it", or (OutsideRoom) that the
 *     producer published a pose outside its own layout.
 */
#pragma once

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ltsm
{

/// One judgement episode, exactly as it goes into the CSV.
struct PassageRow
{
    std::string   door, room, datetime, open_answer, outcome, fold_claim;
    int           seq = 0;
    bool          crossed = false, passable = false, open_requested = false, episode_cut = false;
    bool          fold_llr = false;
    /// ★ `open_prob == 0` means UNKNOWN, not "impassable": door_concept publishes a neutral 0 when
    /// its phi estimate is prior-derived rather than observed. Never fold this column as evidence.
    float         open_prob = 0.f, llr = 0.f, clear_span_m = 0.f, body_width_m = 0.f, duration_s = 0.f;
};

/// The two Beta posteriors a passage carries, plus the counts that make them readable.
struct PassageBelief
{
    float pass_alpha = 1.f, pass_beta = 1.f;   ///< p(passable without intervention)
    float open_alpha = 1.f, open_beta = 1.f;   ///< p(an open request succeeds | requested)
    int   crossings  = 0;
    int   seq_last   = 0;
    std::string last_datetime;
};

/// The inversion audit's verdict. INCONCLUSIVE is a real answer, not a failure to produce one.
struct AuditVerdict
{
    enum class Result { Inconclusive, Ok, Inverted };
    Result      result = Result::Inconclusive;
    int         crossed_rows = 0;
    float       mean_llr = 0.f;
    float       strongest_llr = 0.f;   ///< the top-|llr| crossed row: where an inversion shows first
    int         folded_negatives = 0;
    /// Timeouts whose request the provider ACCEPTED: the door opened and we could not see it. These
    /// credit `open_beta` on a provider that is working perfectly, so they must be countable.
    int         accepted_unconfirmed = 0;
    std::string text;
};

class PassageHarvest
{
public:
    PassageHarvest(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory,
                   std::filesystem::path csv_path);

    /// Scan the live graph for doors carrying a `passage_seq` we have not seen. Returns the number
    /// of episodes recorded this cycle (0 almost always).
    int poll();

    /// Fold everything accumulated for `door_name` into the memory passage node. Called at the
    /// eviction fence, when the room-side face — and therefore the passage — exists to carry it.
    bool fold_into(std::uint64_t passage_node_id, const std::string &door_name);

    /// Read the CSV back and judge whether the silhouette channel is inverted. Never returns Ok on
    /// evidence too weak to distinguish: see the INCONCLUSIVE rule in EVICTION.md.
    [[nodiscard]] AuditVerdict audit(const std::string &door_name) const;

    [[nodiscard]] int rows_written() const { return rows_written_; }

    /// The last row this door wrote, read back FROM THE CSV — so a caller checking what was recorded
    /// exercises the write and the parse, not an in-memory copy of what we meant to write.
    [[nodiscard]] std::optional<PassageRow> last_row_for(const std::string &door_name) const;

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    std::filesystem::path          csv_;

    std::unordered_map<std::string, int>           last_seq_;    ///< door name → last harvested seq
    std::unordered_map<std::string, PassageBelief> pending_;     ///< door name → evidence not yet folded
    std::set<std::string>                          unknown_answers_;  ///< reported once each, never suppressed silently

    int  rows_written_ = 0;

    bool append(const PassageRow &r);
    [[nodiscard]] std::vector<PassageRow> read_rows(const std::string &door_name) const;
    void apply(PassageBelief &b, const PassageRow &r);   // non-static: reports an unknown answer once
};

/// The exact update for a noisy observation: the posterior is a mixture of Beta(a+1,b) and
/// Beta(a,b+1) weighted by the likelihood ratio, moment-matched back to a Beta. `llr` in nats.
/// llr == 0  ⇒ the weight equals the current mean ⇒ the mixture IS the prior ⇒ an exact no-op.
void fold_llr(float &alpha, float &beta, float llr);

}   // namespace ltsm
