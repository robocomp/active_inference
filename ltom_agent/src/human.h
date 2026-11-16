// HumanBook: pure model of the ONE human at the chat — their name and the objects they own (no DSR, no Qt).
// Spec: docs/superpowers/specs/2026-11-11-human-model-design.md. Every fact carries when it was said and the exact
// words; a name change and a retraction are kept as history, never deleted.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ltom
{
class Repertory;

struct Said { std::int64_t t_ms = 0; std::string words; };
struct Ownership
{
    std::uint64_t object_uid = 0;      // the repertory's stable id; chair_N is display only
    std::string cls, name_when_said;
    Said said;                         // confirmed in the chat before it is stored
    std::optional<Said> retracted;     // "that's not my chair": history, never deleted
};
// A belief learned from the person's answers (spec 2026-11-15-changed-object): dated records with their words, never
// a bare number. Keys: "moves_things" (the person moves things), "gets_moved:<memory uid>" (this object gets moved).
// forgotten: dated "forget that" requests; the records are dropped, the request is kept.
struct BeliefRecord { Said said; bool yes = false; std::uint64_t object_uid = 0; };
struct Belief { std::string key; std::vector<BeliefRecord> records; std::vector<Said> forgotten; };
struct NameRecord { std::string value; Said said; };
struct Human
{
    std::string id = "human_1", role = "developer";
    std::vector<NameRecord> names;     // last = current; earlier = history
    std::vector<Ownership> things;
    std::vector<Belief> beliefs;
};
enum class ClaimKind { name, owns, disowns, moved, forget };
// moved: value "yes|no|unknown:<question stamp>", object_uid = the memory asked about. forget: value = belief key.
struct Claim { ClaimKind kind = ClaimKind::name; std::string value; std::uint64_t object_uid = 0; std::string words; std::uint64_t stamp = 0; };
struct Answer { std::uint64_t stamp = 0; std::string text; };   // "recorded" | "rejected: <reason>"

// "recorded" is said only when the file was written (review I4): an unsaved fact would be gone at the next restart.
inline Answer answer_after_save(Answer a, bool saved)
{
    if (a.text == "recorded" and not saved)
        a.text = "rejected: my memory file could not be written, so I would forget it when I restart";
    return a;
}

std::optional<ClaimKind> parse_claim_kind(std::string_view s);
const char* claim_kind_name(ClaimKind k);

class HumanBook
{
public:
    const Human& human() const { return human_; }
    std::optional<std::string> name() const;
    std::vector<const Ownership*> owned() const;
    // Repertory is written by a moved claim (the link is confirmed or cut, the question closed).
    Answer apply(const Claim& c, Repertory& r, std::int64_t now_ms);
    void set(Human h);
    bool dirty() const { return dirty_; }
    void clear_dirty() { dirty_ = false; }

private:
    Belief& belief(std::string_view key);
    Human human_;
    std::optional<Answer> last_;       // the last applied claim's answer (same stamp ⇒ same answer, no re-apply)
    bool dirty_ = false;
};
}
