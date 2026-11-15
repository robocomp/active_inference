#include "repertory.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>

namespace ltom
{
namespace
{
double wrap_pi(double a)
{
    a = std::fmod(a, 2 * std::numbers::pi);
    if (a > std::numbers::pi) a -= 2 * std::numbers::pi;
    else if (a <= -std::numbers::pi) a += 2 * std::numbers::pi;
    return a;
}

// Squared Mahalanobis distance over (x, y, yaw), each axis independent, under the sum of both variances.
double chi2(const Pose2& a, const Pose2& b)
{
    const auto term = [](double r, double v) { return v > 0 ? r * r / v : (r == 0 ? 0.0 : INFINITY); };
    return term(a.x - b.x, a.vx + b.vx) + term(a.y - b.y, a.vy + b.vy)
         + term(wrap_pi(a.yaw - b.yaw), a.vyaw + b.vyaw);
}

bool wants_keyframe(const Entry& e, const Sighting& s)
{
    if (e.keyframes.empty()) return true;
    const Keyframe& k = e.keyframes.back();
    return k.room_key != s.room_key or chi2(s.pose, k.pose) > kKeyframeChi2;
}
}

Changes Repertory::observe(std::span<const Sighting> alive, const std::set<std::uint32_t>& owners_present,
                          const std::set<std::uint64_t>& present, std::int64_t now_ms)
{
    Changes ch;
    std::set<std::uint64_t> seen;

    for (const Sighting& s : alive)
    {
        seen.insert(s.uid);
        pending_.erase(s.uid);                     // reappeared while pending: a flap, life continues
        dirty_ = true;
        auto [it, fresh] = entries_.try_emplace(s.uid);
        Entry& e = it->second;
        bool new_life = fresh;
        if (fresh) e.uid = s.uid;
        else if (e.lives.empty()) new_life = true;
        else
        {
            Life& l = e.lives.back();
            switch (l.end)
            {
                case End::alive: break;
                case End::gap: l.end = End::alive; l.end_ms = -1; ch.reopened.push_back(s.uid); break;
                case End::death: case End::owner_exit: new_life = true; break;
            }
        }
        if (new_life)
        {
            e.lives.push_back(Life{s.born_ms, now_ms, now_ms, -1, End::alive});
            ch.born.push_back(s.uid);
        }
        e.cls = s.cls; e.name = s.name; e.w = s.w; e.d = s.d; e.h = s.h;
        if (not s.color_name.empty()) { e.color_name = s.color_name; e.color_prob = s.color_prob; }   // unmeasured keeps it
        if (s.exist_prob.has_value()) e.exist_prob = s.exist_prob;   // not stated this time keeps the last statement
        ++e.n_sightings;
        e.lives.back().last_seen_ms = now_ms;
        if (wants_keyframe(e, s)) e.keyframes.push_back(Keyframe{now_ms, s.room_key, s.pose});
    }

    for (auto& [uid, e] : entries_)
    {
        if (seen.contains(uid) or e.lives.empty() or e.lives.back().end != End::alive) continue;
        if (present.contains(uid)) { pending_.erase(uid); continue; }   // on the graph, only unresolvable: HOLD
        Life& l = e.lives.back();
        const bool owner_absent = not owners_present.contains(owner_of(uid));
        const auto [pit, first_miss] = pending_.try_emplace(uid, Pending{now_ms});
        const std::int64_t vanish = pit->second.vanish_ms;
        if (owner_absent) close_questions_of_live(uid, false);
        else if (not first_miss and now_ms - vanish >= grace_ms_) close_questions_of_live(uid, true);
        if (owner_absent)
        {
            l.end = End::owner_exit; l.end_ms = vanish; ch.owner_exited.push_back(uid);
            pending_.erase(uid); dirty_ = true;
        }
        else if (not first_miss and now_ms - vanish >= grace_ms_)
        {
            l.end = End::death; l.end_ms = vanish; ch.died.push_back(uid);
            pending_.erase(uid); dirty_ = true;
        }
    }
    tend_links(alive, now_ms);
    return ch;
}

// A link is held while its owner publishes it as the MAP (same_prob > 1/2: the binary {same, not}). An unconfirmed
// link whose sighting no longer says so is undone silently: nobody said "no", the evidence simply changed its mind.
// A confirmed link is the owner's word and evidence does not undo it.
void Repertory::tend_links(std::span<const Sighting> alive, std::int64_t now_ms)
{
    for (const Sighting& s : alive)
    {
        const bool says_same = s.link.has_value() and s.link->same_prob > 0.5f and s.link->memory_uid != s.uid;
        if (says_same)
        {
            const auto m = entries_.find(s.link->memory_uid);
            if (m == entries_.end()) continue;
            // The owner said no for that room: a stale link still on the node (one cycle) must not re-link it.
            if (gone_from(m->second, s.room_key)) continue;
            // The live object's MAP moved to another memory: the previous, unconfirmed link is undone first. A
            // confirmed one is the owner's word and stands.
            if (const auto me = entries_.find(s.uid); me != entries_.end() and me->second.continues.has_value() and
                                                      *me->second.continues != m->first)
            {
                const auto old = entries_.find(*me->second.continues);
                if (old != entries_.end() and not old->second.moves.empty() and old->second.moves.back().confirmed) continue;
                if (old != entries_.end()) cut_link(old->first, now_ms, "");
            }
            if (not m->second.continued_by.has_value()) record_link(m->first, s.uid, now_ms, s.link->same_prob);
            else if (*m->second.continued_by == s.uid and not m->second.moves.empty() and not m->second.moves.back().confirmed)
            {
                m->second.moves.back().prob = s.link->same_prob;
                dirty_ = true;
            }
            if (m->second.continued_by != s.uid or m->second.moves.empty()) continue;
            Move& mv = m->second.moves.back();
            if (not mv.confirmed and not mv.asked and mv.question_stamp == 0 and s.link->left_spot_prob > 0.5f)
            {
                auto stamp = static_cast<std::uint64_t>(now_ms);
                const auto open = questions();
                while (open.contains(stamp)) ++stamp;   // two questions in one cycle keep distinct stamps
                mv.question_stamp = stamp;
                dirty_ = true;
            }
            continue;
        }
        const auto me = entries_.find(s.uid);
        if (me == entries_.end() or not me->second.continues.has_value()) continue;
        const auto m = entries_.find(*me->second.continues);
        if (m == entries_.end() or m->second.moves.empty() or m->second.moves.back().confirmed) continue;
        cut_link(m->first, now_ms, "");
    }
}

void Repertory::close_questions_of_live(std::uint64_t live_uid, bool asked)
{
    const auto l = entries_.find(live_uid);
    if (l == entries_.end() or not l->second.continues.has_value()) return;
    const auto m = entries_.find(*l->second.continues);
    if (m == entries_.end() or m->second.moves.empty()) return;
    Move& mv = m->second.moves.back();
    if (mv.question_stamp != 0) { mv.question_stamp = 0; mv.asked = asked; dirty_ = true; }
}

std::map<std::uint64_t, Question> Repertory::questions() const
{
    std::map<std::uint64_t, Question> out;
    const auto has_confirmed = [](const Entry& e, std::size_t upto)
    { return std::ranges::any_of(e.moves | std::views::take(upto), &Move::confirmed); };
    for (const auto& [uid, e] : entries_)
    {
        if (not e.continued_by.has_value() or e.moves.empty() or e.moves.back().question_stamp == 0) continue;
        const Move& mv = e.moves.back();
        Question q{mv.question_stamp, uid, mv.live_uid, mv.from.room_key, mv.from, mv.to, has_confirmed(e, e.moves.size() - 1)};
        if (const auto l = entries_.find(mv.live_uid); l != entries_.end() and not l->second.keyframes.empty())
            q.new_kf = *std::ranges::max_element(l->second.keyframes, {}, &Keyframe::t_ms);
        if (not q.again and e.continues.has_value())
            if (const auto prev = entries_.find(*e.continues); prev != entries_.end())
                q.again = has_confirmed(prev->second, prev->second.moves.size());
        out.emplace(q.stamp, q);
    }
    return out;
}

bool Repertory::close_question(std::uint64_t memory_uid, std::uint64_t stamp)
{
    const auto m = entries_.find(memory_uid);
    if (m == entries_.end() or not m->second.continued_by.has_value() or m->second.moves.empty()) return false;
    Move& mv = m->second.moves.back();
    if (stamp == 0 or mv.question_stamp != stamp) return false;
    mv.question_stamp = 0;
    mv.asked = true;
    dirty_ = true;
    return true;
}

void Repertory::record_link(std::uint64_t memory_uid, std::uint64_t live_uid, std::int64_t now_ms, float prob)
{
    const auto m = entries_.find(memory_uid);
    const auto l = entries_.find(live_uid);
    if (m == entries_.end() or l == entries_.end() or m->second.keyframes.empty()) return;
    Move mv;
    mv.t_ms = now_ms;
    mv.live_uid = live_uid;
    mv.from = *std::ranges::max_element(m->second.keyframes, {}, &Keyframe::t_ms);
    if (not l->second.keyframes.empty()) mv.to = *std::ranges::max_element(l->second.keyframes, {}, &Keyframe::t_ms);
    mv.prob = prob;
    m->second.moves.push_back(mv);
    m->second.continued_by = live_uid;
    l->second.continues = memory_uid;
    dirty_ = true;
}

void Repertory::confirm_link(std::uint64_t memory_uid)
{
    const auto m = entries_.find(memory_uid);
    if (m == entries_.end() or not m->second.continued_by.has_value() or m->second.moves.empty()) return;
    m->second.moves.back().confirmed = true;
    m->second.moves.back().prob = 1.f;
    dirty_ = true;
}

void Repertory::cut_link(std::uint64_t memory_uid, std::int64_t now_ms, std::string words)
{
    const auto m = entries_.find(memory_uid);
    if (m == entries_.end() or not m->second.continued_by.has_value()) return;
    Entry& mem = m->second;
    if (const auto l = entries_.find(*mem.continued_by); l != entries_.end()) l->second.continues.reset();
    mem.continued_by.reset();
    if (not mem.moves.empty() and not mem.moves.back().confirmed) mem.moves.pop_back();   // it never happened
    if (not words.empty() and not mem.keyframes.empty())
        mem.gone.push_back(GoneFromRoom{now_ms, std::ranges::max_element(mem.keyframes, {}, &Keyframe::t_ms)->room_key, std::move(words)});
    dirty_ = true;
}

void Repertory::close_open_lives_as_gap()
{
    for (auto& [uid, e] : entries_)
        for (Life& l : e.lives)
            if (l.end == End::alive) { l.end = End::gap; l.end_ms = l.last_seen_ms; dirty_ = true; }
    pending_.clear();
}

void Repertory::merge_thumb(std::uint64_t uid, int bin, float score, std::vector<std::uint8_t> jpeg)
{
    const auto entry = entries_.find(uid);
    if (entry == entries_.end() or jpeg.empty()) return;
    const auto old = entry->second.thumbs.find(bin);
    if (old != entry->second.thumbs.end() and old->second.score >= score) return;
    entry->second.thumbs[bin] = {score, std::move(jpeg)};
    dirty_ = true;
}

void Repertory::insert_loaded(Entry e)
{
    const auto uid = e.uid;
    entries_.insert_or_assign(uid, std::move(e));
}
}
