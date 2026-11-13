#include "repertory.h"

#include <algorithm>
#include <cmath>
#include <numbers>

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
    return ch;
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
