// door_decision.h — Task 6 of the door-existence model: the DECISION on a door place's log-odds L, the door's ROLE, and the
// running-median COST estimator. DSR-free, header-only.
//
//   exists  : L > 0                                   (the creation line)
//   remove  : L < -ln(C_delete / C_keep), never for a door being crossed
//
// Both costs are SECONDS of the robot's own experience: C_keep = what keeping a door that is not there costs (a failed passage
// attempt), C_delete = what deleting a door that is there costs (the time until it is found again). A route door's delete cost
// adds the controller's MaxRoomS: losing it can cost a whole room's exploration. No hysteresis, streak or timer: the two lines
// are the whole decision, and L between them neither creates nor deletes.
//
// Persisted as etc/door_costs.csv (kind,seconds,source), written by tools/door_costs_from_logs.py and grown online.
// Build + run: see tests/test_door_decision.cpp.
#pragma once
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <locale>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rc::door::core
{
enum class Role : std::uint8_t { Ordinary, Route, BeingCrossed };
struct Costs { float keep_s; float delete_ordinary_s; float delete_route_s; };
struct Verdict { bool exists; bool remove; float delete_line; };

/// exists = L > 0; delete_line = -ln(C_delete(role) / keep_s); remove = role != BeingCrossed and L < delete_line.
/// A door being crossed has no finite delete cost (deleting it under the robot strands the passage): its line is -inf.
/// When C_delete <= C_keep the line is >= 0 and the two regions overlap (0 < L < line): REMOVE WINS, exists = L > 0 and not remove.
/// Keeping a phantom then costs more than losing a door, so the costed line governs creation as well as deletion.
[[nodiscard]] inline Verdict decide(float L, Role role, const Costs& c)
{
    const float c_del = role == Role::Route ? c.delete_route_s : c.delete_ordinary_s;
    const float line = role == Role::BeingCrossed ? -INFINITY : -std::log(c_del / c.keep_s);
    const bool remove = role != Role::BeingCrossed and L < line;
    return {L > 0.0f and not remove, remove, line};
}

/// Crossing outranks the route; a name on the route faces is a Route door; anything else is Ordinary.
[[nodiscard]] inline Role door_role(const std::string& name, std::span<const std::string> route_faces, bool being_crossed)
{
    if (being_crossed)
        return Role::BeingCrossed;
    return std::ranges::find(route_faces, name) != route_faces.end() ? Role::Route : Role::Ordinary;
}

class CostEstimator
{
public:
    enum class Kind : std::uint8_t { Keep, DeleteOrdinary };

    /// ★ The one constant here: with NO sample of a kind its median is the spec's stated reading, 30 s (the order of one failed
    /// approach), for both kinds so that the two costs are equal and the delete line is L = 0. It is a prior until samples exist.
    static constexpr float kNoSampleS = 30.0f;

    void add(Kind k, float seconds, std::string_view source)
    {
        samples_[idx(k)].push_back(seconds);
        rows_.push_back({k, seconds, std::string(source)});
    }

    /// delete_route_s = delete_ordinary_s + max_room_s.
    [[nodiscard]] Costs costs(float max_room_s) const
    {
        const float del = median(samples_[idx(Kind::DeleteOrdinary)]);
        return {median(samples_[idx(Kind::Keep)]), del, del + max_room_s};
    }

    [[nodiscard]] std::size_t n(Kind k) const { return samples_[idx(k)].size(); }
    [[nodiscard]] int malformed() const { return malformed_; }

    /// kind,seconds,source. from_chars must consume the whole field; a malformed row is skipped, counted, one warning line.
    [[nodiscard]] static CostEstimator load(const std::filesystem::path& p)
    {
        CostEstimator e;
        std::ifstream f(p);
        if (not f)
            return e;
        std::string line;
        std::getline(f, line);   // header
        while (std::getline(f, line))
        {
            if (line.empty())
                continue;
            const auto c1 = line.find(',');
            const auto c2 = c1 == std::string::npos ? c1 : line.find(',', c1 + 1);
            if (c2 == std::string::npos) { ++e.malformed_; continue; }
            const std::string kind = line.substr(0, c1), sec = line.substr(c1 + 1, c2 - c1 - 1);
            float v = 0.0f;
            const char* end = sec.data() + sec.size();
            const auto r = std::from_chars(sec.data(), end, v);
            const bool ok = not sec.empty() and r.ec == std::errc{} and r.ptr == end and std::isfinite(v) and v > 0.0f;
            if (not ok or (kind != "keep" and kind != "delete_ordinary")) { ++e.malformed_; continue; }
            e.add(kind == "keep" ? Kind::Keep : Kind::DeleteOrdinary, v, line.substr(c2 + 1));
        }
        if (e.malformed_ > 0)
            std::fprintf(stderr, "[door_decision] %s: %d malformed row(s) ignored\n", p.string().c_str(), e.malformed_);
        return e;
    }

    void save(const std::filesystem::path& p) const
    {
        std::ofstream f(p);
        f.imbue(std::locale::classic());
        f << "kind,seconds,source\n";
        for (const auto& r : rows_)
            f << (r.kind == Kind::Keep ? "keep" : "delete_ordinary") << ',' << r.seconds << ',' << r.source << '\n';
    }

private:
    struct Row { Kind kind; float seconds; std::string source; };
    static constexpr std::size_t idx(Kind k) { return static_cast<std::size_t>(k); }

    [[nodiscard]] static float median(std::vector<float> v)   // by value: sorted here, every sample kept in the estimator
    {
        if (v.empty())
            return kNoSampleS;
        std::ranges::sort(v);
        const std::size_t m = v.size() / 2;
        return v.size() % 2 == 1 ? v[m] : 0.5f * (v[m - 1] + v[m]);
    }

    std::vector<float> samples_[2];
    std::vector<Row> rows_;
    int malformed_ = 0;
};
}   // namespace rc::door::core
