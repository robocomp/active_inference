// phantom_memory.h — a concept agent's PRIVATE memory of refuted detections ("no door here"). Owner ruling 2026-11-13: a refuted
// detection (a phantom, a YOLO hallucination) is a fact about the agent's own SENSOR, not about the world, so it does not go to LTSM:
// each agent keeps it in its own file, keyed by the LTSM room_id and expressed in that room's own frame (the room's identity, not its
// live name, which a recognition swap renames). The memory scales the birth evidence of a new candidate at the place; it never vetoes.
// Header-only, no DSR, no Qt. Numbers: written with std::to_chars through the classic locale, read with std::from_chars ONLY (the
// machines run LANG=es_ES.UTF-8, where strtof stops at the '.').
// File: `room_id,x,y,yaw,width,height,height_sd,removed_ms,exist_logodds,exist_physical,seen_ms,ledger`; the last four are the refuted
// place's own evidence (what a CSV removed row holds; an 8-column file is read as "absent"; the ledger is one comma-free cell).
// A malformed row is skipped; saved atomically (tmp + rename).
#pragma once
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <locale>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <system_error>
#include <vector>

namespace rc::phantom
{
struct Row
{
    std::uint64_t room_id = 0;
    float x = 0.f, y = 0.f, yaw = 0.f, width = 0.f, height = 0.f, height_sd = 0.f;
    std::int64_t removed_ms = 0;
    std::optional<float> exist_logodds, exist_physical;   // the evidence the place was removed with (PlaceRow::L_evidence, ::physical)
    std::optional<std::int64_t> seen_ms;                  // its last sighting (PlaceRow::last_seen_wall_ms): the age of that evidence
    std::string ledger;                                   // ViewLedger::serialise(); empty = none
    bool operator==(const Row&) const = default;
};

namespace detail
{
inline constexpr std::string_view kHeader8 = "room_id,x,y,yaw,width,height,height_sd,removed_ms";
inline constexpr std::string_view kHeader = "room_id,x,y,yaw,width,height,height_sd,removed_ms,exist_logodds,exist_physical,seen_ms,ledger";

template <class T>
[[nodiscard]] inline bool parse(std::string_view s, T& out)
{
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc{} and r.ptr == s.data() + s.size();
}

[[nodiscard]] inline std::optional<Row> parse_row(std::string_view line)
{
    std::string_view f[12];
    std::size_t n = 0;
    while (true)
    {
        const auto c = line.find(',');
        if (n == 12) return std::nullopt;   // a thirteenth column: not our file
        f[n++] = line.substr(0, c);
        if (c == std::string_view::npos) break;
        line.remove_prefix(c + 1);
    }
    if (n != 8 and n != 12) return std::nullopt;
    Row r;
    if (not (parse(f[0], r.room_id) and parse(f[1], r.x) and parse(f[2], r.y) and parse(f[3], r.yaw) and parse(f[4], r.width)
             and parse(f[5], r.height) and parse(f[6], r.height_sd) and parse(f[7], r.removed_ms)))
        return std::nullopt;
    if (n == 12)
    {
        const auto opt = [&](std::string_view s, auto& dst) -> bool   // empty = absent; garbage = a bad row
        {
            if (s.empty()) return true;
            std::remove_reference_t<decltype(*dst)> v{};
            if (not parse(s, v)) return false;
            dst = v;
            return true;
        };
        if (not (opt(f[8], r.exist_logodds) and opt(f[9], r.exist_physical) and opt(f[10], r.seen_ms))) return std::nullopt;
        r.ledger = std::string(f[11]);
    }
    return r;
}
}   // namespace detail

struct Memory
{
    std::vector<Row> rows;

    /// A missing file is an empty Memory (first run). nullopt only if the file exists and cannot be read.
    [[nodiscard]] static std::optional<Memory> load(const std::filesystem::path& p)
    {
        Memory m;
        std::error_code ec;
        if (not std::filesystem::exists(p, ec)) return m;
        std::ifstream f(p);
        if (not f) return std::nullopt;
        std::string line;
        bool first = true;
        while (std::getline(f, line))
        {
            if (not line.empty() and line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (first) { first = false; if (line == detail::kHeader or line == detail::kHeader8) continue; }
            if (const auto r = detail::parse_row(line); r.has_value()) m.rows.push_back(*r);
        }
        return m;
    }

    /// Whole file, atomically: <path>.tmp then rename.
    [[nodiscard]] bool save(const std::filesystem::path& p) const
    {
        auto tmp = p; tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            if (not f) return false;
            f.imbue(std::locale::classic());
            f << detail::kHeader << '\n';
            for (const auto& r : rows)
            {
                char buf[256];
                char* o = buf;
                char* const end = buf + sizeof buf;
                const auto put = [&](auto v) { o = std::to_chars(o, end, v).ptr; };
                const auto comma = [&] { *o++ = ','; };
                put(r.room_id); comma(); put(r.x); comma(); put(r.y); comma(); put(r.yaw); comma(); put(r.width); comma();
                put(r.height); comma(); put(r.height_sd); comma(); put(r.removed_ms); comma();
                if (r.exist_logodds) put(*r.exist_logodds);
                comma();
                if (r.exist_physical) put(*r.exist_physical);
                comma();
                if (r.seen_ms) put(*r.seen_ms);
                comma();
                f.write(buf, o - buf);
                f << r.ledger << '\n';
            }
            f.flush();
            if (not f) return false;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, p, ec);
        return not ec;
    }

    [[nodiscard]] std::vector<Row> of_room(std::uint64_t room_id) const
    {
        std::vector<Row> out;
        std::ranges::copy_if(rows, std::back_inserter(out), [&](const Row& r) { return r.room_id == room_id; });
        return out;
    }

    /// Rows of rooms LTSM no longer remembers are dropped (their frame is gone). Returns the number dropped.
    int forget_except(std::span<const std::uint64_t> remembered_room_ids)
    {
        const auto n0 = rows.size();
        std::erase_if(rows, [&](const Row& r) { return std::ranges::find(remembered_room_ids, r.room_id) == remembered_room_ids.end(); });
        return static_cast<int>(n0 - rows.size());
    }

    /// Replace every row of `room_id` by `fresh` (the room's removed places as the agent now holds them).
    void set_room(std::uint64_t room_id, std::span<const Row> fresh)
    {
        std::erase_if(rows, [&](const Row& r) { return r.room_id == room_id; });
        rows.insert(rows.end(), fresh.begin(), fresh.end());
    }
};
}   // namespace rc::phantom
