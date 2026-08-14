/*
 * common/tools/ai2_ab.cpp — A/B comparison of two `etc/ai2_log.csv` traces from a concept agent.
 *
 * Built for the mask FRAME-CONTRACT flip (masks moved to camera frame; the room transform now lives in
 * common/mask_ingestor), where the question is: does the transform done at the CONSUMER reproduce the
 * one the voxelizer used to bake into the points? But nothing here is specific to that change — it is a
 * generic "did these two runs converge to the same geometry, and was one steadier than the other?".
 *
 * ★LOCALE (CLAUDE.md). These machines run LANG=es_ES.UTF-8, where the decimal separator is a COMMA.
 * The agents are Qt programs, so Qt's setlocale(LC_ALL,"") activates that for the C library and
 * strtof/atof/strtod stop dead at the '.' of a point-formatted file — SILENTLY, returning the integer
 * part ("-2.23665" → -2). A standalone harness has no Qt, so it stays in "C" and the bug VANISHES,
 * which means a naive tool and the agent would disagree about the same file. So this tool does BOTH of
 * the things that rule demands: it calls setlocale(LC_ALL,"") to stand in the agent's locale, and it
 * parses exclusively with std::from_chars, which is locale-independent by definition and reports
 * failure instead of guessing. Getting this wrong would make two IDENTICAL runs look wildly different.
 *
 * ★COLUMNS BY NAME, NOT INDEX. The agents do not share one ai2_log schema — door_concept has no
 * pkt_fid/pkt_ts, table_concept does. Every column here is resolved from the header line.
 *
 * ★INSTANCES MATCHED BY POSITION, NOT BY NAME. Node names RECYCLE: a freed `table_N` number gets
 * re-handed to a different object, in-run and across restarts, so `table_2` in run A and `table_2` in
 * run B need not be the same physical table (this run pair already shows table_2 vs table_3 for what is
 * plainly one scene). Matching on the name would silently compare two different objects and report a
 * huge bogus delta. So instances are paired by NEAREST SETTLED CENTROID, and any instance left unpaired
 * is reported rather than dropped.
 *
 * WHAT IT COMPARES. Two sequential runs cannot be aligned frame-by-frame — they see different producer
 * frames — so per-row diffing is meaningless. What IS comparable, over a static scene:
 *   converged geometry — the median of the last `settle` fraction of each instance's rows. Tables do
 *                        not move, so both runs must agree here. This is the CORRECTNESS test.
 *   dispersion         — the std-dev of the fit over that same window. This is the STABILITY test, and
 *                        it is where a stale/lagged pose shows up: the mean can be right while the fit
 *                        jitters. For the frame flip this is the number that matters under motion.
 *   motion coverage    — peak motion_dotd per run. A comparison made while PARKED is worthless for the
 *                        pose-extrapolation question (the extrapolation term is identically zero at
 *                        stillness), so the tool refuses to imply a verdict when either run never moved.
 *
 * BUILD (no cmake, no deps):
 *   g++ -std=c++23 -O2 -o /tmp/ai2_ab common/tools/ai2_ab.cpp
 * USE:
 *   /tmp/ai2_ab <A.csv> <B.csv> [--settle 0.3] [--label-a NAME] [--label-b NAME]
 * A run boundary is detected by the `cycle` column going backwards; only the LAST run in each file is
 * compared (the agents APPEND across restarts — the single most likely way to misread these files).
 */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ── Locale-independent scalar parse. Returns nullopt on ANY malformed field instead of guessing. ──
std::optional<double> parse_num(std::string_view s)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
    while (not s.empty() and (s.back()  == ' ' or s.back()  == '\t' or s.back() == '\r')) s.remove_suffix(1);
    if (s.empty()) return std::nullopt;
    double v{};
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} or ptr != s.data() + s.size()) return std::nullopt;
    return v;
}

std::vector<std::string_view> split(std::string_view line, char sep = ',')
{
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (true)
    {
        const auto pos = line.find(sep, start);
        if (pos == std::string_view::npos) { out.push_back(line.substr(start)); break; }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

// One fitted instance's settled statistics.
struct Stats
{
    std::string node;                 // last name seen (informational only — names recycle)
    std::size_t rows = 0;             // rows in the settle window
    std::size_t rows_total = 0;
    std::map<std::string, double> med;   // per-column median over the settle window
    std::map<std::string, double> sd;    // per-column std-dev over the settle window
};

struct Run
{
    std::string label;
    std::vector<Stats> instances;
    double peak_motion = 0.0;
    std::size_t rows = 0;
};

double median_of(std::vector<double> v)
{
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    return v[mid];
}

double stddev_of(const std::vector<double>& v)
{
    if (v.size() < 2) return 0.0;
    double m = 0.0;
    for (const double x : v) m += x;
    m /= static_cast<double>(v.size());
    double s = 0.0;
    for (const double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

// Columns compared. Present-if-present: a schema lacking one simply omits it from the report.
const std::vector<std::string> kGeom  = {"cx", "cy", "H", "w", "h", "yaw"};
const std::vector<std::string> kAux   = {"npts", "range"};

std::optional<Run> load(const std::string& path, const std::string& label, double settle)
{
    std::ifstream f(path);
    if (not f) { std::println(stderr, "[ai2_ab] cannot open {}", path); return std::nullopt; }
    // Belt and braces: even reading through a stream, pin the classic locale so no separator surprise
    // can enter via the stream's own facets. The actual number parsing is from_chars regardless.
    f.imbue(std::locale::classic());

    std::string header;
    if (not std::getline(f, header)) { std::println(stderr, "[ai2_ab] {} is empty", path); return std::nullopt; }

    std::map<std::string, std::size_t> col;
    {
        const auto names = split(header);
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            std::string n{names[i]};
            while (not n.empty() and (n.back() == '\r' or n.back() == ' ')) n.pop_back();
            col[n] = i;
        }
    }
    if (not col.contains("cycle") or not col.contains("node"))
    {
        std::println(stderr, "[ai2_ab] {}: header has no 'cycle'/'node' column — not an ai2_log", path);
        return std::nullopt;
    }

    // Read every row, splitting into runs at a backwards 'cycle'. Keep only the LAST run: these files
    // APPEND across restarts, and comparing a fresh run against a stale one is the classic misread.
    struct Row { double cycle; std::string node; std::map<std::string, double> vals; };
    std::vector<std::vector<Row>> runs(1);
    double prev_cycle = -1.0;
    std::size_t bad_fields = 0, total_rows = 0;

    for (std::string line; std::getline(f, line); )
    {
        if (line.empty()) continue;
        const auto fields = split(line);
        if (fields.size() <= col.at("node")) continue;
        const auto c = parse_num(fields[col.at("cycle")]);
        if (not c) { ++bad_fields; continue; }
        ++total_rows;
        if (*c < prev_cycle) runs.emplace_back();
        prev_cycle = *c;

        Row r;
        r.cycle = *c;
        r.node  = std::string{fields[col.at("node")]};
        for (const auto& want : {kGeom, kAux})
            for (const auto& name : want)
                if (const auto it = col.find(name); it != col.end() and it->second < fields.size())
                    if (const auto v = parse_num(fields[it->second]); v) r.vals[name] = *v;
        if (const auto it = col.find("motion_dotd"); it != col.end() and it->second < fields.size())
            if (const auto v = parse_num(fields[it->second]); v) r.vals["motion_dotd"] = *v;
        runs.back().push_back(std::move(r));
    }

    if (bad_fields > 0)
        std::println("[ai2_ab] {}: {} rows skipped (unparseable 'cycle') — check the file is not truncated",
                     path, bad_fields);

    const auto& last = runs.back();
    if (last.empty()) { std::println(stderr, "[ai2_ab] {}: last run has no rows", path); return std::nullopt; }

    Run out;
    out.label = label;
    out.rows  = last.size();
    if (runs.size() > 1)
        std::println("[ai2_ab] {}: file holds {} runs ({} rows total); comparing the LAST one ({} rows)",
                     path, runs.size(), total_rows, last.size());

    // Group by node name WITHIN a run (names are stable inside one process), then settle-window stats.
    std::map<std::string, std::vector<const Row*>> by_node;
    for (const auto& r : last)
    {
        by_node[r.node].push_back(&r);
        if (const auto it = r.vals.find("motion_dotd"); it != r.vals.end())
            out.peak_motion = std::max(out.peak_motion, it->second);
    }

    for (const auto& [node, rows] : by_node)
    {
        Stats s;
        s.node = node;
        s.rows_total = rows.size();
        const auto keep = std::max<std::size_t>(1, static_cast<std::size_t>(
                              std::llround(static_cast<double>(rows.size()) * settle)));
        const std::size_t first = rows.size() - keep;
        s.rows = keep;
        for (const auto& name : kGeom)
        {
            std::vector<double> v;
            for (std::size_t i = first; i < rows.size(); ++i)
                if (const auto it = rows[i]->vals.find(name); it != rows[i]->vals.end()) v.push_back(it->second);
            if (v.empty()) continue;
            s.med[name] = median_of(v);
            s.sd[name]  = stddev_of(v);
        }
        for (const auto& name : kAux)
        {
            std::vector<double> v;
            for (std::size_t i = first; i < rows.size(); ++i)
                if (const auto it = rows[i]->vals.find(name); it != rows[i]->vals.end()) v.push_back(it->second);
            if (not v.empty()) s.med[name] = median_of(v);
        }
        out.instances.push_back(std::move(s));
    }
    return out;
}

// Pair instances across runs by NEAREST SETTLED CENTROID (never by node name — names recycle).
struct Pair { const Stats* a; const Stats* b; double dist; };

std::vector<Pair> match(const Run& A, const Run& B, double max_dist_m)
{
    std::vector<Pair> pairs;
    std::vector<bool> used(B.instances.size(), false);
    for (const auto& a : A.instances)
    {
        if (not a.med.contains("cx") or not a.med.contains("cy")) continue;
        double best = std::numeric_limits<double>::max();
        long   bi   = -1;
        for (std::size_t j = 0; j < B.instances.size(); ++j)
        {
            if (used[j]) continue;
            const auto& b = B.instances[j];
            if (not b.med.contains("cx") or not b.med.contains("cy")) continue;
            const double dx = a.med.at("cx") - b.med.at("cx");
            const double dy = a.med.at("cy") - b.med.at("cy");
            if (const double d = std::hypot(dx, dy); d < best) { best = d; bi = static_cast<long>(j); }
        }
        if (bi >= 0 and best <= max_dist_m)
        {
            used[static_cast<std::size_t>(bi)] = true;
            pairs.push_back({&a, &B.instances[static_cast<std::size_t>(bi)], best});
        }
        else
            pairs.push_back({&a, nullptr, best});
    }
    for (std::size_t j = 0; j < B.instances.size(); ++j)
        if (not used[j]) pairs.push_back({nullptr, &B.instances[j], 0.0});
    return pairs;
}

} // namespace

int main(int argc, char** argv)
{
    // Stand in the AGENT's locale (CLAUDE.md): a harness that silently stays in "C" answers a different
    // question than the one asked. Every number below is still parsed with from_chars, so this cannot
    // change a single parsed value — it exists so this tool cannot accidentally become locale-dependent.
    std::setlocale(LC_ALL, "");

    if (argc < 3)
    {
        std::println("usage: ai2_ab <A.csv> <B.csv> [--settle 0.3] [--label-a NAME] [--label-b NAME] [--max-dist 0.5]");
        return 2;
    }
    std::string pa = argv[1], pb = argv[2];
    std::string la = "A", lb = "B";
    double settle = 0.3, max_dist = 0.5;
    for (int i = 3; i + 1 < argc; i += 2)
    {
        const std::string k = argv[i], v = argv[i + 1];
        if      (k == "--settle")   { if (const auto x = parse_num(v)) settle = *x; }
        else if (k == "--max-dist") { if (const auto x = parse_num(v)) max_dist = *x; }
        else if (k == "--label-a")  la = v;
        else if (k == "--label-b")  lb = v;
    }
    settle = std::clamp(settle, 0.01, 1.0);

    const auto A = load(pa, la, settle);
    const auto B = load(pb, lb, settle);
    if (not A or not B) return 1;

    std::println("");
    std::println("A = {:<28} {} rows, peak motion {:.3f} m/s, {} instance(s)",
                 la + " (" + pa + ")", A->rows, A->peak_motion, A->instances.size());
    std::println("B = {:<28} {} rows, peak motion {:.3f} m/s, {} instance(s)",
                 lb + " (" + pb + ")", B->rows, B->peak_motion, B->instances.size());
    std::println("settle window = last {:.0f}% of each instance's rows", settle * 100.0);

    // ★A comparison made while parked cannot answer the pose question — say so instead of implying a pass.
    constexpr double kMovedThresh = 0.05;   // m/s
    const bool moved = A->peak_motion > kMovedThresh and B->peak_motion > kMovedThresh;
    if (not moved)
        std::println("\n  ⚠ ONE OR BOTH RUNS NEVER MOVED (peak < {:.2f} m/s). At stillness the pose-extrapolation\n"
                     "    term is identically zero, so agreement here says NOTHING about the moving case —\n"
                     "    which is the only case where the two paths can differ. Re-run over a driven route.",
                     kMovedThresh);

    const auto pairs = match(*A, *B, max_dist);
    std::println("");
    for (const auto& p : pairs)
    {
        if (p.a == nullptr or p.b == nullptr)
        {
            const Stats* s = p.a ? p.a : p.b;
            std::println("UNPAIRED  {:<10} in {}  (cx={:.3f} cy={:.3f}) — no counterpart within {:.2f} m",
                         s->node, p.a ? la : lb,
                         s->med.contains("cx") ? s->med.at("cx") : 0.0,
                         s->med.contains("cy") ? s->med.at("cy") : 0.0, max_dist);
            continue;
        }
        std::println("PAIR  {} [{}]  <->  {} [{}]   centroid separation {:.4f} m",
                     p.a->node, la, p.b->node, lb, p.dist);
        std::println("      {:<8} {:>12} {:>12} {:>12}   {:>10} {:>10}",
                     "col", la + " med", lb + " med", "delta", la + " sd", lb + " sd");
        for (const auto& name : kGeom)
        {
            if (not p.a->med.contains(name) or not p.b->med.contains(name)) continue;
            const double ma = p.a->med.at(name), mb = p.b->med.at(name);
            const double sa = p.a->sd.count(name) ? p.a->sd.at(name) : 0.0;
            const double sb = p.b->sd.count(name) ? p.b->sd.at(name) : 0.0;
            std::println("      {:<8} {:>12.5f} {:>12.5f} {:>12.5f}   {:>10.5f} {:>10.5f}",
                         name, ma, mb, mb - ma, sa, sb);
        }
        for (const auto& name : kAux)
            if (p.a->med.contains(name) and p.b->med.contains(name))
                std::println("      {:<8} {:>12.2f} {:>12.2f} {:>12.2f}",
                             name, p.a->med.at(name), p.b->med.at(name), p.b->med.at(name) - p.a->med.at(name));
        std::println("      rows in settle window: {} / {}  vs  {} / {}",
                     p.a->rows, p.a->rows_total, p.b->rows, p.b->rows_total);
        std::println("");
    }

    std::println("Reading it: DELTA is the correctness test — a static scene must converge to the same");
    std::println("geometry on both paths (same matrix, same timestamp, one process later). SD is the");
    std::println("stability test — that is where a stale/lagged pose shows up, with the mean still right.");
    return 0;
}
