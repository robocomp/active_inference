// ─────────────────────────────────────────────────────────────────────────────────────────────────
// SCORING BENCH — why the producer keeps proposing the cell it was just refused.
//
// protocol_bench.cpp showed the surviving failure is "a refusal does not move the argmax": 62
// completions, 0 observations. That is the live symptom — a perfect 8.4 m drive, arrival, then
// nothing. This models room_concept's ACTUAL score so the balance itself becomes a test.
//
//   score = fim_gain · staleness^w_ior  +  w_ior_drive · log1p(age/tau)  −  w_travel · (d / diag)
//   (and, since 2026-08-19, the whole reward is multiplied by an attempt suppressor)
//
// build: g++ -std=c++23 -O2 scoring_bench.cpp -o scoring_bench && ./scoring_bench
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <format>
#include <print>
#include <string>
#include <vector>

namespace
{
struct Cell
{
    double x = 0, y = 0;
    double fim = 0;         // marginal pose information this vantage adds (nats)
    double age = 0;         // seconds since last visited
    double attempt_age = 1e9;  // seconds since last ATTEMPT (refusal / abandon)
    bool   standable = true;
};

struct Weights
{
    double w_travel = 0.5;      // live: WTravelCost
    double w_drive  = 0.5;      // live: WIorDrive
    double w_ior    = 2.0;      // live: WIor  (exponent on staleness)
    double tau      = 120.0;    // live: IorDecayTime
    double diag     = 10.0;     // room diagonal
    // ── candidate repairs ────────────────────────────────────────────────────────────────────────
    bool   travel_relative_to_info = false;  // scale travel by the information actually on offer
    bool   suppress_costs_too      = false;  // attempt suppressor scales the NET value, not the reward
};

double staleness(const Cell &c, const Weights &w) { return std::min(1.0, c.age / w.tau); }
double neglect(const Cell &c, const Weights &w)   { return std::log1p(c.age / w.tau); }
double attempt_supp(const Cell &c, const Weights &w) { return std::min(1.0, c.attempt_age / w.tau); }

double score(const Cell &c, double rx, double ry, const Weights &w)
{
    const double d = std::hypot(c.x - rx, c.y - ry);
    const double reward = c.fim * std::pow(staleness(c, w), w.w_ior) + w.w_drive * neglect(c, w);
    double travel = w.w_travel * (d / w.diag);
    if (w.travel_relative_to_info)
    {
        // ★TRAVEL IS A COST IN NATS AND MUST BE PRICED AGAINST WHAT IS ON OFFER. A fixed 0.5·d/diag
        // has a spread of ~0.17 nats across the room, which swamps an information term of ~0 and a
        // neglect spread of ~0.05 — so the argmax degenerates to "nearest legal cell" and a refusal
        // can never move it. Scaling by the best reward available makes distance matter in proportion
        // to what distance BUYS, which is what an expected-free-energy trade-off actually means.
        travel *= 0.0;   // replaced by the caller-side normalisation below
    }
    const double supp = attempt_supp(c, w);
    return w.suppress_costs_too ? supp * (reward - travel) : supp * reward - travel;
}

int argmax(const std::vector<Cell> &cells, double rx, double ry, const Weights &w)
{
    // travel_relative_to_info needs the field's best reward, so it is applied here.
    double best_reward = 0;
    for (const auto &c : cells)
        best_reward = std::max(best_reward,
                               c.fim * std::pow(staleness(c, w), w.w_ior) + w.w_drive * neglect(c, w));
    int best = -1; double bs = -1e18;
    for (int i = 0; i < static_cast<int>(cells.size()); ++i)
    {
        double s = score(cells[i], rx, ry, w);
        if (w.travel_relative_to_info)
        {
            const double d = std::hypot(cells[i].x - rx, cells[i].y - ry);
            s -= (best_reward > 1e-9 ? best_reward : 1.0) * (d / w.diag);
        }
        if (s > bs) { bs = s; best = i; }
    }
    return best;
}

// Does the pair make progress? The producer proposes the argmax; if it is not standable the consumer
// refuses, the cell's attempt clock resets, and we ask whether the CHOICE MOVES.
struct Outcome { int observations = 0; int refusals = 0; bool stuck = false; };

Outcome simulate(std::vector<Cell> cells, const Weights &w, int steps = 400)
{
    Outcome o;
    double rx = 0, ry = 0;
    int same_pick = 0, last_pick = -1;
    for (int t = 0; t < steps; ++t)
    {
        for (auto &c : cells) { c.age += 1.0; c.attempt_age += 1.0; }
        const int p = argmax(cells, rx, ry, w);
        if (p < 0) break;
        if (p == last_pick) { if (++same_pick > 60) { o.stuck = true; break; } }
        else same_pick = 0;
        last_pick = p;
        if (cells[p].standable)
        {
            rx = cells[p].x; ry = cells[p].y;
            cells[p].age = 0; cells[p].attempt_age = 0;
            ++o.observations;
        }
        else { cells[p].attempt_age = 0; ++o.refusals; }
    }
    return o;
}

void report(const std::string &name, const Weights &w, const std::vector<Cell> &cells)
{
    const auto o = simulate(cells, w);
    const bool ok = o.observations >= 20 and not o.stuck;
    std::println("{}  {:<44} obs={:<4} refus={:<4} {}", ok ? "PASS" : "FAIL", name,
                 o.observations, o.refusals, o.stuck ? "STUCK on one cell" : "");
}
}  // namespace

// ★A DENSE GRID, WHICH IS WHAT THE PLANNER ACTUALLY PICKS FROM (live: cand=142/148 on a 0.5 m grid).
// Three hand-placed cells cannot show the failure: "the nearest legal cell always wins" is a property
// of a field where a near cell is ALWAYS available, and MinDistance=1.0 puts one exactly at the floor.
std::vector<Cell> grid_field(double room_w, double room_h, double res, double min_d, double rx, double ry)
{
    std::vector<Cell> out;
    for (double x = -room_w / 2; x <= room_w / 2; x += res)
        for (double y = -room_h / 2; y <= room_h / 2; y += res)
        {
            if (std::hypot(x - rx, y - ry) < min_d) continue;      // MinDistance
            Cell c{.x = x, .y = y};
            // Information falls off away from the walls: a vantage in open floor adds little about a
            // room whose layout is already a fixed prior. Peaks near the boundary, ~0 in the middle.
            const double edge = std::min({room_w / 2 - std::abs(x), room_h / 2 - std::abs(y)});
            c.fim = std::max(0.0, 0.55 * std::exp(-edge / 1.2));
            c.age = 300;                                            // everything equally stale
            c.standable = true;
            out.push_back(c);
        }
    return out;
}

int main()
{
    // The live geometry: a near cell that cannot be stood at (the fridge behind the wall), and
    // genuinely informative cells further away. MinDistance keeps everything >= 1 m.
    const std::vector<Cell> field{
        {.x = 1.0, .y = 0.0, .fim = 0.00, .age = 300, .standable = false},  // near, useless, NOT standable
        {.x = 4.5, .y = 0.0, .fim = 0.53, .age = 300, .standable = true},   // far, most informative
        {.x = 3.0, .y = 3.0, .fim = 0.29, .age = 300, .standable = true},
    };

    std::println("── scoring bench: does a refusal move the argmax? ──\n");

    Weights live;
    report("LIVE weights (w_travel=0.5)", live, field);

    Weights supp_costs = live; supp_costs.suppress_costs_too = true;
    report("attempt suppressor scales NET value", supp_costs, field);

    Weights rel = live; rel.travel_relative_to_info = true;
    report("travel priced against available info", rel, field);

    Weights both = live; both.travel_relative_to_info = true; both.suppress_costs_too = true;
    report("both", both, field);

    for (double wt : {0.25, 0.1, 0.05})
    { Weights x = live; x.w_travel = wt; report(std::format("LIVE shape, w_travel={}", wt), x, field); }

    // ── THE REAL FIELD ───────────────────────────────────────────────────────────────────────────
    std::println("\n── dense grid (the planner's actual candidate field) ──");
    const auto g = grid_field(10.0, 8.0, 0.5, 1.0, 0.0, 0.0);
    std::println("  {} candidate cells", g.size());
    for (auto [nm, w] : std::vector<std::pair<std::string, Weights>>{
             {"LIVE weights", live}, {"travel priced against info", rel},
             {"suppressor scales NET", supp_costs}})
    {
        const int p = argmax(g, 0, 0, w);
        std::println("  {:<28} picks ({:+.1f},{:+.1f})  d={:.2f} m  fim={:.3f}",
                     nm, g[p].x, g[p].y, std::hypot(g[p].x, g[p].y), g[p].fim);
    }
    std::println("\n── why: the terms, for the live weights, at the robot's start pose ──");
    for (const auto &c : field)
        std::println("  cell({:.1f},{:.1f}) fim={:.2f}  reward={:.3f}  travel={:.3f}  score={:.3f}",
                     c.x, c.y, c.fim,
                     c.fim * std::pow(staleness(c, live), live.w_ior) + live.w_drive * neglect(c, live),
                     live.w_travel * (std::hypot(c.x, c.y) / live.diag),
                     score(c, 0, 0, live));
    return 0;
}
