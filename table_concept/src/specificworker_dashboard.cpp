/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit). Owns the evidence-monitor snapshot
 * push (refresh_evidence_monitor), the belief-timeseries pruning (prune_dead_series), and the dashboard
 * window-geometry persistence (restore/save_dashboard_geometry). The plots themselves are members of
 * SpecificWorker, fed from compute(); these methods manage their lifecycle and the monitor view.
 */

#include "specificworker.h"

#include "../../common/dashboard/belief_certainty.h"   // rc::dash::fill_certainty (SHARED)

#include "../../common/dashboard/window_geometry.h"    // rc::dashboard:: (SHARED)
#include "../../common/dashboard/concept_dashboard.h"  // rc::dashboard::build (SHARED)
#include "../../common/dashboard/belief_card_fill.h"   // rc::dashboard::fill_* (SHARED)
#include "../../common/dashboard/series_pruner.h"     // rc::dashboard:: (SHARED)
#include "table_geometry.h"   // rc::geom::belief_uncertainty
#include "table_dof.h"        // rc::kTableDofs — names/units/σ* for the BeliefInspector rows

#include <QByteArray>
#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>
#include <QColor>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>
#include <array>
#include <chrono>

// ─── Evidence monitor ────────────────────────────────────────────────────────────────────────────

// Section 1: push this cycle's evidence-pipeline counters, then (same tick, same data) the belief
// inspector — so the two sections can never show different cycles. Throttled to ~5 Hz. Main-thread.
void SpecificWorker::refresh_evidence_monitor()
{
    if (not evidence_monitor_)
        return;
    const auto now = std::chrono::steady_clock::now();
    if (last_monitor_tp_.time_since_epoch().count() != 0
        and std::chrono::duration<float>(now - last_monitor_tp_).count() < 0.2f)
        return;
    last_monitor_tp_ = now;

    evidence_monitor_->update_view(ev_g_);
    refresh_belief_inspector();   // same tick, same instance pass — the two views can never disagree
    refresh_belief_strip();       // …and the compact strip reads the same cycle as both
}

// ─── Compact belief strip ────────────────────────────────────────────────────────────────────────

// One row per table: the adequacy gap (nats still missing before Σ meets the consumer's σ*), the
// existence probability, the FE surprise, and the node's birth stamp. The widget owns the history — this
// only pushes the current instant, on the same ~5 Hz tick as the two panels above.
void SpecificWorker::refresh_belief_strip()
{
    if (not belief_strip_)
        return;   // headless (cfg_.show_dashboard=false): nothing was built

    std::vector<rc::BeliefStripRow> rows;
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefStripRow r;
        r.node        = inst.node_name;
        r.model       = inst.subtype;
        r.p_exists    = inst.existence.p_exists();
        r.surprise    = inst.fe_surprise;
        r.initialized = inst.ai2_initialized;

        // Same REPORTED covariance the inspector and the NBV planner use (its yaw entry carries the
        // discrete-mode entropy), so the strip cannot disagree with either.
        const auto S = inst.ai2_belief.covariance_reported();
        // `adequacy_gap_nats` returns 0 for a DOF table with no σ* anywhere — an empty sum — and 0 is
        // exactly the value that means "adequate". Ask first, and carry "no demand" as the -1 sentinel.
        rc::dash::fill_certainty(r, S, rc::kTableDofs);
        // Birth from the node's own creation stamp, so `age` survives a dashboard opened long after the
        // table was born. Absent (an older node, or one this agent adopted) ⇒ 0 ⇒ the widget falls back
        // to when it first saw the row and says so by measuring from there.
        if (const auto n = G->get_node(inst.node_id); n.has_value())
            r.birth_ms = G->get_attrib_by_name<timestamp_creation_att>(n.value()).value_or(0);

        rows.push_back(std::move(r));
    }
    belief_strip_->update_view(rows);
}

// Geometry persistence is SHARED (common/dashboard/window_geometry.h). The strip passes a max size: a
// restored geometry carries the window STATE, so a strip that was ever maximised comes back filling the
// screen — indistinguishable, to the user, from the big dashboard having opened itself.
void SpecificWorker::restore_strip_geometry()
{
    rc::dashboard::restore_window_geometry(strip_window_, QStringLiteral("table_concept"), QStringLiteral("BeliefStripWindow"),
                                           QSize(520, 210), QSize(900, 420));
}

void SpecificWorker::save_strip_geometry() const
{
    rc::dashboard::save_window_geometry(strip_window_, QStringLiteral("table_concept"), QStringLiteral("BeliefStripWindow"));
}

// ─── Belief inspector ────────────────────────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (state, Σ, discrete modes, scalar gauges) and push it to the
// bottom panel. Called from refresh_evidence_monitor(), so it inherits that method's ~5 Hz gate and its
// single pass over the live instances. All reads are main-thread.
void SpecificWorker::refresh_belief_inspector()
{
    if (not belief_inspector_)
        return;   // headless (cfg_.show_dashboard=false): nothing was built
    const auto now = std::chrono::steady_clock::now();

    std::vector<rc::BeliefCard> cards;
    cards.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefCard c;
        c.node  = inst.node_name;
        c.model = inst.subtype;   // "square" | "round" — the free-energy shape model-selection winner

        // REPORTED covariance: the yaw entry carries the discrete-mode entropy, so a mode-ambiguous table
        // shows the honest σ_yaw rather than the within-mode one. Same matrix the NBV planner scores.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::TableBelief::N> v = {s.cx, s.cy, s.H, s.w, s.h, s.yaw, s.t};
        rc::dashboard::fill_dofs(c, rc::kTableDofs, v, S);

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        rc::dashboard::fill_cov(c, S, rc::TableBelief::N);
        // Under the C2v symmetry quotient only Σ's DIAGONAL is mapped back to (w,h,yaw); the off-diagonals
        // stay in the (s,a1,a2) chart. Label the heatmap axes for what they are rather than mislabel them.
        if (rc::TableBeliefState::use_quotient)
            c.cov_labels = {"cx", "cy", "H", "s", "a1", "a2", "t"};

        const float p_swap  = inst.ai2_belief.mode_posterior();          // p(the w↔h swapped mode)
        c.modes.push_back({"w<->h", "keep", 1.0f - p_swap});
        c.modes.push_back({"w<->h", "swap", p_swap});
        const float p_round = 1.0f / (1.0f + std::exp(-inst.shape_evidence));   // accumulated log-Bayes factor
        c.modes.push_back({"shape", "round",  p_round});
        c.modes.push_back({"shape", "square", 1.0f - p_round});

        rc::dashboard::fill_scalars(c, inst, now);
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// ─── Dashboard construction + geometry ────────────────────────────────────────────────────────────


// No max size here: the full dashboard is one the user may legitimately maximise, so its saved state is
// honoured as-is. That asymmetry is the whole reason max_restored is a parameter and not a constant.
void SpecificWorker::restore_dashboard_geometry()
{
    rc::dashboard::restore_window_geometry(dashboard_window_, QStringLiteral("table_concept"), QStringLiteral("DashboardWindow"),
                                           QSize(1180, 900));
}

void SpecificWorker::save_dashboard_geometry() const
{
    rc::dashboard::save_window_geometry(dashboard_window_, QStringLiteral("table_concept"), QStringLiteral("DashboardWindow"));
}


// Remove the timeseries of instances no longer in the graph, so a removed one's lines don't persist (and one
// reborn under the same name gets fresh series via the idempotent add_series in the feed). SHARED — the
// agent supplies only the live names and its own remembered set.
void SpecificWorker::prune_dead_series()
{
    std::vector<std::string> live;
    live.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances()) live.push_back(inst.node_name);
    rc::dashboard::prune_dead_series({ts_plot_, ts_surprise_plot_, ts_cov_plot_, ts_res_plot_},
                                     live, ts_known_tables_);
}

// Build the belief strip (small, always open) and the full dashboard behind its "details" button. Both
// windows, their layout, the toggle and the geometry are SHARED (common/dashboard/concept_dashboard.h);
// what stays here is the four strings that name this concept and where the widget pointers land.
//
// Main thread, from initialize(). cfg_.show_dashboard = false builds NOTHING: every pointer stays null and
// the compute feed already no-ops on null.
void SpecificWorker::build_dashboard()
{
    const rc::dashboard::BuildSpec spec{
        .agent        = QStringLiteral("table_concept"),
        .series_title = QStringLiteral("Table — Free Energy, Surprise, Belief Uncertainty & Residuals"),
        .strip_title  = QStringLiteral("table_concept — beliefs"),
        .strip_label  = QStringLiteral("tables"),
        .dashboard_size = QSize(1180, 900),
    };
    const auto w = rc::dashboard::build(spec, cfg_.show_dashboard);

    custom_widget_    = w.custom;
    ts_plot_          = w.fe;
    ts_surprise_plot_ = w.surprise;
    ts_cov_plot_      = w.cov;
    ts_res_plot_      = w.res;
    belief_inspector_ = w.inspector;
    evidence_monitor_ = w.evidence;
    dashboard_window_ = w.dashboard_window;
    strip_window_     = w.strip_window;
    belief_strip_     = w.strip;
}
