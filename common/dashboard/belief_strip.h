#pragma once

/*
 * belief_strip.h  —  the COMPACT fleet view: one row per instance, and the row is a time series
 *
 * SHARED, header-only dashboard widget (same pattern as evidence_monitor.h / belief_inspector.h:
 * header-only, NO Q_OBJECT, built + updated on the Qt/GUI thread, which in these agents is the compute
 * thread). It is the display you keep open; belief_inspector.h is the one you open when this says
 * something is wrong.
 *
 *     ▇ table_0   ▔▔╲▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  0.00   ████████ 1.00   4m12
 *     ▇ table_1   ▔▔▔▔▔▔▔▔╲▁▁▁▁╱▔╲▁▁▁▁▁  0.31   ██████░░ 0.78     51s
 *
 * WHY A SECOND WIDGET. The inspector is a full-fidelity dump of ONE instance — 7 DOF rows × 4 columns,
 * an N×N correlation heatmap, mode bars, six gauges. That is the right instrument when the question is
 * "why is yaw wrong". It is the wrong one for the standing question, which is "which instance is in
 * trouble, and is it getting better or worse" — nobody can hold eight covariance matrices at once, and a
 * snapshot cannot show a trend at all. So this widget drops everything except the one scalar that
 * answers it, and spends the recovered space on TIME.
 *
 * WHAT THE SCALAR IS. The certainty channel is the ADEQUACY GAP in nats,
 *
 *     gap = Σ_j max(0, ½·ln(Σ_jj / σ*_j²))            (common/ai_belief/dof_spec.h)
 *
 * — the information still missing before the belief meets the CONSUMER's precision demand. It is the
 * right choice for three reasons a raw σ cannot match: it is unit-free, so a chair (N=3) and a table
 * (N=7) are on ONE axis; it is the same quantity the epistemic planner maximises, so the display and the
 * behaviour cannot disagree; and its ZERO is meaningful — "adequately resolved" is set by the consumer's
 * tolerance, not by a tuned bound (CLAUDE.md: no thresholds). A falling trace is a belief being resolved;
 * a flat-high trace is an instance starved of evidence; a rising one is a belief losing its object.
 *
 * An agent whose planner publishes NO σ* (chair, bottle) must not have one invented for it. Those agents
 * pass gap_nats < 0 and the widget falls back to ½·ln det Σ — the belief's differential entropy up to a
 * constant. That is honest but WEAKER: comparable across time for one instance, not across instances,
 * and with no meaningful zero. The header says which channel is on screen.
 *
 * SCALES. The certainty axis is SHARED across rows (max over the visible window, printed in the header),
 * so row heights are comparable — a resolved instance is visibly flat at the floor while a starved one
 * rides high. Surprise (FE − baseline) is drawn as a faint second trace on its OWN per-widget scale: it
 * is an attention signal, and its units have nothing to do with the gap's.
 *
 * ORDER. Rows are ordered by BIRTH — oldest at top — from the node's `timestamp_creation`
 * (common/graph_provenance/creation_stamp.h), falling back to first-seen for a node with no stamp. Not
 * sorted by severity on purpose: a list that re-sorts itself while you read it is unreadable, and birth
 * order makes the strip a timeline of the scene. `age` is measured from that same instant, so a row
 * reads "born 51 s ago and still 0.31 nats short".
 *
 * A removed instance LINGERS greyed for `linger_s` before vanishing, so churn is visible — a phantom
 * that lives three seconds leaves a stub you can see instead of a flicker you had to catch in a log.
 *
 * Pure view: the agent fills vector<BeliefStripRow> and calls update_view() (throttle it to a few Hz).
 * The widget owns the history; the agent keeps no ring buffers. No DSR, no Eigen — just Qt.
 */

#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPaintEvent>
#include <QPen>
#include <QPolygonF>
#include <QString>
#include <QWidget>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace rc
{

// ─── View model ──────────────────────────────────────────────────────────────────────────────────

// One instance's CURRENT snapshot. The agent pushes these; the widget accumulates the history.
struct BeliefStripRow
{
    std::string node;                 // "table_0" — also the history key
    std::string model;                // optional tag ("round", "wall-run base"); "" = none

    // Certainty channel, preferred form: remaining nats to the consumer's σ*. < 0 ⇒ this agent publishes
    // no σ* at all, and the widget falls back to `logdet_nats`. NEVER pass a made-up target to get a
    // prettier trace — an invented σ* is a tuning knob wearing a display's clothes.
    float gap_nats    = -1.0f;
    // Fallback certainty channel: ½·ln det Σ (nats, up to a constant). Always computable. May be negative.
    float logdet_nats = std::numeric_limits<float>::quiet_NaN();

    float p_exists    = std::numeric_limits<float>::quiet_NaN();   // NaN ⇒ agent has no existence belief
    float surprise    = 0.0f;         // smoothed FE − baseline (attention signal)
    bool  initialized = false;        // belief live? false ⇒ row greyed, traces still recorded

    // Node birth, ms since the Unix epoch, from the graph's `timestamp_creation`. 0 ⇒ unknown, and the
    // widget substitutes the moment it first saw the row (correct for a node this agent just created,
    // wrong-but-harmless for one adopted at startup — the age then reads "since we noticed").
    std::uint64_t birth_ms = 0;
};

// ─── The strip ───────────────────────────────────────────────────────────────────────────────────

class BeliefStrip : public QWidget
{
public:
    explicit BeliefStrip(const QString& title, QWidget* parent = nullptr)
        : QWidget(parent), title_(title)
    {
        QFont f = font();
        f.setPointSizeF(f.pointSizeF() - 0.5);
        f.setFamily(QStringLiteral("monospace"));   // numeric columns; a mono face keeps them aligned
        setFont(f);
        setMinimumWidth(360);
        // The legend lives here rather than in the header, which is now column headings only.
        setToolTip(QStringLiteral(
            "One row per instance; each row is a time series.\n"
            "adequacy — nats of information still missing before Σ meets the consumer's demand σ*:\n"
            "    Σⱼ max(0, ln(σⱼ/σ*ⱼ)).  0 = adequately resolved.  DOWN = resolving, UP = losing it,\n"
            "    flat-high = starved of admissible evidence.\n"
            "    Agents that publish no σ* fall back to ½·ln|Σ| (comparable over time, not across rows).\n"
            "faint orange — FE surprise, on its own scale.\n"
            "p(existence) — the removal channel's probability that the object is still there.\n"
            "age — time since the node's creation stamp. Rows are ordered by birth, oldest first;\n"
            "    a removed instance lingers greyed for a few seconds."));
    }

    // Seconds of history on screen. The widget keeps a little more than this and prunes the rest.
    void set_visible_window(float seconds) { window_s_ = std::max(5.0f, seconds); }
    // How long a vanished instance stays on screen, greyed, before its row is dropped.
    void set_linger(float seconds)         { linger_s_ = std::max(0.0f, seconds); }

    // Push this cycle's snapshot for EVERY live instance. Rows absent from `rows` are treated as gone.
    void update_view(const std::vector<BeliefStripRow>& rows)
    {
        const float t = elapsed_s();

        for (const auto& r : rows)
        {
            Track& tr = tracks_[r.node];
            if (tr.samples.empty())
            {
                tr.first_seen_s = t;
                tr.order        = next_order_++;
            }
            tr.row       = r;
            tr.last_seen = t;
            tr.alive     = true;
            tr.samples.push_back({t, r.gap_nats, r.logdet_nats, r.p_exists, r.surprise});
        }

        // Anything not in this push has gone. Keep sampling it as dead so its trace runs off the right
        // edge instead of freezing mid-plot.
        for (auto& [name, tr] : tracks_)
        {
            if (tr.last_seen == t)
                continue;
            tr.alive = false;
            tr.samples.push_back({t, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::quiet_NaN(), 0.0f});
        }

        prune(t);
        const int want_h = header_h() + static_cast<int>(tracks_.size()) * row_h();
        if (want_h != height_hint_) { height_hint_ = want_h; updateGeometry(); }
        update();
    }

    QSize sizeHint() const override { return {480, std::max(header_h() + row_h(), height_hint_)}; }
    QSize minimumSizeHint() const override { return {360, header_h() + row_h()}; }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        const QFontMetrics fm = fontMetrics();
        const int ch  = std::max(1, fm.horizontalAdvance(QLatin1Char('0')));
        const int pad = ch;
        const float t = elapsed_s();
        const float t0 = t - window_s_;

        p.fillRect(rect(), QColor(250, 250, 250));
        p.setPen(QColor(180, 180, 180));
        p.drawRect(rect().adjusted(0, 0, -1, -1));

        // Rows in birth order (oldest first); a row with no stamp falls back to when we first saw it.
        std::vector<const Track*> ordered;
        ordered.reserve(tracks_.size());
        for (const auto& [name, tr] : tracks_) ordered.push_back(&tr);
        std::sort(ordered.begin(), ordered.end(), [](const Track* a, const Track* b)
        {
            if (a->row.birth_ms != 0 and b->row.birth_ms != 0) return a->row.birth_ms < b->row.birth_ms;
            if (a->row.birth_ms != b->row.birth_ms)            return a->row.birth_ms != 0;  // stamped first
            return a->order < b->order;
        });

        // ── Shared certainty scale ──────────────────────────────────────────────────────────────
        // One scale for every row so the rows are comparable. The gap channel pins its FLOOR at 0 —
        // that zero is the whole point of the measure ("adequate"), and letting it float would hide
        // whether a flat trace sits at adequacy or just stopped changing.
        const bool  use_gap = any_gap();
        float lo = use_gap ? 0.0f : std::numeric_limits<float>::max();
        float hi = -std::numeric_limits<float>::max();
        for (const Track* tr : ordered)
            for (const Sample& s : tr->samples)
            {
                if (s.t < t0) continue;
                const float v = use_gap ? s.gap : s.logdet;
                if (not std::isfinite(v) or (use_gap and v < 0.0f)) continue;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        if (hi <= -std::numeric_limits<float>::max() / 2) { lo = 0.0f; hi = 1.0f; }   // no data yet
        if (not (hi > lo)) hi = lo + 1.0f;                                            // flat ⇒ give it height

        float surp_hi = 0.0f;
        for (const Track* tr : ordered)
            for (const Sample& s : tr->samples)
                if (s.t >= t0 and std::isfinite(s.surprise)) surp_hi = std::max(surp_hi, s.surprise);

        // ── Column geometry, from the widest FORMATTED sample (no pixel constants) ───────────────
        int w_name = fm.horizontalAdvance(QStringLiteral("instance"));
        int w_val  = fm.horizontalAdvance(QStringLiteral("0.00"));
        int w_age  = fm.horizontalAdvance(QStringLiteral("age"));
        for (const Track* tr : ordered)
        {
            w_name = std::max(w_name, fm.horizontalAdvance(name_text(*tr)));
            w_val  = std::max(w_val,  fm.horizontalAdvance(value_text(*tr, use_gap)));
            w_age  = std::max(w_age,  fm.horizontalAdvance(age_text(*tr, t)));
        }
        const int w_pnum = fm.horizontalAdvance(QStringLiteral("0.00"));
        // The bar is widened if necessary so the "p(existence)" heading fits over its own group — a
        // heading that has to be elided to fit is not a heading.
        const int w_pbar = std::max(8 * ch, fm.horizontalAdvance(kExistHdr()) - pad - w_pnum);

        const int x_name  = pad;
        const int x_spark = x_name + w_name + pad;
        const int x_pbar  = width() - pad - w_age - pad - w_pnum - pad - w_pbar;
        const int x_pnum  = x_pbar + w_pbar + pad;
        const int x_age   = width() - pad - w_age;
        const int x_val   = x_pbar - pad - w_val;
        const int w_spark = std::max(0, x_val - pad - x_spark);

        // ── Header: real COLUMN HEADINGS, each sitting over the data it names ───────────────────
        // Not a caption strip. The certainty heading spans the sparkline AND its numeric column and is
        // right-aligned, so "…nats" lands directly above the numbers; "p(existence)" spans the bar and
        // its number; "age" sits over the ages. The legend that used to live here ("↓ = resolving") is
        // in the widget's tooltip — a heading should label a column, not explain one.
        {
            const int hh = header_h();
            p.fillRect(QRect(1, 1, width() - 2, hh - 1), QColor(238, 238, 238));
            QFont hf = p.font(); hf.setBold(true); p.setFont(hf);
            p.setPen(QColor(60, 60, 60));
            const int w_title = std::max(0, x_spark - pad - x_name);
            p.drawText(QRect(x_name, 0, w_title, hh), Qt::AlignVCenter | Qt::AlignLeft,
                       fm.elidedText(title_, Qt::ElideRight, w_title));
            hf.setBold(false); p.setFont(hf);
            p.setPen(QColor(120, 120, 120));

            const QString cert = use_gap
                ? QStringLiteral("adequacy 0–%1 nats").arg(hi, 0, 'f', 2)
                // Bracketed, not "lo–hi": ½ln|Σ| is normally NEGATIVE, and "-8.5–-5.3" is unreadable.
                : QStringLiteral("½ln|Σ| [%1, %2] nats").arg(lo, 0, 'f', 1).arg(hi, 0, 'f', 1);
            const int cert_w = (x_val + w_val) - x_spark;
            p.drawText(QRect(x_spark, 0, cert_w, hh), Qt::AlignVCenter | Qt::AlignRight, cert);
            // The window span labels the sparkline's x extent, at its left edge — but only when it can
            // sit there without crowding the heading it shares the group with.
            const QString span = QStringLiteral("%1 s").arg(window_s_, 0, 'f', 0);
            if (fm.horizontalAdvance(span) + fm.horizontalAdvance(cert) + 2 * pad <= cert_w)
                p.drawText(QRect(x_spark, 0, cert_w, hh), Qt::AlignVCenter | Qt::AlignLeft, span);

            p.drawText(QRect(x_pbar, 0, (x_pnum + w_pnum) - x_pbar, hh),
                       Qt::AlignVCenter | Qt::AlignRight, kExistHdr());
            p.drawText(QRect(x_age, 0, w_age, hh), Qt::AlignVCenter | Qt::AlignRight,
                       QStringLiteral("age"));
        }

        if (ordered.empty())
        {
            p.setPen(QColor(150, 150, 150));
            p.drawText(QRect(pad, header_h(), width() - 2 * pad, row_h()),
                       Qt::AlignVCenter | Qt::AlignLeft, QStringLiteral("no instances"));
            return;
        }

        // ── Rows ────────────────────────────────────────────────────────────────────────────────
        int y = header_h();
        for (const Track* tr : ordered)
        {
            const QRect line(0, y, width(), row_h());
            const bool  dead = not tr->alive;
            const bool  cold = not tr->row.initialized;

            p.setPen(QColor(232, 232, 232));
            p.drawLine(line.left() + pad, line.bottom(), line.right() - pad, line.bottom());

            const QColor ink = dead ? QColor(170, 170, 170) : (cold ? QColor(130, 130, 130) : QColor(40, 40, 40));

            p.setPen(ink);
            p.drawText(QRect(x_name, y, w_name, row_h()), Qt::AlignVCenter | Qt::AlignLeft, name_text(*tr));

            if (w_spark > 8)
                paint_spark(p, QRect(x_spark, y + 3, w_spark, row_h() - 6), *tr, t0, t, lo, hi,
                            surp_hi, use_gap, dead);

            p.setPen(ink);
            p.drawText(QRect(x_val, y, w_val, row_h()), Qt::AlignVCenter | Qt::AlignRight,
                       value_text(*tr, use_gap));

            // Existence bar — the second question a human asks ("is it still there?"), so it gets its
            // own channel rather than being folded into the certainty trace.
            const float pe = tr->row.p_exists;
            const QRect bar(x_pbar, y + row_h() / 2 - fm.height() / 3, w_pbar, 2 * fm.height() / 3);
            p.setPen(Qt::NoPen);
            p.fillRect(bar, QColor(226, 226, 226));
            if (std::isfinite(pe))
            {
                const int fw = static_cast<int>(std::lround(std::clamp(pe, 0.0f, 1.0f) * bar.width()));
                p.fillRect(QRect(bar.left(), bar.top(), fw, bar.height()),
                           dead ? QColor(200, 200, 200) : QColor(96, 150, 96));
            }
            p.setPen(ink);
            p.drawText(QRect(x_pnum, y, w_pnum, row_h()), Qt::AlignVCenter | Qt::AlignRight,
                       std::isfinite(pe) ? QString::number(pe, 'f', 2) : dash());
            p.drawText(QRect(x_age, y, w_age, row_h()), Qt::AlignVCenter | Qt::AlignRight, age_text(*tr, t));

            y += row_h();
        }
    }

private:
    struct Sample { float t, gap, logdet, p_exists, surprise; };
    struct Track
    {
        BeliefStripRow     row;
        std::deque<Sample> samples;
        float              first_seen_s = 0.0f;
        float              last_seen    = 0.0f;
        bool               alive        = true;
        std::uint64_t      order        = 0;   // arrival order — tiebreak for unstamped nodes
    };

    // ── Painting helpers ────────────────────────────────────────────────────────────────────────

    // The certainty trace, plus the surprise trace beneath it on its own scale.
    void paint_spark(QPainter& p, const QRect& box, const Track& tr, float t0, float t1,
                     float lo, float hi, float surp_hi, bool use_gap, bool dead) const
    {
        p.setPen(Qt::NoPen);
        p.fillRect(box, QColor(245, 245, 245));

        const auto x_of = [&](float t) { return box.left() + (t - t0) / std::max(1e-3f, t1 - t0) * box.width(); };
        const auto y_of = [&](float v)
        {
            const float f = std::clamp((v - lo) / std::max(1e-6f, hi - lo), 0.0f, 1.0f);
            return box.bottom() - f * box.height();
        };

        // "Adequate" floor: with the gap channel, the bottom edge IS gap=0 — mark it, because a trace
        // resting on it means DONE, which is a different statement from "not changing".
        if (use_gap)
        {
            p.setPen(QPen(QColor(200, 220, 200), 1, Qt::DashLine));
            p.drawLine(box.left(), box.bottom(), box.right(), box.bottom());
        }

        // Surprise first, so the certainty trace draws over it.
        if (surp_hi > 0.0f)
        {
            QPolygonF sp;
            for (const Sample& s : tr.samples)
            {
                if (s.t < t0 or not std::isfinite(s.surprise)) continue;
                const float f = std::clamp(s.surprise / surp_hi, 0.0f, 1.0f);
                sp << QPointF(x_of(s.t), box.bottom() - f * box.height() * 0.45f);
            }
            if (sp.size() > 1)
            {
                p.setPen(QPen(QColor(205, 160, 110), 1));
                p.drawPolyline(sp);
            }
        }

        // Certainty: filled area under the trace. Gaps in the data (a dead stretch, or a channel the
        // instance does not publish) BREAK the polyline rather than interpolate across them.
        QPolygonF run;
        const auto flush = [&]()
        {
            if (run.size() > 1)
            {
                QPolygonF fill = run;
                fill << QPointF(run.back().x(), box.bottom()) << QPointF(run.front().x(), box.bottom());
                p.setPen(Qt::NoPen);
                p.setBrush(dead ? QColor(215, 215, 215, 120) : QColor(90, 120, 175, 60));
                p.drawPolygon(fill);
                p.setBrush(Qt::NoBrush);
                p.setPen(QPen(dead ? QColor(185, 185, 185) : QColor(60, 95, 160), 1.4));
                p.drawPolyline(run);
            }
            run.clear();
        };
        for (const Sample& s : tr.samples)
        {
            if (s.t < t0) continue;
            const float v = use_gap ? s.gap : s.logdet;
            if (not std::isfinite(v) or (use_gap and v < 0.0f)) { flush(); continue; }
            run << QPointF(x_of(s.t), y_of(v));
        }
        flush();

        p.setPen(QColor(226, 226, 226));
        p.setBrush(Qt::NoBrush);
        p.drawRect(box.adjusted(0, 0, -1, -1));
    }

    QString name_text(const Track& tr) const
    {
        QString s = QString::fromStdString(tr.row.node);
        if (not tr.alive)              s += QStringLiteral(" (gone)");
        else if (not tr.row.model.empty()) s += QStringLiteral(" ") + QString::fromStdString(tr.row.model);
        return s;
    }

    QString value_text(const Track& tr, bool use_gap) const
    {
        if (not tr.alive) return dash();
        const float v = use_gap ? tr.row.gap_nats : tr.row.logdet_nats;
        if (not std::isfinite(v) or (use_gap and v < 0.0f)) return dash();
        return QString::number(v, 'f', 2);
    }

    // Age since BIRTH (the node's creation stamp), not since the last belief touch: the question this
    // column answers is "how long has this thing been in the graph".
    QString age_text(const Track& tr, float t) const
    {
        const float age = tr.row.birth_ms != 0
                        ? static_cast<float>(now_ms() - std::min(now_ms(), tr.row.birth_ms)) / 1000.0f
                        : t - tr.first_seen_s;
        if (age < 60.0f)   return QStringLiteral("%1s").arg(age, 0, 'f', 0);
        if (age < 3600.0f) return QStringLiteral("%1m%2").arg(static_cast<int>(age) / 60)
                                                         .arg(static_cast<int>(age) % 60, 2, 10, QLatin1Char('0'));
        return QStringLiteral("%1h%2").arg(static_cast<int>(age) / 3600)
                                      .arg((static_cast<int>(age) % 3600) / 60, 2, 10, QLatin1Char('0'));
    }

    static QString dash() { return QStringLiteral("–"); }
    // Heading for the existence group. Named once: it also SIZES that group's bar (see w_pbar).
    static QString kExistHdr() { return QStringLiteral("p(existence)"); }

    static std::uint64_t now_ms()
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    }

    // ── Bookkeeping ─────────────────────────────────────────────────────────────────────────────

    bool any_gap() const
    {
        for (const auto& [name, tr] : tracks_)
            if (tr.alive and tr.row.gap_nats >= 0.0f) return true;
        return false;
    }

    void prune(float t)
    {
        const float keep_from = t - 1.2f * window_s_;
        for (auto it = tracks_.begin(); it != tracks_.end();)
        {
            Track& tr = it->second;
            while (not tr.samples.empty() and tr.samples.front().t < keep_from) tr.samples.pop_front();
            const bool expired = not tr.alive and (t - tr.last_seen) > linger_s_;
            if (expired or tr.samples.empty()) it = tracks_.erase(it);
            else ++it;
        }
    }

    float elapsed_s() const
    {
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - t_origin_).count();
    }

    int row_h() const    { return fontMetrics().height() + 8; }
    int header_h() const { return fontMetrics().height() + 6; }

    QString title_;
    std::unordered_map<std::string, Track> tracks_;
    std::chrono::steady_clock::time_point  t_origin_ = std::chrono::steady_clock::now();
    float         window_s_    = 60.0f;
    float         linger_s_    = 6.0f;
    std::uint64_t next_order_  = 0;
    int           height_hint_ = 0;
};

}  // namespace rc
