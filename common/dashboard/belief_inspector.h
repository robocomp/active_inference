#pragma once

/*
 * belief_inspector.h  —  live per-instance BELIEF inspector for the concept agents
 *
 * SHARED, header-only dashboard widget (mirrors dashboard/evidence_monitor.h: header-only, NO Q_OBJECT,
 * built + updated on the Qt/GUI thread, which in these agents is the compute() thread). It replaces the old
 * bottom "posterior σ" time-series panel with the WHOLE belief, at a glance:
 *   • every CONTINUOUS state DOF — posterior mean, marginal σ, the consumer's precision demand σ*, and the
 *     remaining adequacy gap ½·ln(σ²/σ*²) as a bar (the same quantity the epistemic planner maximises);
 *   • the posterior covariance Σ rendered as a CORRELATION heatmap ρ_ij = Σ_ij/√(Σ_ii Σ_jj) — the structure
 *     the σ column cannot show (which DOFs are jointly unresolved, e.g. w↔yaw on a grazing view);
 *   • the DISCRETE-mode posteriors (shape round/square, cabinet tier, yaw flip) as bars;
 *   • the scalar gauges — free energy + its baseline + the surprise, existence log-odds, belief age.
 *
 * Agent-agnostic BY CONSTRUCTION: it knows nothing about tables/cabinets/bottles, only about a vector of
 * BeliefDof + a flat NxN covariance + a vector of BeliefModeBar. That one API serves N=3 (chair), N=5
 * (bottle, cabinet wall-run), N=6 (refrigerator) and N=7 (table, cabinet box).
 *
 * Pure view: the agent fills vector<BeliefCard> and calls update_view() (throttle it to a few Hz). No DSR,
 * no Eigen, no torch — just Qt.
 */

#include <QWidget>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QPainter>
#include <QPaintEvent>
#include <QFontMetrics>
#include <QLabel>
#include <QColor>
#include <QFont>
#include <QString>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace rc
{

// ─── View model ──────────────────────────────────────────────────────────────────────────────────

// One CONTINUOUS state DOF = one row of a card's left table.
struct BeliefDof
{
    std::string name;              // "cx", "yaw", "t", "t0" …
    std::string unit;              // "m" / "rad" / "" — drawn after the value
    float value      = 0.0f;       // posterior mean, in `unit`
    float sigma      = 0.0f;       // marginal posterior std sqrt(Sigma_ii), in `unit`
    // The consumer's precision demand. < 0 ⇒ this DOF has NO published σ*: the σ*/adequacy/gap cells render
    // "–" and the DOF is excluded from the gap sum. NEVER invent one (CLAUDE.md: no thresholds) — an agent
    // whose planner publishes no σ* (chair, bottle) passes -1 on every DOF and the whole column group is
    // dropped, which is honest; a made-up target would silently become a tuning knob.
    float sigma_star = -1.0f;
};

// One bar of a discrete-mode posterior. Bars sharing `group` are drawn on ONE footer line.
struct BeliefModeBar
{
    std::string group;   // "shape" | "tier" | "yaw" | "w<->h" | "front"
    std::string label;   // "round" | "square" | "base" | "wall" | "keep" | "swap" …
    float       p = 0.0f;   // posterior mass in [0,1]
};

// Scalar gauges (the card's header strip). Non-finite / negative sentinels render "–", never a fake zero.
struct BeliefScalars
{
    float fe            = std::numeric_limits<float>::quiet_NaN();  // free energy of the last fit
    float fe_baseline   = -1.0f;                                    // <0 = uninitialised (no baseline yet)
    float fe_surprise   = 0.0f;                                     // smoothed FE − baseline (attention signal)
    float logodds       = std::numeric_limits<float>::quiet_NaN();  // NaN ⇒ agent has no existence belief
    float p_exists      = std::numeric_limits<float>::quiet_NaN();
    float age_s         = -1.0f;   // seconds since the belief was last touched; <0 ⇒ never
    int   remove_streak = 0;       // removal-decision debounce streak
    int   since_det     = 0;       // frames since the last fresh detection
    bool  initialized   = false;   // belief live? false ⇒ the card is tinted grey
};

// One instance's belief snapshot = one CARD.
struct BeliefCard
{
    std::string node;                       // "table_0" / the kitchen cell id
    std::string model;                      // optional tag: "round" / "wall-run base" — "" = none
    BeliefScalars              s;
    std::vector<BeliefDof>     dofs;        // N rows, in Sigma index order
    std::vector<float>         cov;         // N*N ROW-MAJOR Sigma (same order); size != N*N ⇒ heatmap hidden
    // Axis labels for `cov` when the covariance lives in a DIFFERENT chart than the DOF rows (the table's
    // C2v symmetry quotient: rows are w/h/yaw, Σ's off-diagonals are s/a1/a2). Empty ⇒ use dofs[i].name.
    std::vector<std::string>   cov_labels;
    std::vector<BeliefModeBar> modes;       // may be empty (bottle) ⇒ the mode footer is omitted
};

// ─── Per-instance card painter ───────────────────────────────────────────────────────────────────

class BeliefCardView : public QWidget
{
public:
    explicit BeliefCardView(QWidget* parent = nullptr) : QWidget(parent)
    {
        QFont f = font();
        f.setPointSizeF(f.pointSizeF() - 0.5);
        f.setFamily(QStringLiteral("monospace"));   // the columns are numeric; a mono face keeps them aligned
        setFont(f);
    }

    // Store one snapshot, re-derive the fixed height from it, and schedule a repaint.
    void set_card(const BeliefCard& c)
    {
        card_ = c;
        setFixedHeight(preferred_height());
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        const QFontMetrics fm = fontMetrics();
        const int row_h = fm.height() + 4;
        const int ch    = std::max(1, fm.horizontalAdvance(QLatin1Char('0')));
        const int pad   = ch;

        // ── Card chrome ────────────────────────────────────────────────────────────────────────
        p.fillRect(rect(), QColor(250, 250, 250));
        p.setPen(QColor(180, 180, 180));
        p.drawRect(rect().adjusted(0, 0, -1, -1));

        int y = pad;

        // ── Header strip: node, model tag, FE, existence, age ──────────────────────────────────
        // Tint carries the same semantics as the EvidenceMonitor row tint above it, so the two views
        // agree at a glance: red = removing, grey = belief not live, amber = stale, green = healthy.
        p.fillRect(QRect(1, 1, width() - 2, row_h), header_tint());
        p.setPen(ink());
        {
            QFont hf = p.font(); hf.setBold(true); p.setFont(hf);
            const QString title = QString::fromStdString(card_.node)
                                + (card_.model.empty() ? QString()
                                                       : QStringLiteral("  (%1)").arg(QString::fromStdString(card_.model)));
            p.drawText(QRect(pad, y, width() - 2 * pad, row_h), Qt::AlignVCenter | Qt::AlignLeft, title);
            hf.setBold(false); p.setFont(hf);

            QString gauges = QStringLiteral("FE %1").arg(num(card_.s.fe, 2));
            if (card_.s.fe_baseline >= 0.0f)
                gauges += QStringLiteral(" (base %1, surpr %2)")
                              .arg(num(card_.s.fe_baseline, 2))
                              .arg(num(card_.s.fe_surprise, 2, true));
            gauges += QStringLiteral("   p(exist) %1").arg(num(card_.s.p_exists, 2));
            if (std::isfinite(card_.s.logodds))
                gauges += QStringLiteral(" (L %1)").arg(num(card_.s.logodds, 1, true));
            gauges += card_.s.age_s >= 0.0f ? QStringLiteral("   age %1s").arg(num(card_.s.age_s, 1))
                                            : QStringLiteral("   age ") + dash();   // no unit on "unknown"
            if (card_.s.remove_streak > 0) gauges += QStringLiteral("   strk %1").arg(card_.s.remove_streak);
            p.drawText(QRect(pad, y, width() - 2 * pad, row_h), Qt::AlignVCenter | Qt::AlignRight, gauges);
        }
        y += row_h;

        // ── Column geometry ────────────────────────────────────────────────────────────────────
        // Widths come from the widest FORMATTED sample in each column, so nothing is a pixel constant.
        const bool show_star = has_any_star();
        int w_name = fm.horizontalAdvance(QStringLiteral("DOF"));
        int w_val  = fm.horizontalAdvance(QStringLiteral("value"));
        int w_sig  = fm.horizontalAdvance(QStringLiteral("σ"));
        int w_star = show_star ? fm.horizontalAdvance(QStringLiteral("σ*")) : 0;
        for (const auto& d : card_.dofs)
        {
            w_name = std::max(w_name, fm.horizontalAdvance(QString::fromStdString(d.name)));
            w_val  = std::max(w_val,  fm.horizontalAdvance(fmt_value(d)));
            w_sig  = std::max(w_sig,  fm.horizontalAdvance(fmt_sigma(d)));
            if (show_star) w_star = std::max(w_star, fm.horizontalAdvance(fmt_star(d)));
        }
        const int w_bar = show_star ? 10 * ch : 0;   // adequacy track
        const int w_gap = show_star ? std::max(fm.horizontalAdvance(QStringLiteral("adequacy")),
                                               fm.horizontalAdvance(QStringLiteral("0.00"))) : 0;

        const int x_name = pad;
        const int x_val  = x_name + w_name + pad;
        const int x_sig  = x_val  + w_val  + pad;
        const int x_star = x_sig  + w_sig  + pad;
        const int x_bar  = x_star + w_star + (show_star ? pad : 0);
        const int x_gap  = x_bar  + w_bar  + (show_star ? pad : 0);
        const int x_left_end = x_gap + w_gap;
        const int x_split    = x_left_end + 2 * pad;

        // ── Column header ──────────────────────────────────────────────────────────────────────
        p.setPen(QColor(110, 110, 110));
        p.drawText(QRect(x_name, y, w_name, row_h), Qt::AlignVCenter | Qt::AlignLeft,  QStringLiteral("DOF"));
        p.drawText(QRect(x_val,  y, w_val,  row_h), Qt::AlignVCenter | Qt::AlignRight, QStringLiteral("value"));
        p.drawText(QRect(x_sig,  y, w_sig,  row_h), Qt::AlignVCenter | Qt::AlignRight, QStringLiteral("σ"));
        if (show_star)
        {
            p.drawText(QRect(x_star, y, w_star, row_h), Qt::AlignVCenter | Qt::AlignRight, QStringLiteral("σ*"));
            p.drawText(QRect(x_bar,  y, w_bar + pad + w_gap, row_h), Qt::AlignVCenter | Qt::AlignLeft,
                       QStringLiteral("adequacy"));
        }
        const int y_rows = y + row_h;   // ← the heatmap's first row starts HERE too (see below)

        // ── Heatmap geometry ───────────────────────────────────────────────────────────────────
        // The matrix row pitch is EXACTLY row_h and its first row starts at y_rows, so matrix row i lines
        // up with DOF i's name on the left — that alignment is what removes the need for row labels.
        const int n = static_cast<int>(card_.dofs.size());
        const bool have_cov = n > 0 and static_cast<int>(card_.cov.size()) == n * n;
        const int avail = width() - x_split - pad;
        const int cell  = (n > 0 and avail > 0) ? std::min(row_h, avail / n) : 0;
        const bool show_cov = have_cov and cell >= 6;   // too narrow to read ⇒ drop it rather than clip

        // The heatmap's own axis labels occupy this same row (they must sit directly above matrix row 0),
        // so the caption goes to the RIGHT of the matrix — and only when it actually fits.
        if (show_cov)
        {
            const QString cap  = QStringLiteral("Σ correlation");
            const int     xcap = x_split + n * cell + pad;
            if (xcap + fm.horizontalAdvance(cap) <= width() - pad)
            {
                p.setPen(QColor(110, 110, 110));
                p.drawText(QRect(xcap, y, width() - pad - xcap, row_h), Qt::AlignVCenter | Qt::AlignLeft, cap);
            }
        }

        // ── DOF rows ───────────────────────────────────────────────────────────────────────────
        y = y_rows;
        float gap_total = 0.0f;
        for (const auto& d : card_.dofs)
        {
            p.setPen(ink());
            p.drawText(QRect(x_name, y, w_name, row_h), Qt::AlignVCenter | Qt::AlignLeft,  QString::fromStdString(d.name));
            p.drawText(QRect(x_val,  y, w_val,  row_h), Qt::AlignVCenter | Qt::AlignRight, fmt_value(d));
            p.drawText(QRect(x_sig,  y, w_sig,  row_h), Qt::AlignVCenter | Qt::AlignRight, fmt_sigma(d));

            if (show_star)
            {
                p.drawText(QRect(x_star, y, w_star, row_h), Qt::AlignVCenter | Qt::AlignRight, fmt_star(d));

                const QRect track(x_bar, y + row_h / 4, w_bar, row_h / 2);
                if (d.sigma_star >= 0.0f)
                {
                    // Remaining information to carry this DOF down to σ*, in nats — the SAME per-DOF term the
                    // epistemic planner sums into its adequacy gap: ½·ln(σ²/σ*²) = ln(σ/σ*), clamped at 0.
                    const float g = std::max(0.0f, std::log(std::max(1e-9f, d.sigma)
                                                          / std::max(1e-9f, d.sigma_star)));
                    gap_total += g;
                    // Fill fraction is a RATIO of two quantities that already exist — no new constant: the bar
                    // is FULL exactly when σ ≤ σ* (adequate) and shrinks as 1/σ.
                    const float f = std::clamp(d.sigma_star / std::max(d.sigma, d.sigma_star), 0.0f, 1.0f);
                    p.fillRect(track, QColor(225, 225, 225));
                    QRect fill = track; fill.setWidth(static_cast<int>(track.width() * f));
                    p.fillRect(fill, g <= 0.0f ? QColor(120, 190, 120) : QColor(230, 170, 70));
                    p.setPen(ink());
                    p.drawText(QRect(x_gap, y, w_gap, row_h), Qt::AlignVCenter | Qt::AlignRight,
                               g <= 0.0f ? QStringLiteral("ok") : QString::asprintf("%.2f", g));
                }
                else
                {   // no published demand for this DOF — say so, don't draw an empty bar that reads as "0%"
                    p.setPen(QPen(QColor(190, 190, 190), 1, Qt::DashLine));
                    p.setBrush(Qt::NoBrush);
                    p.drawRect(track);
                    p.setPen(ink());
                    p.drawText(QRect(x_gap, y, w_gap, row_h), Qt::AlignVCenter | Qt::AlignRight, dash());
                }
            }
            y += row_h;
        }

        // ── Σ gap total ────────────────────────────────────────────────────────────────────────
        if (show_star)
        {
            p.setPen(QColor(90, 90, 90));
            p.drawText(QRect(x_name, y, x_left_end - x_name, row_h), Qt::AlignVCenter | Qt::AlignRight,
                       QStringLiteral("Σ gap = %1 nats").arg(QString::asprintf("%.2f", gap_total)));
        }

        // ── Correlation heatmap ────────────────────────────────────────────────────────────────
        if (show_cov)
        {
            p.setRenderHint(QPainter::Antialiasing, false);   // crisp cells

            // Column labels, elided to whatever the cell width affords (usually 2-3 chars).
            p.setPen(QColor(110, 110, 110));
            for (int j = 0; j < n; ++j)
            {
                const std::string& lbl = (static_cast<int>(card_.cov_labels.size()) == n)
                                       ? card_.cov_labels[j] : card_.dofs[j].name;
                p.drawText(QRect(x_split + j * cell, y_rows - row_h, cell, row_h),
                           Qt::AlignVCenter | Qt::AlignHCenter,
                           fm.elidedText(QString::fromStdString(lbl), Qt::ElideRight, cell));
            }

            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    const QRect cr(x_split + j * cell, y_rows + i * cell, cell - 1, cell - 1);
                    if (i == j) { p.fillRect(cr, QColor(90, 90, 90)); continue; }   // ρ_ii ≡ 1: no information
                    const float vii = card_.cov[i * n + i], vjj = card_.cov[j * n + j];
                    float rho = (vii > 0.0f and vjj > 0.0f)
                              ? card_.cov[i * n + j] / std::sqrt(vii * vjj)
                              : std::numeric_limits<float>::quiet_NaN();
                    if (not std::isfinite(rho))
                    {   // a degenerate/unobserved DOF must read as UNKNOWN, not as ρ=0
                        p.fillRect(cr, QBrush(QColor(200, 200, 200), Qt::BDiagPattern));
                        continue;
                    }
                    p.fillRect(cr, rho_colour(std::clamp(rho, -1.0f, 1.0f)));
                }

            // Legend strip −1 ─ 0 ─ +1, anchored to the Σ-gap band (`y`) so it stays inside the ONE row the
            // height calculation reserves — cell ≤ row_h, so the matrix always ends at or above `y`.
            const int w_leg = n * cell;
            const int h_leg = std::max(3, row_h / 3);
            for (int k = 0; k < 24; ++k)
            {
                const float t = -1.0f + 2.0f * static_cast<float>(k) / 23.0f;
                p.fillRect(QRect(x_split + k * w_leg / 24, y, w_leg / 24 + 1, h_leg), rho_colour(t));
            }
            p.setRenderHint(QPainter::Antialiasing, true);
            p.setPen(QColor(110, 110, 110));
            const QRect leg_txt(x_split, y + h_leg, w_leg, row_h - h_leg);
            p.drawText(leg_txt, Qt::AlignVCenter | Qt::AlignLeft,  QStringLiteral("-1"));
            p.drawText(leg_txt, Qt::AlignVCenter | Qt::AlignRight, QStringLiteral("+1"));
        }
        y += row_h;

        // ── Discrete-mode footer: one line per group, in first-appearance order ─────────────────
        for (const auto& grp : mode_groups())
        {
            p.setPen(ink());
            const QString gname = QString::fromStdString(grp);
            p.drawText(QRect(x_name, y, width() - x_name, row_h), Qt::AlignVCenter | Qt::AlignLeft, gname);
            // Group names ("shape", "w<->h") are wider than the DOF-name column, so start the bars past
            // whichever is longer rather than letting the name run under the first label.
            int x = std::max(x_val, x_name + fm.horizontalAdvance(gname) + pad);
            for (const auto& m : card_.modes)
            {
                if (m.group != grp) continue;
                const QString lbl = QString::fromStdString(m.label);
                p.setPen(ink());
                p.drawText(QRect(x, y, fm.horizontalAdvance(lbl), row_h), Qt::AlignVCenter | Qt::AlignLeft, lbl);
                x += fm.horizontalAdvance(lbl) + ch / 2;
                const QRect track(x, y + row_h / 4, 8 * ch, row_h / 2);
                p.fillRect(track, QColor(225, 225, 225));
                QRect fill = track; fill.setWidth(static_cast<int>(track.width() * std::clamp(m.p, 0.0f, 1.0f)));
                p.fillRect(fill, QColor(90, 140, 200));
                x += track.width() + ch / 2;
                const QString pv = QString::asprintf("%.2f", m.p);
                p.drawText(QRect(x, y, fm.horizontalAdvance(pv), row_h), Qt::AlignVCenter | Qt::AlignLeft, pv);
                x += fm.horizontalAdvance(pv) + 2 * ch;
            }
            y += row_h;
        }
    }

private:
    // Explicit dark ink on pastel fills: readable in either app theme (same rule as evidence_monitor.h).
    // Functions, not constexpr statics — QColor/QString are not literal types across all supported Qt builds.
    static QColor  ink()  { return QColor(20, 20, 20); }
    static QString dash() { return QStringLiteral("–"); }

    BeliefCard card_;

    // Distinct mode groups, in first-appearance order.
    std::vector<std::string> mode_groups() const
    {
        std::vector<std::string> g;
        for (const auto& m : card_.modes)
            if (std::ranges::find(g, m.group) == g.end())
                g.push_back(m.group);
        return g;
    }

    bool has_any_star() const
    {
        return std::ranges::any_of(card_.dofs, [](const BeliefDof& d) { return d.sigma_star >= 0.0f; });
    }

    QColor header_tint() const
    {
        if (card_.s.remove_streak > 0) return QColor(245, 200, 200);
        if (not card_.s.initialized)   return QColor(228, 228, 228);
        if (card_.s.since_det > 30)    return QColor(250, 235, 190);
        return QColor(205, 238, 205);
    }

    int preferred_height() const
    {
        const QFontMetrics fm = fontMetrics();
        const int row_h = fm.height() + 4;
        const int pad   = std::max(1, fm.horizontalAdvance(QLatin1Char('0')));
        const int n     = static_cast<int>(card_.dofs.size());
        // header + column header + N rows + (Σ gap / legend) + one line per mode group
        int rows = 3 + n + static_cast<int>(mode_groups().size());
        // The heatmap occupies exactly N cells of pitch ≤ row_h plus a legend, so it never exceeds the
        // left column's height — no extra allowance needed.
        return pad * 2 + rows * row_h;
    }

    // Diverging RdBu, white at zero. Opaque endpoints so it reads under either app theme.
    static QColor rho_colour(float rho)
    {
        const float a = std::clamp(std::fabs(rho), 0.0f, 1.0f);
        const int r0 = 247, g0 = 247, b0 = 247;
        const int r1 = rho >= 0.0f ? 202 :   5;
        const int g1 = rho >= 0.0f ?   0 : 113;
        const int b1 = rho >= 0.0f ?  32 : 176;
        return QColor(static_cast<int>(r0 + a * (r1 - r0)),
                      static_cast<int>(g0 + a * (g1 - g0)),
                      static_cast<int>(b0 + a * (b1 - b0)));
    }

    // "–" for a non-finite scalar — an unavailable quantity must never print as 0.
    static QString num(float v, int prec, bool signed_ = false)
    {
        if (not std::isfinite(v) or v < 0.0f) return dash();
        return signed_ ? QString::asprintf("%+.*f", prec, static_cast<double>(v))
                       : QString::asprintf("%.*f",  prec, static_cast<double>(v));
    }

    static QString fmt_value(const BeliefDof& d)
    {
        return QString::asprintf("%.3f", static_cast<double>(d.value))
             + (d.unit.empty() ? QString() : QStringLiteral(" ") + QString::fromStdString(d.unit));
    }
    static QString fmt_sigma(const BeliefDof& d)
    {
        return std::isfinite(d.sigma) ? QString::asprintf("±%.3f", static_cast<double>(d.sigma)) : dash();
    }
    static QString fmt_star(const BeliefDof& d)
    {
        return d.sigma_star >= 0.0f ? QString::asprintf("%.3f", static_cast<double>(d.sigma_star)) : dash();
    }
};


// ─── Scrollable stack of per-instance cards ──────────────────────────────────────────────────────

class BeliefInspector : public QWidget
{
public:
    explicit BeliefInspector(const QString& title = QStringLiteral("belief inspector"),
                             QWidget* parent = nullptr) : QWidget(parent)
    {
        setWindowTitle(title);
        auto* outer = new QVBoxLayout(this);
        outer->setContentsMargins(2, 2, 2, 2);
        outer->setSpacing(2);

        scroll_ = new QScrollArea(this);
        scroll_->setWidgetResizable(true);
        scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        inner_  = new QWidget(scroll_);
        stack_  = new QVBoxLayout(inner_);
        stack_->setContentsMargins(2, 2, 2, 2);
        stack_->setSpacing(4);
        stack_->addStretch(1);           // keeps short stacks top-aligned
        scroll_->setWidget(inner_);
        outer->addWidget(scroll_);
    }

    // Rebuild the card stack from one cycle's snapshot (throttle to a few Hz). Grows a POOL of card views;
    // surplus views are hidden, never deleted — so a removed instance's card simply disappears next refresh
    // and there is no per-series pruning bookkeeping to keep in sync.
    void update_view(const std::vector<BeliefCard>& cards)
    {
        while (views_.size() < cards.size())
        {
            auto* v = new BeliefCardView(inner_);
            stack_->insertWidget(static_cast<int>(views_.size()), v);   // before the trailing stretch
            views_.push_back(v);
        }
        for (std::size_t i = 0; i < views_.size(); ++i)
        {
            if (i < cards.size()) { views_[i]->set_card(cards[i]); views_[i]->show(); }
            else                  { views_[i]->hide(); }
        }
    }

private:
    QScrollArea* scroll_ = nullptr;
    QWidget*     inner_  = nullptr;
    QVBoxLayout* stack_  = nullptr;
    std::vector<BeliefCardView*> views_;
};

}  // namespace rc
