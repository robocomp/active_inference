#pragma once

/*
 * controller_affordance_flow.h — the affordance codelet drawn as a FLOW CHART.
 *
 * WHY NOT A LIST. A list renders a sequence and a loop identically, and this program is not a sequence:
 * the pipeline (claim → navigate → align) runs once and ENTERS a servo LOOP that oscillates settle ⇄
 * step until an AND-GATE over the contract's clauses holds for stable_n consecutive measurements, or
 * the timeout kills it. The loop and the gate are the two things worth seeing, and they are exactly the
 * two a list cannot show. So: a spine of pipeline nodes, then a box containing the loop with the clause
 * chips feeding a gate.
 *
 * DYNAMIC WITHOUT A TIMER. Everything animated is driven by values already in the snapshot — the active
 * node pulses on `elapsed_s`, the loop glyph rotates with the phase, the timeout drains. The panel is
 * repainted at the control rate whenever it is visible, so no QTimer is needed and nothing keeps
 * spinning when the window is closed.
 *
 * Header-only, NO Q_OBJECT, painted on the GUI thread — the fleet dashboard convention.
 *
 * ★IT PAINTS ITS OWN GROUND. Every colour below was chosen against a DARK background, but the widget
 * used to draw straight onto the theme's palette — light — so 0xdd label text landed on a near-white
 * fill and the whole chart read as washed-out grey. A widget whose palette assumes a ground must PAINT
 * that ground; inheriting the theme's is how a legible design becomes an unreadable one on a different
 * desktop. Labels are bold unconditionally (liveness is carried by colour and the pulse, which do not
 * have to compete with weight), and the secondary greys are lifted well clear of the ground.
 */

#include "controller_affordance_view.h"

#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QWidget>

#include <cmath>

namespace rc
{

class AffordanceFlow : public QWidget
{
public:
    explicit AffordanceFlow(QWidget *parent = nullptr) : QWidget(parent)
    {
        setMinimumHeight(220);
    }

    void set_view(const AffordanceExecution &v)
    {
        view_ = v;
        // The scroll area is widgetResizable, so it squashes the child to the viewport unless the
        // child's MINIMUM says otherwise. Without this a contract with several clauses would be
        // compressed instead of scrolled.
        setMinimumHeight(sizeHint().height());
        updateGeometry();
        update();
    }

    QSize sizeHint() const override
    {
        int n_pipeline = 0, n_inner = 0;
        for (const auto &s : view_.steps)
            (s.kind == AffordanceStepView::Kind::Pipeline ? n_pipeline : n_inner)++;
        return {620, 30 + n_pipeline * (kNodeH + kGap) + (n_inner ? 34 + n_inner * kChipH + 18 : 0) + 30};
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), kGround);
        // One step up from the widget's default: this chart is read at a glance while the robot moves.
        QFont base = p.font();
        base.setPointSizeF(base.pointSizeF() + 0.5);
        p.setFont(base);

        if (not view_.active and view_.steps.empty())
        {
            p.setPen(kMuted);
            p.drawText(rect(), Qt::AlignCenter,
                       QStringLiteral("no affordance has run yet"));
            return;
        }

        const int x = 16;
        const int w = width() - 2 * x;
        int y = 12;

        // ── THE PIPELINE SPINE ───────────────────────────────────────────────────────────────────
        const AffordanceStepView *prev = nullptr;
        for (const auto &s : view_.steps)
        {
            if (s.kind != AffordanceStepView::Kind::Pipeline) continue;
            if (prev != nullptr) { draw_edge(p, x + 18, y - kGap, y, is_past(*prev)); }
            draw_node(p, QRect(x, y, w, kNodeH), s, /*rounded=*/12);
            y += kNodeH + kGap;
            prev = &s;
        }

        // ── THE SERVO LOOP BOX ───────────────────────────────────────────────────────────────────
        const AffordanceStepView *servo = find(AffordanceStepView::Kind::ServoLoop);
        if (servo != nullptr)
        {
            if (prev != nullptr) draw_edge(p, x + 18, y - kGap, y, is_past(*prev));
            int inner = 0;
            for (const auto &s : view_.steps)
                if (s.kind == AffordanceStepView::Kind::Clause or s.kind == AffordanceStepView::Kind::Stable)
                    ++inner;
            const int box_h = 34 + inner * kChipH + 14;
            const QRect box(x, y, w, box_h);

            const bool live = servo->state == AffordanceStepView::State::Active;
            p.setPen(QPen(colour(*servo), live ? 2.0 : 1.0, live ? Qt::SolidLine : Qt::DashLine));
            p.setBrush(QColor(colour(*servo).red(), colour(*servo).green(), colour(*servo).blue(), 18));
            p.drawRoundedRect(box, 10, 10);

            // Header: the loop glyph (rotating with the phase — the ONE thing that says "this is a
            // loop, and it is going round"), the label, and the timeout draining left to right.
            const int cx = box.left() + 22, cy = box.top() + 18;
            draw_loop_glyph(p, cx, cy, colour(*servo), live ? view_.elapsed_s : 0.f);
            p.setPen(kLabel);
            QFont f = p.font(); f.setBold(true); p.setFont(f);
            p.drawText(box.left() + 40, box.top() + 23, QString::fromStdString(servo->label));
            f.setBold(false); p.setFont(f);
            p.setPen(kDetail);
            const QString right = QString::fromStdString(
                servo->blocked_why.empty() ? servo->detail : servo->blocked_why);
            if (not right.isEmpty())
            {
                p.setPen(servo->blocked_why.empty() ? kDetail : kAlert);
                p.drawText(QRect(box.left() + 190, box.top() + 8, box.width() - 200, 18),
                           Qt::AlignLeft | Qt::AlignVCenter, right);
            }
            if (view_.timeout_s > 0.f)
            {
                const float left_frac = std::clamp(1.f - view_.elapsed_s / view_.timeout_s, 0.f, 1.f);
                const QRect bar(box.left() + 10, box.top() + 28, box.width() - 20, 3);
                p.setPen(Qt::NoPen);
                p.setBrush(kTrack);
                p.drawRect(bar);
                p.setBrush(left_frac < 0.15f ? kAlert : kMuted);
                p.drawRect(QRect(bar.left(), bar.top(), int(bar.width() * left_frac), bar.height()));
            }

            // Clause chips, each an AND-input. The gate is drawn as a brace collecting them into
            // "hold stable" — the clauses are simultaneous, not successive, and this is the only way to
            // say that visually.
            int cy2 = box.top() + 40;
            int first_chip = -1, last_chip = -1;
            for (const auto &s : view_.steps)
            {
                if (s.kind != AffordanceStepView::Kind::Clause) continue;
                if (first_chip < 0) first_chip = cy2 + kChipH / 2;
                last_chip = cy2 + kChipH / 2;
                draw_chip(p, QRect(box.left() + 18, cy2, box.width() - 96, kChipH - 4), s);
                cy2 += kChipH;
            }
            const AffordanceStepView *stable = find(AffordanceStepView::Kind::Stable);
            if (stable != nullptr)
            {
                if (first_chip >= 0)
                {
                    // The AND brace: one vertical bar spanning the clauses, joining into the gate.
                    const int bx = box.right() - 74;
                    p.setPen(QPen(kMuted, 1.4));
                    p.drawLine(bx, first_chip, bx, last_chip);
                    p.drawLine(bx, (first_chip + last_chip) / 2, bx + 10, (first_chip + last_chip) / 2);
                    p.setPen(kDetail);
                    p.drawText(QRect(bx - 30, (first_chip + last_chip) / 2 - 9, 28, 18),
                               Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("AND"));
                }
                draw_chip(p, QRect(box.left() + 18, cy2, box.width() - 96, kChipH - 4), *stable);
            }
            y += box_h + kGap;
        }

        // ── TERMINAL ─────────────────────────────────────────────────────────────────────────────
        {
            const bool locked = view_.phase == "locked";
            const bool gave_up = view_.phase == "gave up";
            const bool reached = view_.phase == "reached";
            if (locked or gave_up or reached)
            {
                if (prev != nullptr or servo != nullptr) draw_edge(p, x + 18, y - kGap, y, true);
                AffordanceStepView t;
                t.label = gave_up ? "gave up" : locked ? "locked" : "reached";
                t.state = gave_up ? AffordanceStepView::State::Failed : AffordanceStepView::State::Done;
                t.detail = std::string("after ") + std::to_string(int(view_.elapsed_s)) + " s";
                draw_node(p, QRect(x, y, w, kNodeH), t, 16);
            }
        }
    }

private:
    static constexpr int kNodeH = 34, kGap = 18, kChipH = 22;

    // The ground this palette was designed against, and three text weights lifted well clear of it.
    // Anything dimmer than kMuted is unreadable at a glance, which is the only way this window is read.
    static inline const QColor kGround{0x23, 0x25, 0x28};
    static inline const QColor kLabel {0xf2, 0xf2, 0xf2};   // node + chip labels (always bold)
    static inline const QColor kDetail{0xc4, 0xc8, 0xcc};   // the NUMBERS beside a label
    static inline const QColor kMuted {0x93, 0x99, 0x9f};   // structure: edges, braces, skipped
    static inline const QColor kTrack {0x3a, 0x3d, 0x41};   // unfilled progress/timeout track
    static inline const QColor kAlert {0xe7, 0x4c, 0x3c};   // blocked / out of time

    const AffordanceStepView *find(AffordanceStepView::Kind k) const
    {
        for (const auto &s : view_.steps) if (s.kind == k) return &s;
        return nullptr;
    }
    static bool is_past(const AffordanceStepView &s)
    {
        return s.state == AffordanceStepView::State::Done or s.state == AffordanceStepView::State::Skipped;
    }
    static QColor colour(const AffordanceStepView &s)
    {
        using S = AffordanceStepView::State;
        switch (s.state)
        {
            // Brightened for the dark ground: the 0x27/0xc0/0x29 originals are mid-tones that a dark
            // fill swallows, and these are OUTLINE colours — a swallowed outline is a missing node.
            case S::Done:    return {0x2e, 0xcc, 0x71};
            case S::Failed:  return {0xe7, 0x4c, 0x3c};
            case S::Active:  return {0x4a, 0xa3, 0xe0};
            case S::Skipped: return {0x7a, 0x80, 0x86};
            default:         return kMuted;
        }
    }

    // An edge with a TOKEN on it when the step below is running: the dot says which way control is
    // flowing, which a static arrow does not.
    static void draw_edge(QPainter &p, int cx, int y0, int y1, bool past)
    {
        p.setPen(QPen(past ? QColor(0x2e, 0xcc, 0x71) : kMuted, 1.6));
        p.drawLine(cx, y0, cx, y1);
        QPainterPath head;
        head.moveTo(cx, y1); head.lineTo(cx - 4, y1 - 6); head.lineTo(cx + 4, y1 - 6); head.closeSubpath();
        p.setBrush(past ? QColor(0x2e, 0xcc, 0x71) : kMuted);
        p.setPen(Qt::NoPen);
        p.drawPath(head);
    }

    static void draw_node(QPainter &p, const QRect &r, const AffordanceStepView &s, int radius)
    {
        using S = AffordanceStepView::State;
        const QColor c = colour(s);
        const bool live = s.state == S::Active;
        // The active node PULSES. Derived from elapsed_s, so it needs no timer and it stops the moment
        // the step does — a node that has been "active" for 30 s pulses just as visibly as one that
        // just started, which is the point: a stall should be loud.
        const int alpha = live ? 30 + int(22.0 * (0.5 + 0.5 * std::sin(s.elapsed_s * 3.0))) : 16;
        p.setBrush(QColor(c.red(), c.green(), c.blue(), alpha));
        p.setPen(QPen(c, live ? 2.0 : 1.0, s.state == S::Skipped ? Qt::DashLine : Qt::SolidLine));
        p.drawRoundedRect(r, radius, radius);

        QFont f = p.font(); f.setBold(true); p.setFont(f);
        p.setPen(s.state == S::Skipped ? kMuted : kLabel);
        p.drawText(QRect(r.left() + 12, r.top(), 190, r.height()), Qt::AlignLeft | Qt::AlignVCenter,
                   QString::fromStdString(s.label));
        f.setBold(false); p.setFont(f);

        // Progress, drawn INSIDE the node as a filled underline rather than a separate bar: the number
        // stays the thing you read, and the bar is peripheral.
        if (s.progress >= 0.f)
        {
            const QRect u(r.left() + 12, r.bottom() - 7, 170, 3);
            p.setPen(Qt::NoPen);
            p.setBrush(kTrack); p.drawRect(u);
            p.setBrush(c);
            p.drawRect(QRect(u.left(), u.top(), int(u.width() * std::clamp(s.progress, 0.f, 1.f)), u.height()));
        }
        // blocked_why WINS: if something is stuck, the reason is the only thing worth reading here.
        const bool blocked = not s.blocked_why.empty();
        p.setPen(blocked ? kAlert : kDetail);
        p.drawText(QRect(r.left() + 210, r.top(), r.width() - 260, r.height()),
                   Qt::AlignLeft | Qt::AlignVCenter,
                   QString::fromStdString(blocked ? s.blocked_why : s.detail));
        if (s.state != S::Pending and s.state != S::Skipped and s.elapsed_s > 0.f)
        {
            p.setPen(kMuted);
            p.drawText(QRect(r.right() - 46, r.top(), 40, r.height()), Qt::AlignRight | Qt::AlignVCenter,
                       QStringLiteral("%1s").arg(double(s.elapsed_s), 0, 'f', 1));
        }
    }

    static void draw_chip(QPainter &p, const QRect &r, const AffordanceStepView &s)
    {
        const QColor c = colour(s);
        p.setBrush(QColor(c.red(), c.green(), c.blue(), 22));
        p.setPen(QPen(c, 1.0));
        p.drawRoundedRect(r, 7, 7);
        p.setPen(kLabel);
        p.drawText(QRect(r.left() + 8, r.top(), r.width() - 120, r.height()),
                   Qt::AlignLeft | Qt::AlignVCenter, QString::fromStdString(s.label));
        const bool blocked = not s.blocked_why.empty();
        p.setPen(blocked ? kAlert : kDetail);
        p.drawText(QRect(r.right() - 150, r.top(), 144, r.height()), Qt::AlignRight | Qt::AlignVCenter,
                   QString::fromStdString(blocked ? s.blocked_why : s.detail));
    }

    // A circular arrow that ROTATES while the loop is live. Static, it would just be an icon; turning,
    // it is the difference between "there is a loop here" and "the loop is going round right now".
    static void draw_loop_glyph(QPainter &p, int cx, int cy, const QColor &c, float phase_s)
    {
        p.setBrush(Qt::NoBrush);
        p.setPen(QPen(c, 1.8));
        const int R = 8;
        const int start = int(-phase_s * 220.f) % 360;
        p.drawArc(QRect(cx - R, cy - R, 2 * R, 2 * R), start * 16, 280 * 16);
        const double a = (start + 280) * M_PI / 180.0;
        QPainterPath head;
        const QPointF tip(cx + R * std::cos(a), cy - R * std::sin(a));
        head.moveTo(tip);
        head.lineTo(tip + QPointF(4 * std::cos(a - 2.2), -4 * std::sin(a - 2.2)));
        head.lineTo(tip + QPointF(4 * std::cos(a + 2.2), -4 * std::sin(a + 2.2)));
        head.closeSubpath();
        p.setBrush(c); p.setPen(Qt::NoPen);
        p.drawPath(head);
    }

    AffordanceExecution view_;
};

}   // namespace rc
