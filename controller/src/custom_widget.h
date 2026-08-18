/*
 *    Copyright (C) 2020 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/



#ifndef CUSTOMWIDGET_H
#define CUSTOMWIDGET_H

#if Qt5_FOUND
	#include <QtWidgets>
#else
	#include <QtGui>
	#include <QtWidgets>
#endif

#include <cmath>
#include <limits>

#include "../../common/dashboard/timeseries_plot.h"
#include "controller_mission_panel.h"


class Custom_widget : public QWidget
{
public:
    explicit Custom_widget(QWidget *parent = nullptr) : QWidget(parent)
    {
        auto *main_layout = new QVBoxLayout(this);
        main_layout->setContentsMargins(8, 8, 8, 8);
        main_layout->setSpacing(8);

        // ── Row 1: what the base is being told to do, and what arrival is waiting on ──
        // QLCDNumber rather than a formatted label: these are the numbers you WATCH while the robot
        // moves, and a proportional label re-flows as digits change, which makes a jittering value hard
        // to read precisely when it matters. Fixed-width segments do not move.
        auto *pose_panel = new QFrame(this);
        pose_panel->setFrameShape(QFrame::StyledPanel);
        pose_panel->setFrameShadow(QFrame::Raised);

        auto *pose_row = new QHBoxLayout(pose_panel);
        pose_row->setContentsMargins(4, 3, 4, 3);
        pose_row->setSpacing(3);

        const auto make_lcd = [&](QWidget *parent, int digits, const QString &tip)
        {
            auto *lcd = new QLCDNumber(digits, parent);
            lcd->setSegmentStyle(QLCDNumber::Flat);
            lcd->setFrameShape(QFrame::NoFrame);
            // ★WIDTH FOLLOWS THE DIGIT COUNT. It was a flat 58 px for every LCD, so a 5-digit reading
            // reserved as much room as an 8-digit one and the row's MINIMUM width — which is what the
            // window can never be dragged below — carried that slack seven times over.
            lcd->setMinimumSize(9 * digits + 4, 24);
            // ...and it must not grow past what it needs either: the default policy is Expanding, so a
            // wide window handed the surplus to the LCDs rather than to the plots below, which is the
            // other half of why this row looked spread out.
            lcd->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
            lcd->setToolTip(tip);
            lcd->display(QStringLiteral("---"));
            return lcd;
        };

        pose_row->addWidget(new QLabel("adv", pose_panel));
        cmd_adv_lcd_ = make_lcd(pose_panel, 5, QStringLiteral("Commanded forward speed (mm/s)."));
        pose_row->addWidget(cmd_adv_lcd_);
        pose_row->addWidget(new QLabel("side", pose_panel));
        cmd_side_lcd_ = make_lcd(pose_panel, 5, QStringLiteral("Commanded lateral speed (mm/s)."));
        pose_row->addWidget(cmd_side_lcd_);
        pose_row->addWidget(new QLabel("rot", pose_panel));
        cmd_rot_lcd_ = make_lcd(pose_panel, 5, QStringLiteral("Commanded rotation (rad/s)."));
        pose_row->addWidget(cmd_rot_lcd_);

        // Alerts (LiDAR stall, recovery). Hidden while nothing is wrong, so the row stays numeric —
        // but a stalled sensor must never be silent just because the row is now made of digits.
        alert_label_ = new QLabel(QString(), pose_panel);
        alert_label_->setVisible(false);
        alert_label_->setStyleSheet("QLabel { background-color: #c0392b; color: white; font-weight: bold;"
                                    " border-radius: 4px; padding: 2px 8px; }");
        pose_row->addWidget(alert_label_);

        // ★A FIXED GAP, NOT A STRETCH. Two addStretch() calls used to float the session pair mid-row and
        // push the goal pair to the far right, which reads as three widely separated islands and gives
        // the window no reason to be narrow. A fixed gap keeps the groups legibly apart while letting
        // the row pack to its natural width, and the single trailing stretch below takes up whatever
        // slack is left rather than distributing it between the numbers.
        pose_row->addSpacing(14);

        // ── SESSION TOTALS ──────────────────────────────────────────────────────────────────────
        // Distance driven and time elapsed since the agent started — every mission, click target and
        // affordance, NOT per run. MissionRunner integrates both, but only while a mission is RUNNING,
        // so nothing ever counted a target or the drive back to a start point. LCDs like the rest of
        // this row: these tick continuously, and a proportional label re-flows as digits change.
        // ★UNITS TRAIL THE NUMBER, and there is no group label. "run" named the pair but left both
        // readings unitless, so the odometer and the clock were two anonymous numbers side by side;
        // "12.4 m." and "07:31 mins." each say what they are without a caption to bind them to.
        // A stretch on BOTH sides floats the pair mid-row between the command LCDs and the goal
        // readout, instead of leaving it crowded against the right end.
        session_dist_lcd_ = make_lcd(pose_panel, 6,
            QStringLiteral("Distance driven this session (m, or km past 1000).\n"
                           "All missions, targets and affordances since the agent started."));
        pose_row->addWidget(session_dist_lcd_);
        pose_row->addWidget(new QLabel("m.", pose_panel));
        session_time_lcd_ = make_lcd(pose_panel, 8,
            QStringLiteral("Wall time since the agent started (mm:ss, or h:mm:ss past an hour)."));
        pose_row->addWidget(session_time_lcd_);
        pose_row->addWidget(new QLabel("mins.", pose_panel));

        pose_row->addSpacing(14);

        // Right end: what the ARRIVAL test is waiting on — remaining distance and heading error.
        pose_row->addWidget(new QLabel("d", pose_panel));
        goal_dist_lcd_ = make_lcd(pose_panel, 5,
            QStringLiteral("Remaining linear distance to the end of the planned path (m)."));
        pose_row->addWidget(goal_dist_lcd_);
        pose_row->addWidget(new QLabel("\u03b8", pose_panel));
        goal_theta_lcd_ = make_lcd(pose_panel, 5,
            QStringLiteral("Angular error to the commanded facing yaw (deg).\n"
                           "Blank when the target carries none. Amber = rotating in place to align."));
        pose_row->addWidget(goal_theta_lcd_);
        // The ONE stretch, at the end: slack goes here rather than between the readings.
        pose_row->addStretch();

        main_layout->addWidget(pose_panel);

        // ---- toolbar row ----
        auto *toolbar = new QFrame(this);
        toolbar->setFrameShape(QFrame::StyledPanel);
        auto *toolbar_layout = new QHBoxLayout(toolbar);
        toolbar_layout->setContentsMargins(4, 2, 4, 2);
        toolbar_layout->setSpacing(6);
        lidar_toggle_btn = new QPushButton("Lidar points", toolbar);
        lidar_toggle_btn->setCheckable(true);
        lidar_toggle_btn->setChecked(false);
        toolbar_layout->addWidget(lidar_toggle_btn);
        toolbar_layout_ = toolbar_layout;


        mppi_paths_toggle_btn = new QPushButton("MPPI paths", toolbar);
        mppi_paths_toggle_btn->setCheckable(true);
        mppi_paths_toggle_btn->setChecked(false);
        toolbar_layout->addWidget(mppi_paths_toggle_btn);

        toolbar_layout->addStretch();

        // Stuck-recovery indicator, pinned to the right of the toolbar. Hidden while the robot
        // drives normally; lights up orange while an escape maneuver is reversing/turning the
        // base out of a wedge (see ControllerSession stuck recovery).
        stuck_status_label_ = new QLabel(QString(), toolbar);
        stuck_status_label_->setAlignment(Qt::AlignCenter);
        stuck_status_label_->setVisible(false);
        toolbar_layout->addWidget(stuck_status_label_);

        // Which affordance the epistemic planner picked. It belongs beside the drive controls — it is
        // the answer to "why is it going there", not a pose readout.
        toolbar_layout->addWidget(new QLabel("affordance", toolbar));
        selected_affordance_value_ = new QLabel(toolbar);
        // CLICKABLE: it opens the affordance program window. An affordance is a small program with
        // steps, and the name alone says only WHICH one is running, never how far in it is or what is
        // holding it up.
        selected_affordance_value_->setCursor(Qt::PointingHandCursor);
        selected_affordance_value_->setToolTip(QStringLiteral("Click to open the affordance program "
                                                              "— steps, progress, and what is blocking."));
        selected_affordance_value_->installEventFilter(this);
        selected_affordance_value_->setTextFormat(Qt::RichText);
        selected_affordance_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        // Bounded: affordance names come from graph nodes and can be long, and an unbounded label at the
        // end of the row sets the window's minimum width. The full text stays in the tooltip.
        selected_affordance_value_->setMaximumWidth(240);
        toolbar_layout->addWidget(selected_affordance_value_);

        main_layout->addWidget(toolbar);

        // ---- mission row ----
        // All of it lives in MissionPanel; this widget only decides WHERE it goes.
        // Created later by attach_mission_panel(), because it needs the worker's callbacks.
        mission_row_index = main_layout->count();

        frame = new QFrame(this);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Sunken);
        frame->setMinimumSize(320, 320);
        main_layout->addWidget(frame, 3);

        // Affordance EFE time-series panel, below the 2D view.
        auto *efe_panel = new QFrame(this);
        efe_panel->setFrameShape(QFrame::StyledPanel);
        efe_panel->setFrameShadow(QFrame::Sunken);
        auto *efe_layout = new QVBoxLayout(efe_panel);
        efe_layout->setContentsMargins(4, 2, 4, 2);
        efe_layout->setSpacing(2);
        efe_layout->addWidget(new QLabel("Affordance EFE score (gain − λ·dist; higher = selected)", efe_panel));
        affordance_efe_plot = new rc::TimeSeriesPlot(efe_panel);
        affordance_efe_plot->setMinimumHeight(80);
        affordance_efe_plot->set_visible_window(60.f);
        efe_layout->addWidget(affordance_efe_plot, 1);
        main_layout->addWidget(efe_panel, 1);

        // ── RUNNING J, directly below the EFE score ──────────────────────────────────────────────
        // ── WHAT THE WHEELS ARE ACTUALLY ASKED TO DO ────────────────────────────────────────────
        // The commanded advance and rotation, sampled at the OUTPUT rate (every command the motion
        // commander emits, ~40 Hz) rather than at the GUI rate — smoothness is a property of the signal
        // the base receives, and sampling it at the redraw rate would alias exactly the jitter worth
        // seeing. Both live in roughly 0..0.8 (m/s and rad/s), so they share an axis legibly.
        // ★This replaced the L-adaptation plot (objective cross_rms vs constraint rot_per_m). That pair
        // was fed from MissionRunner::summary() and therefore read zero unless a MISSION was running —
        // through every clicked target and every affordance drive, which is most of what the robot does.
        auto *j_panel = new QFrame(this);
        j_panel->setFrameShape(QFrame::StyledPanel);
        j_panel->setFrameShadow(QFrame::Sunken);
        auto *j_layout = new QVBoxLayout(j_panel);
        j_layout->setContentsMargins(4, 2, 4, 2);
        j_layout->setSpacing(2);
        j_layout->addWidget(new QLabel("commanded velocity (whoever is driving) — adv (m/s)  |  rot (rad/s)  |  "
                                       "σxy scaled: TOP = PoseXYStdStop (throttle floored)   "
                                       "[gap = nothing commanding]", j_panel));
        mission_j_plot = new rc::TimeSeriesPlot(j_panel);
        mission_j_plot->setMinimumHeight(80);
        // ~30 s: long enough to hold a whole approach and its arrival, short enough that individual
        // command steps are still distinguishable. A 120 s window compressed them into a smear.
        mission_j_plot->set_visible_window(30.f);
        // Registered HERE, on the GUI thread at construction, so the output thread only ever calls
        // add_point() — which is mutex-guarded and documented safe from any thread. Registering lazily
        // from that thread would be the one unguarded moment.
        mission_j_plot->add_series("adv", QColor(55, 55, 55), 2.0f);
        mission_j_plot->add_series("rot", QColor(70, 130, 200), 2.0f);
        // ── THE LOCALISATION UNCERTAINTY THAT IS THROTTLING THE BASE, ON THE SAME AXES ───────────
        // sigma_xy is metres, so it is mapped onto the m/s axis before it gets here (stop knee -> max
        // adv, clamped) — see ControllerDisplay::set_uncertainty_trace_value. Without that mapping it
        // would either sit invisibly along the bottom or, once past the knee, blow the auto-scale and
        // squash the two traces it exists to be read against.
        // Thin and violet: it is a REFERENCE line, not a command, and neither the near-black adv nor the
        // blue rot can be confused with it. The question it answers at a glance is the one that took a
        // whole session to answer offline — when adv sags, is this line high?
        mission_j_plot->add_series("sigma", QColor(170, 60, 175), 1.4f);
        j_layout->addWidget(mission_j_plot, 1);
        main_layout->addWidget(j_panel, 1);
    }
	~Custom_widget()
    {

    }

    // Session totals. Dedups: called every cycle, and re-rendering an unchanged string churns Qt for
    // nothing (the same reasoning as set_cmd_vel below). 0.1 m and 1 s are the displayed resolutions.
    void set_session(float metres, float seconds)
    {
        if (session_dist_lcd_ != nullptr
            and (not std::isfinite(session_metres_shown_) or std::abs(metres - session_metres_shown_) >= 0.1f))
        {
            session_metres_shown_ = metres;
            session_dist_lcd_->display(metres >= 1000.f
                ? QString::number(static_cast<double>(metres) / 1000.0, 'f', 2) + "k"
                : QString::number(static_cast<double>(metres), 'f', 1));
        }
        const int secs = static_cast<int>(seconds);
        if (session_time_lcd_ != nullptr and secs != session_secs_shown_)
        {
            session_secs_shown_ = secs;
            const int h = secs / 3600, m = (secs / 60) % 60, sec = secs % 60;
            session_time_lcd_->display(h > 0
                ? QStringLiteral("%1:%2:%3").arg(h).arg(m, 2, 10, QLatin1Char('0'))
                                            .arg(sec, 2, 10, QLatin1Char('0'))
                : QStringLiteral("%1:%2").arg(m, 2, 10, QLatin1Char('0'))
                                         .arg(sec, 2, 10, QLatin1Char('0')));
        }
    }

    // The commanded base velocity, as numbers rather than a sentence. Dedups: this is called on every
    // command (~20 Hz) and re-displaying an unchanged value churns Qt for nothing.
    void set_cmd_vel(float adv_mm_s, float side_mm_s, float rot_rps)
    {
        const auto put = [](QLCDNumber *lcd, float v, int decimals, float &shown)
        {
            if (lcd == nullptr or (std::isfinite(shown) and std::abs(v - shown) < 1e-4f)) return;
            shown = v;
            lcd->display(QString::number(static_cast<double>(v), 'f', decimals));
        };
        put(cmd_adv_lcd_, adv_mm_s, 0, cmd_adv_shown_);
        put(cmd_side_lcd_, side_mm_s, 0, cmd_side_shown_);
        put(cmd_rot_lcd_, rot_rps, 2, cmd_rot_shown_);
    }

    // Alerts that are not numbers (LiDAR stall, recovery). Empty hides the badge.
    void set_cmd_vel_text(const QString &text)
    {
        if (alert_label_ == nullptr or text == alert_shown_) return;
        alert_shown_ = text;
        alert_label_->setText(text);
        alert_label_->setVisible(not text.isEmpty());
    }

    // The CURRENT affordance is the thing you are looking for on this row, so it is coloured and bold;
    // the previous one is context and is dimmed. Formatting lives here rather than in the worker: the
    // worker knows which affordance is selected, not what colour "selected" should be.
    // "none" is deliberately styled as absence (grey, italic) — a selection and the lack of one must not
    // read the same at a glance.
    void set_affordance_clicked(std::function<void()> cb) { affordance_clicked_ = std::move(cb); }

    void set_selected_affordance(const QString &current, const QString &previous)
    {
        if (selected_affordance_value_ == nullptr) return;
        const bool have = not current.isEmpty() and current != QStringLiteral("none");
        const QString html =
            (have ? QStringLiteral("<span style='color:#2e86de; font-weight:bold;'>%1</span>")
                  : QStringLiteral("<span style='color:#7f8c8d; font-style:italic;'>%1</span>"))
                .arg(current.toHtmlEscaped())
            + QStringLiteral("<span style='color:#95a5a6;'> &nbsp;(prev: %1)</span>")
                  .arg(previous.toHtmlEscaped());
        if (html == affordance_shown_) return;   // called every cycle; don't churn Qt
        affordance_shown_ = html;
        selected_affordance_value_->setText(html);
        selected_affordance_value_->setToolTip(current + QStringLiteral("   (prev: ") + previous + ")");
    }

    // Toggle the stuck-recovery indicator. Orange + visible while the robot is escaping a wedge;
    // hidden otherwise. Dedups so repeated same-state calls (one per cycle) don't churn Qt.
    void set_stuck_active(bool active)
    {
        if (stuck_status_label_ == nullptr or active == stuck_active_shown_)
            return;
        stuck_active_shown_ = active;
        if (active)
        {
            stuck_status_label_->setText(QStringLiteral(" ⚠ STUCK — escaping "));
            stuck_status_label_->setStyleSheet(
                "QLabel { background-color: #e67e22; color: white; font-weight: bold;"
                " border-radius: 4px; padding: 2px 8px; }");
            stuck_status_label_->setVisible(true);
        }
        else
            stuck_status_label_->setVisible(false);
    }

    // What the ARRIVAL test is waiting on, at the right end of the pose row. `yaw_err_rad` is nullopt
    // when the target carries no commanded facing yaw (a plain mouse target), in which case theta reads
    // "---" rather than 0 — an absent constraint and a satisfied one must not look alike.
    void set_goal_distance(std::optional<float> dist_m, std::optional<float> yaw_err_rad, bool aligning)
    {
        if (goal_dist_lcd_ != nullptr)
        {
            const QString d = dist_m.has_value()
                ? QString::number(static_cast<double>(*dist_m), 'f', 2) : QStringLiteral("---");
            if (d != goal_dist_shown_) { goal_dist_shown_ = d; goal_dist_lcd_->display(d); }
        }
        if (goal_theta_lcd_ != nullptr)
        {
            const QString th = yaw_err_rad.has_value()
                ? QString::number(static_cast<double>(*yaw_err_rad) * 180.0 / M_PI, 'f', 1)
                : QStringLiteral("---");
            if (th != goal_theta_shown_) { goal_theta_shown_ = th; goal_theta_lcd_->display(th); }
            if (aligning != goal_aligning_shown_)
            {
                goal_aligning_shown_ = aligning;
                // Amber while rotating in place, so an alignment that hunts instead of converging is
                // visible immediately rather than inferred from the robot's behaviour.
                goal_theta_lcd_->setStyleSheet(aligning ? "QLCDNumber { background-color: #e67e22; }" : "");
            }
        }
    }


public:
    QFrame *frame = nullptr;
    rc::MissionPanel *mission_panel = nullptr;
    int mission_row_index = 0;   // where attach_mission_panel() inserts the panel
    QHBoxLayout *toolbar_layout_ = nullptr;   // top line; hosts the drive button

    void attach_mission_panel(rc::MissionPanel *panel)
    {
        mission_panel = panel;
        if (panel == nullptr)
            return;
        if (auto *lay = qobject_cast<QVBoxLayout *>(layout()); lay != nullptr)
            lay->insertWidget(mission_row_index, panel);
        // The drive control is the primary action, so it goes FIRST on the top toolbar rather than at the
        // end of the mission row. The panel still owns it; this only decides where it is shown.
        if (toolbar_layout_ != nullptr and panel->drive_button() != nullptr)
            toolbar_layout_->insertWidget(0, panel->drive_button());
        if (toolbar_layout_ != nullptr and panel->pause_button() != nullptr)
            toolbar_layout_->insertWidget(1, panel->pause_button());
    }

    rc::TimeSeriesPlot *affordance_efe_plot = nullptr;
    rc::TimeSeriesPlot *mission_j_plot = nullptr;   // commanded adv/rot trace, below the EFE panel
    QPushButton *lidar_toggle_btn = nullptr;
    QPushButton *mppi_paths_toggle_btn = nullptr;

protected:
    // QLabel carries no clicked() signal and this widget has no Q_OBJECT (the fleet dashboard
    // convention), so the affordance name's click is hit-tested here instead of pulling in a subclass
    // just to get one signal. Mapped through the label's own parent, because it lives in the toolbar
    // frame and its geometry() is in THAT frame's coordinates, not this widget's.
    // ★AN EVENT FILTER, NOT mousePressEvent. The label sets TextSelectableByMouse (so a long node name
    // can be copied), and that makes QLabel CONSUME the press — it never reaches this widget's handler,
    // so hit-testing there would silently never fire. A filter sees the event before the label does.
    // Returns false on purpose: the label still gets it, so selecting the text keeps working.
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (watched == selected_affordance_value_ and event->type() == QEvent::MouseButtonRelease
            and affordance_clicked_)
            if (const auto *me = static_cast<QMouseEvent *>(event); me->button() == Qt::LeftButton)
                affordance_clicked_();
        return QWidget::eventFilter(watched, event);
    }

private:
    QLCDNumber *cmd_adv_lcd_ = nullptr;
    QLCDNumber *cmd_side_lcd_ = nullptr;
    QLCDNumber *cmd_rot_lcd_ = nullptr;
    QLCDNumber *goal_dist_lcd_ = nullptr;
    QLCDNumber *goal_theta_lcd_ = nullptr;
    QLabel *alert_label_ = nullptr;
    QString alert_shown_;
    float cmd_adv_shown_ = std::numeric_limits<float>::quiet_NaN();
    float cmd_side_shown_ = std::numeric_limits<float>::quiet_NaN();
    float cmd_rot_shown_ = std::numeric_limits<float>::quiet_NaN();
    QString goal_dist_shown_, goal_theta_shown_;
    QLabel *selected_affordance_value_ = nullptr;
    QString affordance_shown_;
    QLabel *stuck_status_label_ = nullptr;

    std::function<void()> affordance_clicked_;   // opens the affordance program window
    QLCDNumber *session_dist_lcd_ = nullptr;   // session totals, top row
    QLCDNumber *session_time_lcd_ = nullptr;

    float session_metres_shown_ = std::numeric_limits<float>::quiet_NaN();
    int session_secs_shown_ = -1;
    bool stuck_active_shown_ = false;
    bool goal_aligning_shown_ = false;
};
#endif
