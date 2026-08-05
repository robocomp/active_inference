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
        pose_row->setContentsMargins(8, 4, 8, 4);
        pose_row->setSpacing(6);

        const auto make_lcd = [&](QWidget *parent, int digits, const QString &tip)
        {
            auto *lcd = new QLCDNumber(digits, parent);
            lcd->setSegmentStyle(QLCDNumber::Flat);
            lcd->setFrameShape(QFrame::NoFrame);
            lcd->setMinimumSize(58, 26);
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

        pose_row->addStretch();

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
        // The same quality metric the run is graded on at STOP, accumulated live so a bad stretch is
        // visible while it happens rather than in a CSV afterwards. Four series, the terms of J, so the
        // plot says WHICH part is costing: smoothness of speed, of rotation, deviation, or clearance.
        auto *j_panel = new QFrame(this);
        j_panel->setFrameShape(QFrame::StyledPanel);
        j_panel->setFrameShadow(QFrame::Sunken);
        auto *j_layout = new QVBoxLayout(j_panel);
        j_layout->setContentsMargins(4, 2, 4, 2);
        j_layout->setSpacing(2);
        j_layout->addWidget(new QLabel("Running J = smooth_lin + smooth_rot + dev_norm + clear_norm "
                                       "(lower = better)", j_panel));
        mission_j_plot = new rc::TimeSeriesPlot(j_panel);
        mission_j_plot->setMinimumHeight(80);
        mission_j_plot->set_visible_window(120.f);   // a lap is ~95 s, so one window holds a whole lap
        j_layout->addWidget(mission_j_plot, 1);
        main_layout->addWidget(j_panel, 1);
    }
	~Custom_widget()
    {

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
    rc::TimeSeriesPlot *mission_j_plot = nullptr;   // running J, below the EFE panel
    QPushButton *lidar_toggle_btn = nullptr;
    QPushButton *mppi_paths_toggle_btn = nullptr;

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
    bool stuck_active_shown_ = false;
    bool goal_aligning_shown_ = false;
};
#endif
