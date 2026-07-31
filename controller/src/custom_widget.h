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

        auto *pose_panel = new QFrame(this);
        pose_panel->setFrameShape(QFrame::StyledPanel);
        pose_panel->setFrameShadow(QFrame::Raised);

        auto *pose_layout = new QVBoxLayout(pose_panel);
        pose_layout->setContentsMargins(8, 4, 8, 4);
        pose_layout->setSpacing(4);

        auto *cmd_row = new QHBoxLayout();
        cmd_row->setSpacing(8);
        auto *cmd_title = new QLabel("Cmd vel:", pose_panel);
        cmd_vel_value_ = new QLabel("adv 0 mm/s   side 0 mm/s   rot 0.00 rad/s", pose_panel);
        cmd_vel_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        cmd_row->addWidget(cmd_title);
        cmd_row->addWidget(cmd_vel_value_, 1);
        pose_layout->addLayout(cmd_row);

        auto *affordance_row = new QHBoxLayout();
        affordance_row->setSpacing(8);
        auto *affordance_title = new QLabel("Selected affordance:", pose_panel);
        selected_affordance_value_ = new QLabel("none", pose_panel);
        selected_affordance_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        affordance_row->addWidget(affordance_title);
        affordance_row->addWidget(selected_affordance_value_, 1);
        pose_layout->addLayout(affordance_row);

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

        // Live distance-to-target readout, immediately right of "MPPI paths". Shows what the ARRIVAL test is
        // actually looking at: remaining linear distance to the end of the path, and remaining angular error
        // to the commanded facing yaw. Monospaced so the digits don't jitter the layout as they change.
        // Turns amber while rotating in place to align, so an alignment that hunts instead of converging is
        // visible immediately rather than inferred from the robot's behaviour.
        goal_distance_label_ = new QLabel(QStringLiteral("d — m   θ — °"), toolbar);
        {
            QFont f = goal_distance_label_->font();
            f.setStyleHint(QFont::Monospace);
            f.setFamily(QStringLiteral("monospace"));
            goal_distance_label_->setFont(f);
        }
        goal_distance_label_->setToolTip(
            QStringLiteral("Remaining distance to target.\n"
                           "d = linear distance to the end of the planned path\n"
                           "θ = angular error to the commanded facing yaw (blank if the target has none)\n"
                           "Amber = rotating in place to align."));
        toolbar_layout->addWidget(goal_distance_label_);
        toolbar_layout->addStretch();

        // Stuck-recovery indicator, pinned to the right of the toolbar. Hidden while the robot
        // drives normally; lights up orange while an escape maneuver is reversing/turning the
        // base out of a wedge (see ControllerSession stuck recovery).
        stuck_status_label_ = new QLabel(QString(), toolbar);
        stuck_status_label_->setAlignment(Qt::AlignCenter);
        stuck_status_label_->setVisible(false);
        toolbar_layout->addWidget(stuck_status_label_);

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
    }
	~Custom_widget()
    {

    }

    void set_cmd_vel_text(const QString &text)
    {
        if (cmd_vel_value_ != nullptr)
            cmd_vel_value_->setText(text);
    }

    void set_selected_affordance_text(const QString &text)
    {
        if (selected_affordance_value_ != nullptr)
            selected_affordance_value_->setText(text);
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

    // Remaining distance to target, shown next to "MPPI paths". `yaw_err_rad` is nullopt when the target
    // carries no commanded facing yaw (a plain mouse target), in which case the angular field reads "—".
    // Dedups on the rendered string so a per-cycle call doesn't churn Qt.
    void set_goal_distance(std::optional<float> dist_m, std::optional<float> yaw_err_rad, bool aligning)
    {
        if (goal_distance_label_ == nullptr)
            return;
        const QString text = dist_m.has_value()
            ? QStringLiteral("d %1 m   θ %2")
                  .arg(*dist_m, 5, 'f', 2)
                  .arg(yaw_err_rad.has_value()
                           ? QStringLiteral("%1°").arg(*yaw_err_rad * 180.f / static_cast<float>(M_PI), 6, 'f', 1)
                           : QStringLiteral("   —  "))
            : QStringLiteral("d   —  m   θ   —  ");
        if (text != goal_distance_shown_)
        {
            goal_distance_shown_ = text;
            goal_distance_label_->setText(text);
        }
        if (aligning != goal_aligning_shown_)
        {
            goal_aligning_shown_ = aligning;
            goal_distance_label_->setStyleSheet(
                aligning ? "QLabel { background-color: #e67e22; color: white; font-weight: bold;"
                           " border-radius: 4px; padding: 2px 6px; }"
                         : "QLabel { padding: 2px 6px; }");
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
    }

    rc::TimeSeriesPlot *affordance_efe_plot = nullptr;
    QPushButton *lidar_toggle_btn = nullptr;
    QPushButton *mppi_paths_toggle_btn = nullptr;

private:
    QLabel *cmd_vel_value_ = nullptr;
    QLabel *selected_affordance_value_ = nullptr;
    QLabel *stuck_status_label_ = nullptr;
    bool stuck_active_shown_ = false;
    QLabel *goal_distance_label_ = nullptr;
    QString goal_distance_shown_;
    bool goal_aligning_shown_ = false;
};
#endif
