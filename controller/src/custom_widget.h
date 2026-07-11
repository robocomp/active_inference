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

        follow_toggle_btn = new QPushButton("Start", toolbar);
        follow_toggle_btn->setCheckable(true);
        follow_toggle_btn->setChecked(false);
        follow_toggle_btn->setStyleSheet(
            "QPushButton:checked { background-color: #c0392b; color: white; font-weight: bold; }"
            "QPushButton:!checked { background-color: #27ae60; color: white; font-weight: bold; }");
        toolbar_layout->addWidget(follow_toggle_btn);

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

        main_layout->addWidget(toolbar);

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

public:
    QFrame *frame = nullptr;
    rc::TimeSeriesPlot *affordance_efe_plot = nullptr;
    QPushButton *lidar_toggle_btn = nullptr;
    QPushButton *follow_toggle_btn = nullptr;
    QPushButton *mppi_paths_toggle_btn = nullptr;

private:
    QLabel *cmd_vel_value_ = nullptr;
    QLabel *selected_affordance_value_ = nullptr;
    QLabel *stuck_status_label_ = nullptr;
    bool stuck_active_shown_ = false;
};
#endif
