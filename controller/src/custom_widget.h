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


class Custom_widget : public QWidget
{
public:
    Custom_widget()
    {
        auto *main_layout = new QVBoxLayout(this);
        main_layout->setContentsMargins(8, 8, 8, 8);
        main_layout->setSpacing(8);

        auto *pose_panel = new QFrame(this);
        pose_panel->setFrameShape(QFrame::StyledPanel);
        pose_panel->setFrameShadow(QFrame::Raised);

        auto *pose_layout = new QHBoxLayout(pose_panel);
        pose_layout->setContentsMargins(8, 4, 8, 4);
        pose_layout->setSpacing(8);

        auto *pose_title = new QLabel("Robot in room:", pose_panel);
        pose_value_ = new QLabel("x 0.00 m   y 0.00 m   th 0.0 deg", pose_panel);
        pose_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);

        QFont value_font = pose_value_->font();
        value_font.setBold(true);
        pose_value_->setFont(value_font);

        pose_layout->addWidget(pose_title);
        pose_layout->addWidget(pose_value_, 1);

        main_layout->addWidget(pose_panel);

        frame = new QFrame(this);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Sunken);
        frame->setMinimumSize(320, 320);
        auto *frame_layout = new QVBoxLayout(frame);
        frame_layout->setContentsMargins(0, 0, 0, 0);
        frame_layout->setSpacing(0);
        main_layout->addWidget(frame, 1);
    }
	~Custom_widget()
    {

    }

    void set_pose_text(const QString &text)
    {
        if (pose_value_ != nullptr)
            pose_value_->setText(text);
    }

public:
    QFrame *frame = nullptr;

private:
    QLabel *pose_value_ = nullptr;
};
#endif
