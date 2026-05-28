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
#endif

#include <ui_localUI.h>


class Custom_widget : public QWidget, public Ui_local_guiDlg
{
Q_OBJECT
public:
    Custom_widget() : Ui_local_guiDlg()
    {
        setupUi(this);

        auto *fps_panel = new QFrame(this);
        fps_panel->setFrameShape(QFrame::StyledPanel);
        fps_panel->setFrameShadow(QFrame::Raised);

        auto *panel_layout = new QVBoxLayout(fps_panel);
        panel_layout->setContentsMargins(8, 4, 8, 4);
        panel_layout->setSpacing(4);

        auto *fps_layout = new QHBoxLayout();
        fps_layout->setContentsMargins(0, 0, 0, 0);
        fps_layout->setSpacing(8);

        auto *fps_title = new QLabel("Compute FPS:", fps_panel);
        fps_value_ = new QLabel("0.0 Hz", fps_panel);
        fps_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        QFont value_font = fps_value_->font();
        value_font.setBold(true);
        fps_value_->setFont(value_font);

        fps_layout->addWidget(fps_title);
        fps_layout->addWidget(fps_value_, 1);

        panel_layout->addLayout(fps_layout);

        verticalLayout->insertWidget(1, fps_panel);
    }
	~Custom_widget()
    {

    }

    void set_fps_text(const QString &text)
    {
        if (fps_value_ != nullptr)
            fps_value_->setText(text);
    }

private:
    QLabel *fps_value_ = nullptr;



};
#endif
