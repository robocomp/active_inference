#pragma once

#if Qt5_FOUND
    #include <QtWidgets>
#else
    #include <QtGui>
#endif

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QFrame>
#include <QString>

/**
 * Shared minimal dockable belief-dashboard host for the concept agents (bottle/table/chair/…).
 * Hosts a QFrame (frame_series) into which one or more rc::TimeSeriesPlot are inserted at runtime.
 * The title is passed by the agent (e.g. "Bottle Model — …"), so this is a single shared widget.
 */
class Custom_widget : public QWidget
{
public:
    QFrame* frame_series = nullptr;

    explicit Custom_widget(const QString& title = QStringLiteral("Belief Model"),
                           QWidget* parent = nullptr) : QWidget(parent)
    {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(4, 4, 4, 4);
        layout->setSpacing(4);

        auto* title_label = new QLabel(title, this);
        title_label->setMaximumHeight(22);
        QFont f = title_label->font();
        f.setBold(true);
        title_label->setFont(f);
        layout->addWidget(title_label);

        frame_series = new QFrame(this);
        frame_series->setFrameShape(QFrame::StyledPanel);
        frame_series->setFrameShadow(QFrame::Raised);
        frame_series->setMinimumHeight(200);
        layout->addWidget(frame_series, 1);

        setLayout(layout);
        setMinimumSize(220, 240);
    }
};
