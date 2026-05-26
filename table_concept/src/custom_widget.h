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

/**
 * Minimal dockable widget for table_concept.
 * Hosts a QFrame (frame_series) into which TimeSeriesPlot is inserted at runtime.
 */
class Custom_widget : public QWidget
{
public:
    QFrame* frame_series = nullptr;

    explicit Custom_widget(QWidget* parent = nullptr) : QWidget(parent)
    {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(4, 4, 4, 4);
        layout->setSpacing(4);

        auto* title = new QLabel("Table Model — Free Energy & Coverage Deficit", this);
        title->setMaximumHeight(22);
        QFont f = title->font();
        f.setBold(true);
        title->setFont(f);
        layout->addWidget(title);

        frame_series = new QFrame(this);
        frame_series->setFrameShape(QFrame::StyledPanel);
        frame_series->setFrameShadow(QFrame::Raised);
        frame_series->setMinimumHeight(300);
        layout->addWidget(frame_series, 1);

        setLayout(layout);
        setMinimumSize(300, 340);
    }
};
