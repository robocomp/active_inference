#pragma once
/*
 * controller_status_line.h — the STATUS row of the controller window: one read-only line saying what the
 * controller is doing right now.
 *
 *   ● RUN · command · passage door_19: OPEN (executing) · aff aff_door_19_open [lock-on] · goal 1.2 m · hold: <why>
 *
 * It replaced the door-passage row (door combo + Cross / Stop). Those two commands still exist — chat_agent and the
 * remote file send `cross <door>` / `stop`, and the mission row's command box takes them — so the row's buttons
 * were a third way to issue them, while nothing in the window said in ONE place what the robot was up to.
 *
 * Header-only, no Q_OBJECT (the fleet dashboard convention, like Custom_widget). GUI thread only: the line is
 * COMPOSED in ControllerDisplay::present() from the snapshot the control thread staged, so no field here is ever
 * read off the GUI thread and no new state is kept — every word comes from something already shown elsewhere.
 */

#include <string>

#include <QLabel>
#include <QResizeEvent>

namespace rc
{

/// The door passage as the control thread staged it (DoorPassage::view()). `step` is DoorPassage::step_name();
/// empty `door` = never started.
struct DoorPassageStatus
{
    std::string door;
    std::string step;
    std::string step_state;
    int attempts = 0;
    std::string last_event;
};

class StatusLine : public QLabel
{
public:
    explicit StatusLine(QWidget *parent) : QLabel(parent)
    {
        setFrameShape(QFrame::StyledPanel);
        setTextFormat(Qt::PlainText);
        setTextInteractionFlags(Qt::TextSelectableByMouse);
        setContentsMargins(4, 2, 4, 2);
        // IGNORED horizontally: a label sizes to its text, and a long status at the end of a row would set the
        // window's minimum width (the toolbar's affordance label had that bug). Elided instead; full text in the tooltip.
        setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
        setToolTip(QStringLiteral("What the controller is doing now"));
    }

    /// GUI thread. Deduplicated: present() calls it every frame and an unchanged line costs nothing.
    void apply(const QString &line)
    {
        if (line == full_) return;
        full_ = line;
        setToolTip(full_);
        relayout();
    }

protected:
    void resizeEvent(QResizeEvent *e) override
    {
        QLabel::resizeEvent(e);
        relayout();
    }

private:
    void relayout()
    {
        const int w = contentsRect().width();
        setText(w > 0 ? fontMetrics().elidedText(full_, Qt::ElideRight, w) : full_);
    }

    QString full_;
};

}   // namespace rc
