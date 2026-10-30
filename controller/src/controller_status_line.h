#pragma once
/*
 * controller_status_line.h — the STATUS column of the controller window, to the right of the 2D view: read-only
 * text saying what the controller is doing right now, one item per line:
 *
 *   RUN
 *   command
 *   passage door_19: OPEN (executing)
 *   aff aff_door_19_open [lock-on] 3/20 s
 *   goal 1.20 m
 *   hold: <why>
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

#include <QPlainTextEdit>

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

class StatusLine : public QPlainTextEdit
{
public:
    explicit StatusLine(QWidget *parent) : QPlainTextEdit(parent)
    {
        setReadOnly(true);
        setFrameShape(QFrame::StyledPanel);
        // ONE ITEM PER LINE, wrapped, in a fixed-width column to the right of the 2D view. As a single elided
        // line under the mission row it set the window's width and was still cut off; a column costs no width
        // beyond its own and shows every word. Wrap ANYWHERE as a last resort: node names have no spaces.
        setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
        setLineWrapMode(QPlainTextEdit::WidgetWidth);
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setFixedWidth(230);
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        setToolTip(QStringLiteral("What the controller is doing now"));
    }

    /// GUI thread. Deduplicated: present() calls it every frame and an unchanged text costs nothing (and
    /// re-setting it would also drop a selection the user is making).
    /// `colour` frames the column in the motion verdict's class colour (who has to act).
    void apply(const QString &text, const QString &colour)
    {
        if (text != full_)
        {
            full_ = text;
            setPlainText(full_);
        }
        if (colour != colour_)
        {
            colour_ = colour;
            setStyleSheet(QStringLiteral("QPlainTextEdit { border: 3px solid %1; border-radius: 4px; }").arg(colour_));
        }
    }

private:
    QString full_, colour_;
};

}   // namespace rc
