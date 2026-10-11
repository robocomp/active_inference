#pragma once
/*
 * controller_door_passage_bar.h — the DOOR PASSAGE row of the controller window (controller_door_passage.h).
 *
 *   door [door_19 ▾] [Cross] [Stop]   approach ✓ → open ✓ → CROSS (executing)   <last transition>
 *
 * Header-only, no Q_OBJECT (the fleet dashboard convention, like Custom_widget): the buttons are connected to lambdas.
 * GUI thread only. It is fed a snapshot staged from the control thread (ControllerDisplay::set_door_passage) and its
 * buttons call back into the worker, which ENQUEUES the start/stop for the control thread — the same rule every
 * other control in this window follows.
 */

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <QComboBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

namespace rc
{

class DoorPassageBar : public QFrame
{
public:
    struct Callbacks
    {
        std::function<void(const std::string &)> on_cross;   // start a passage through this door
        std::function<void()>                    on_stop;    // end it; selection unrestricted again
    };
    /// What the control thread staged. `step` is DoorPassage::step_name(); empty `door` = never started.
    struct View
    {
        std::string door;
        std::string step;
        std::string step_state;
        int attempts = 0;
        std::string last_event;
        std::vector<std::string> doors;   // door nodes in the graph, for the combo
    };

    DoorPassageBar(QWidget *parent, Callbacks cb) : QFrame(parent), cb_(std::move(cb))
    {
        setFrameShape(QFrame::StyledPanel);
        auto *lay = new QHBoxLayout(this);
        lay->setContentsMargins(4, 2, 4, 2);
        lay->setSpacing(6);
        lay->addWidget(new QLabel("door", this));
        // EDITABLE: the list is what the graph holds now, but a door's node may not exist yet when the operator
        // already knows its name — DoorPassage waits for the affordances to be offered, so a typed name is valid.
        doors_ = new QComboBox(this);
        doors_->setEditable(true);
        doors_->setMinimumContentsLength(10);
        doors_->setToolTip(QStringLiteral("Door node to cross (door_concept's belief). Typing a name is allowed."));
        lay->addWidget(doors_);
        cross_ = new QPushButton("Cross", this);
        cross_->setToolTip(QStringLiteral("Sequence aff_<door>_approach → _open → _cross. The drive must be "
                                          "running in an affordance mode for the robot to move."));
        lay->addWidget(cross_);
        stop_ = new QPushButton("Stop", this);
        stop_->setToolTip(QStringLiteral("End the passage; the selector is unrestricted again."));
        lay->addWidget(stop_);
        steps_ = new QLabel(this);
        steps_->setTextFormat(Qt::RichText);
        lay->addWidget(steps_);
        event_ = new QLabel(this);
        event_->setStyleSheet(QStringLiteral("color: gray;"));
        // Bounded, for the same reason as the toolbar's affordance label: an unbounded label at the end of a row
        // sets the window's minimum width. The full line stays in the tooltip.
        event_->setMaximumWidth(420);
        lay->addWidget(event_, 1);

        QObject::connect(cross_, &QPushButton::clicked, this, [this]
        {
            const std::string name = doors_->currentText().trimmed().toStdString();
            if (not name.empty() and cb_.on_cross) cb_.on_cross(name);
        });
        QObject::connect(stop_, &QPushButton::clicked, this, [this] { if (cb_.on_stop) cb_.on_stop(); });
        apply(View{});
    }

    void apply(const View &v)
    {
        // The combo is refilled only when the graph's door set CHANGES, and never while the operator is typing in
        // it — a refill at the control rate would wipe the edit and the selection under the cursor.
        if (v.doors != doors_shown_ and not doors_->lineEdit()->hasFocus())
        {
            const QString keep = doors_->currentText();
            doors_->clear();
            for (const auto &d : v.doors) doors_->addItem(QString::fromStdString(d));
            if (not keep.isEmpty()) doors_->setCurrentText(keep);
            doors_shown_ = v.doors;
        }
        const bool running = v.step == "approach" or v.step == "open" or v.step == "cross";
        stop_->setEnabled(running);

        QString html;
        if (v.door.empty()) html = QStringLiteral("<i>no passage</i>");
        else
        {
            // approach → open → cross: done steps ticked, the current one in bold with its affordance's state.
            constexpr std::array<const char *, 3> kSteps{"approach", "open", "cross"};
            const int cur = v.step == "approach" ? 0 : v.step == "open" ? 1 : v.step == "cross" ? 2
                          : v.step == "done" ? 3 : -1;
            html = QStringLiteral("<b>%1</b>: ").arg(QString::fromStdString(v.door).toHtmlEscaped());
            for (int i = 0; i < 3; ++i)
            {
                if (i > 0) html += QStringLiteral(" → ");
                const QString s = QString::fromLatin1(kSteps[i]);
                if (i < cur) html += s + QStringLiteral(" ✓");
                else if (i == cur)
                {
                    html += QStringLiteral("<b style='color:#c87814'>%1</b> (%2)")
                                .arg(s.toUpper(), QString::fromStdString(v.step_state).toHtmlEscaped());
                    if (v.attempts > 0) html += QStringLiteral(" <span style='color:#b03030'>×%1 failed</span>").arg(v.attempts);
                }
                else html += QStringLiteral("<span style='color:gray'>%1</span>").arg(s);
            }
            if (cur == 3) html += QStringLiteral(" — <b style='color:#2c9258'>DONE</b>");
            else if (cur < 0) html += QStringLiteral(" — <i>%1</i>").arg(QString::fromStdString(v.step).toHtmlEscaped());
        }
        if (html != steps_shown_) { steps_->setText(html); steps_shown_ = html; }
        const QString ev = QString::fromStdString(v.last_event);
        if (ev != event_shown_)
        {
            event_->setText(event_->fontMetrics().elidedText(ev, Qt::ElideRight, event_->maximumWidth()));
            event_->setToolTip(ev);
            event_shown_ = ev;
        }
    }

private:
    Callbacks cb_;
    QComboBox *doors_ = nullptr;
    QPushButton *cross_ = nullptr;
    QPushButton *stop_ = nullptr;
    QLabel *steps_ = nullptr;
    QLabel *event_ = nullptr;
    std::vector<std::string> doors_shown_;
    QString steps_shown_, event_shown_;
};

}   // namespace rc
