/*
 * controller_mission_panel.cpp — see controller_mission_panel.h
 */

#include "controller_mission_panel.h"

namespace rc
{

// Fewer than this and there is no tour to store; must match MissionRunner::finish_recording().
constexpr int kMinWaypoints = 2;

MissionPanel::MissionPanel(QWidget *parent, Callbacks callbacks)
    : QWidget(parent), cb_(std::move(callbacks))
{
    auto *frame = new QFrame(this);
    frame->setFrameShape(QFrame::StyledPanel);
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->addWidget(frame);

    auto *row = new QHBoxLayout(frame);
    row->setContentsMargins(4, 2, 4, 2);
    row->setSpacing(6);

    // WHAT DRIVES THE ROBOT. Named explicitly rather than implied by a combination of widgets: this is the
    // experimental condition of a run, it is written into the metrics CSV, and "which mode was this?" must
    // be answerable at a glance and months later.
    row->addWidget(new QLabel("Drive:", frame));
    drive_mode_ = new QComboBox(frame);
    // Order must match rc::to_index().
    drive_mode_->addItem("Affordances");
    drive_mode_->addItem("Mission");
    drive_mode_->addItem("Mission+Aff");
    drive_mode_->addItem("Target");
    drive_mode_->setToolTip(
        QStringLiteral("What is driving the robot.\n"
                       "Affordances — normal operation; the epistemic planner chooses.\n"
                       "Mission — the tour is the sole target source (the clean benchmark).\n"
                       "Mission+Aff — mission with affordances interleaved. NOT IMPLEMENTED YET;\n"
                       "              Run will refuse and say why.\n"
                       "Target — a single clicked point. Selected automatically when you click one."));
    row->addWidget(drive_mode_);
    QObject::connect(drive_mode_, &QComboBox::currentIndexChanged, this,
                     [this](int i) { if (cb_.on_drive_mode) cb_.on_drive_mode(i); });

    row->addWidget(new QLabel("Mission:", frame));
    missions_ = new QComboBox(frame);
    missions_->setMinimumWidth(110);
    missions_->setToolTip(QStringLiteral("Recorded missions (etc/missions.toml)."));
    row->addWidget(missions_);
    QObject::connect(missions_, &QComboBox::currentTextChanged, this,
                     [this](const QString &n)
                     { if (cb_.on_select and not n.isEmpty()) cb_.on_select(n.toStdString()); });

    // ── ROUTE CHARACTER: speed <-> safety ──
    // One dial over the optimiser's loss, not a pair of weights: it moves precision from the curvature
    // prior to the clearance preference and back (see RouteOptimizerConfig::safety_bias). It is here
    // rather than in a config file because its effect is a JUDGEMENT — how much lap time a metre of
    // clearance is worth — and that is the operator's call, made while watching the robot.
    // ★It applies to the NEXT route build or repair; it does not reshape the curve under the robot.
    row->addWidget(new QLabel("Route:", frame));
    safety_ = new QSlider(Qt::Horizontal, frame);
    safety_->setRange(0, 100);
    safety_->setValue(50);
    safety_->setFixedWidth(90);
    safety_->setToolTip(
        QStringLiteral("Route optimiser: speed <-> safety.\n"
                       "Left  — curvature dominates: wider, smoother corners, hugging obstacles to get\n"
                       "        them, and a higher speed allowed through v = sqrt(a_lat/kappa).\n"
                       "Right — clearance dominates: the route climbs onto the medial axis and accepts\n"
                       "        winding to stay there.\n"
                       "Measured on the 30-waypoint tour, the full sweep buys +32% p05 clearance for\n"
                       "+4.4% lap time. Applies to the NEXT route build or repair."));
    row->addWidget(safety_);
    safety_label_ = new QLabel("0.50", frame);
    safety_label_->setFixedWidth(30);
    row->addWidget(safety_label_);
    QObject::connect(safety_, &QSlider::valueChanged, this,
                     [this](int v)
                     {
                         const float bias = static_cast<float>(v) / 100.f;
                         if (safety_label_) safety_label_->setText(QString::asprintf("%.2f", bias));
                         if (cb_.on_safety_bias) cb_.on_safety_bias(bias);
                     });

    // ── Mission actions ──
    // A menu, not a row of verbs. Everything here acts on the SELECTED mission, and the item list
    // changes with state so it only ever offers what is actually possible: while recording, the only
    // sensible actions are save and cancel.
    actions_ = new QComboBox(frame);
    actions_->setMinimumWidth(100);
    actions_->setToolTip(
        QStringLiteral("Actions on the selected mission.\n"
                       "New — start clicking a fresh route (LEFT-CLICK appends, Ctrl+RIGHT-CLICK undoes).\n"
                       "Delete — remove it from etc/missions.toml.\n"
                       "Smooth — relax the waypoints, keeping every one footprint-feasible against the\n"
                       "         live occupancy grid. Endpoints are held; shifts are bounded so the\n"
                       "         route stays the one you authored."));
    row->addWidget(actions_);
    QObject::connect(actions_, &QComboBox::activated, this,
                     [this](int index)
                     {
                         if (index <= 0) return;                       // 0 is the placeholder
                         const QString action = actions_->itemText(index);
                         // Snap back BEFORE acting: the action may open a modal dialog, and a combo
                         // left showing "Delete" would read as a state the panel does not have.
                         { const QSignalBlocker block(actions_); actions_->setCurrentIndex(0); }

                         if (action.startsWith("New"))
                         { if (cb_.on_record_begin) cb_.on_record_begin(); return; }
                         if (action.startsWith("Cancel"))
                         { if (cb_.on_record_finish) cb_.on_record_finish({}); return; }
                         if (action.startsWith("Save"))
                         {
                             if (recorded_points_ < kMinWaypoints)
                             { if (cb_.on_record_finish) cb_.on_record_finish({}); return; }
                             const QString name = QInputDialog::getText(this, QStringLiteral("Save mission"),
                                                                        QStringLiteral("Mission name:"));
                             if (cb_.on_record_finish) cb_.on_record_finish(name.trimmed().toStdString());
                             return;
                         }
                         if (action.startsWith("Smooth"))
                         { if (cb_.on_smooth) cb_.on_smooth(); return; }
                         if (action.startsWith("Delete"))
                         {
                             const std::string name = selected_mission();
                             if (name.empty()) return;
                             // Deleting a recorded tour destroys a baseline other runs are compared
                             // against, and it sits one item away from Smooth. Confirm it.
                             if (QMessageBox::question(
                                     this, QStringLiteral("Delete mission"),
                                     QStringLiteral("Delete mission '%1'?\n\nRuns already in the metrics CSV "
                                                    "will no longer have their route on file.")
                                         .arg(QString::fromStdString(name)),
                                     QMessageBox::Yes | QMessageBox::No, QMessageBox::No) == QMessageBox::Yes
                                 and cb_.on_delete)
                                 cb_.on_delete(name);
                         }
                     });
    rebuild_actions(false, 0);

    row->addWidget(new QLabel("laps", frame));
    loops_ = new QSpinBox(frame);
    loops_->setRange(1, 99);
    loops_->setValue(1);
    loops_->setToolTip(
        QStringLiteral("Repeat the tour this many times.\n"
                       "Laps 2..N are scored against lap 1 — that difference IS the repeatability number.\n"
                       "Lap 1 starts from standstill, so consider it a warm-up and use N+1."));
    row->addWidget(loops_);

    // Laps REMAINING, counting down. The spin box says how many were asked for; this says how many are
    // left, which is the question you actually have while watching a long run. Two digits is enough for
    // the spin box's own 1..99 range, and it costs almost no width.
    laps_left_ = new QLCDNumber(2, frame);
    laps_left_->setSegmentStyle(QLCDNumber::Flat);
    laps_left_->setFrameShape(QFrame::NoFrame);
    laps_left_->setMinimumSize(30, 22);
    laps_left_->setToolTip(QStringLiteral("Laps remaining, including the one in progress."));
    laps_left_->display(QStringLiteral("--"));
    row->addWidget(laps_left_);

    // The ONE drive control. It replaces both the old toolbar Start/Stop toggle and the mission row's
    // separate Run and Stop: three buttons for two states, any two of which could disagree. It is NEVER
    // disabled — a halt that greys out in some modes is not a halt, which is exactly how the previous Stop
    // came to do nothing in "affordances only".
    // Parentless on purpose: Custom_widget reparents it into the top toolbar. A parented-but-unlaid-out
    // button would render on top of the panel until that happened.
    drive_btn_ = new QPushButton("Run", nullptr);
    drive_btn_->setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; }");
    drive_btn_->setToolTip(
        QStringLiteral("Run — start driving. In a mission mode this starts the selected tour;\n"
                       "in 'affordances only' it simply lets the epistemic planner drive.\n"
                       "Stop — halt the robot and end any running mission, whatever is driving."));
    QObject::connect(drive_btn_, &QPushButton::clicked, this,
                     [this]()
                     {
                         if (driving_) { if (cb_.on_stop) cb_.on_stop(); }
                         else if (cb_.on_run) cb_.on_run(loops_ != nullptr ? loops_->value() : 1);
                     });

    // Pause: a hold, not an abort. Parentless for the same reason as the drive button — Custom_widget
    // reparents it into the toolbar next to Run/Stop.
    pause_btn_ = new QPushButton("Pause", nullptr);
    pause_btn_->setStyleSheet("QPushButton { background-color: #f39c12; color: white; font-weight: bold; }");
    pause_btn_->setToolTip(
        QStringLiteral("Pause — hold the current activity where it is. The mission, route and lap\n"
                       "counter are kept, so pressing it again resumes from the same place.\n"
                       "Use Stop to ABORT instead: that clears the route and trace and disarms the base."));
    QObject::connect(pause_btn_, &QPushButton::clicked, this,
                     [this]() { if (cb_.on_pause) cb_.on_pause(not paused_); });

    row->addStretch();
}

void MissionPanel::set_missions(const std::vector<std::string> &names, const std::string &selected)
{
    if (missions_ == nullptr) return;
    // Repopulating fires currentTextChanged for every intermediate state, which would round-trip a
    // selection change back into the worker for missions the user never picked. Block it.
    QString shown;
    {
        const QSignalBlocker block(missions_);
        missions_->clear();
        for (const auto &n : names)
            missions_->addItem(QString::fromStdString(n));
        if (const int idx = missions_->findText(QString::fromStdString(selected)); idx >= 0)
            missions_->setCurrentIndex(idx);
        shown = missions_->currentText();
    }

    // CLOSE THE DESYNC. If `selected` was not in the list, setCurrentIndex never ran and the combo is
    // left showing item 0 while the worker still believes something else is selected. The user then
    // clicks the item already displayed, currentTextChanged does NOT fire (the text did not change),
    // and nothing happens — the classic "changing mission does nothing". Blocking the signal above is
    // right (it stops a repopulate being echoed back as a user choice), so the agreement has to be
    // restored explicitly here.
    if (not shown.isEmpty() and shown.toStdString() != selected and cb_.on_select)
        cb_.on_select(shown.toStdString());
}

void MissionPanel::rebuild_actions(bool recording, int recorded_points)
{
    if (actions_ == nullptr) return;
    const QSignalBlocker block(actions_);
    actions_->clear();
    if (recording)
    {
        actions_->addItem(recorded_points >= kMinWaypoints
                              ? QStringLiteral("Recording (%1)…").arg(recorded_points)
                              : QStringLiteral("Recording (%1)").arg(recorded_points));
        if (recorded_points >= kMinWaypoints)
            actions_->addItem(QStringLiteral("Save…"));
        actions_->addItem(QStringLiteral("Cancel"));
    }
    else
    {
        actions_->addItem(QStringLiteral("Mission…"));
        actions_->addItem(QStringLiteral("New"));
        actions_->addItem(QStringLiteral("Delete…"));
        actions_->addItem(QStringLiteral("Smooth"));
    }
    actions_->setCurrentIndex(0);
}

void MissionPanel::apply(const View &view)
{
    running_ = view.running;
    recording_ = view.recording;

    // The action list follows the state, so it only ever offers what is possible — and while
    // recording it carries the running point count, which is the one thing the user needs to know.
    if (recorded_points_ != view.recorded_points or recording_was_ != view.recording)
    {
        recorded_points_ = view.recorded_points;
        recording_was_ = view.recording;
        rebuild_actions(view.recording, view.recorded_points);
    }
    if (pause_btn_ != nullptr and paused_ != view.paused)
    {
        paused_ = view.paused;
        pause_btn_->setText(paused_ ? "Resume" : "Pause");
        pause_btn_->setStyleSheet(paused_
            ? "QPushButton { background-color: #2980b9; color: white; font-weight: bold; }"
            : "QPushButton { background-color: #f39c12; color: white; font-weight: bold; }");
    }
    // Pause means nothing when nothing is running.
    if (pause_btn_ != nullptr) pause_btn_->setEnabled(view.driving or view.paused);
    if (drive_btn_ != nullptr and driving_ != view.driving)
    {
        driving_ = view.driving;
        drive_btn_->setText(driving_ ? "Stop" : "Run");
        drive_btn_->setStyleSheet(driving_
            ? "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }"
            : "QPushButton { background-color: #27ae60; color: white; font-weight: bold; }");
    }

    status_ = view.status;

    if (laps_left_ != nullptr and laps_left_shown_ != view.laps_remaining)
    {
        laps_left_shown_ = view.laps_remaining;
        // "--" not "0" when idle: nothing is counting down, and a zero would read as "finished now".
        laps_left_->display(view.laps_remaining > 0 ? QString::number(view.laps_remaining)
                                                    : QStringLiteral("--"));
    }

    // The selector REPORTS what is driving, so it has to follow state the user did not set with it —
    // clicking a target switches it to "Target". Blocked, or the echo would be sent back as a fresh
    // mode change and clear the very click target that caused it.
    if (drive_mode_ != nullptr and drive_mode_->currentIndex() != view.mode_index)
    {
        const QSignalBlocker block(drive_mode_);
        drive_mode_->setCurrentIndex(view.mode_index);
    }

    // The drive button is deliberately NOT in this list: it must work in every mode.
    for (QWidget *w : {static_cast<QWidget *>(missions_),
                       static_cast<QWidget *>(actions_), static_cast<QWidget *>(loops_)})
        if (w != nullptr and w->isEnabled() != view.controls_enabled)
            w->setEnabled(view.controls_enabled);

}

bool MissionPanel::confirm_supersede()
{
    const auto answer = QMessageBox::question(
        this, QStringLiteral("Cancel running mission?"),
        QStringLiteral("A mission is running. Driving to the clicked target will END it and discard the "
                       "rest of the run.\n\nCancel the mission and go to the clicked target?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    return answer == QMessageBox::Yes;
}

std::string MissionPanel::selected_mission() const
{
    return missions_ != nullptr ? missions_->currentText().toStdString() : std::string{};
}

}  // namespace rc
