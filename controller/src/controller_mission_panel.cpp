/*
 * controller_mission_panel.cpp — see controller_mission_panel.h
 */

#include "controller_mission_panel.h"

#include <algorithm>   // std::clamp — set_plain_l; do not rely on Qt pulling it in
#include <cmath>       // std::lround

namespace rc
{

// Fewer than this and there is no tour to store; must match MissionRunner::finish_recording().
constexpr int kMinWaypoints = 2;

// Plain-tracker L slider range, in MILLIMETRES of arc (a QSlider is integer-valued). 0.15 m is inside
// the region where the loop is expected to ring; 1.50 m is well past visible corner-cutting. Both ends
// are reachable on purpose — this control exists to find the edge, not to stay inside a known-good band.
constexpr int kPlainLMinMm = 150;
constexpr int kPlainLMaxMm = 1500;

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

    // ── HOW HARD THE ROBOT STICKS TO THE ROUTE: the plain tracker's L ──
    // ★REPLACED the Risk <-> Safe route-optimiser dial (2026-08-12). That slider moved
    // RouteOptimizerConfig::safety_bias, which applies only to the NEXT route BUILD — so moving it
    // while watching the robot did nothing at all until the next replan, which is the opposite of what
    // a live slider is for. Its value is a considered, measured one (config RouteSafetyBias = 0.75,
    // with a clearance/length frontier recorded beside it), and it belongs in the config file where a
    // measurement can sit next to it. L is the one that is genuinely a live judgement.
    // L is the closed-loop error-decay LENGTH: an off-route error decays over L metres of arc at any
    // speed, because the feedback is critically damped in the arc-length domain rather than in time
    // (plain_tracker.cpp). Both feedback gains follow from it — 2/L on heading, 1/L^2 on cross-track —
    // so it is the single number that says how hard the robot is pulled back onto the line.
    // ★The labels follow the VALUE and need no inversion: the slider carries L in millimetres, small L
    // means large gains means a tighter line, so Stick is the left (low) end and Loose the right (high)
    // one. The same pedantry the Risk/Safe label carried — a control whose ends are mislabelled is
    // worse than no control — but here it comes out the easy way round.
    // ★It acts on the very next control cycle — this is a live gain, not a route property.
    row->addWidget(new QLabel("Stick", frame));
    plain_l_ = new QSlider(Qt::Horizontal, frame);
    // Range in millimetres of arc. The low end is where the loop is expected to ring (below ~0.3 m the
    // 1/L^2 term dominates and TV(w) climbs sharply); the high end is well past the point where the
    // robot visibly cuts corners. Both ends are deliberately reachable: this is a control for finding
    // out, and clamping it to the measured-safe band would make it useless for that.
    plain_l_->setRange(kPlainLMinMm, kPlainLMaxMm);
    plain_l_->setValue(500);
    plain_l_->setFixedWidth(90);
    plain_l_->setToolTip(
        QStringLiteral("Plain tracker: how hard the robot sticks to the route (L, metres of arc).\n"
                       "An off-route error decays over L metres AT ANY SPEED — the feedback is critically\n"
                       "damped in arc length, so one gain is right at every speed.\n"
                       "Stick (left, small L)  — 2/L and 1/L^2 grow: tighter line, more rotational effort,\n"
                       "                         and below ~0.3 m the loop starts to ring.\n"
                       "Loose (right, large L) — smoother and quieter, with a larger standing offset on\n"
                       "                         curves; the robot cuts corners.\n"
                       "Offline (tools/tracker_sim, identified plant): rms 53 mm at 0.25, 31 mm at 0.50,\n"
                       "45 mm at 0.90, 76 mm at 1.40 — so it has an interior optimum near the shipped 0.50.\n"
                       "The policy that set it is 'minimise cross-track rms subject to rotational effort\n"
                       "<= 0.87 rad/m'; watch rot/m in the metrics, not rms alone.\n"
                       "Takes effect on the NEXT control cycle."));
    row->addWidget(plain_l_);
    row->addWidget(new QLabel("Loose", frame));
    plain_l_label_ = new QLabel("0.50", frame);
    plain_l_label_->setFixedWidth(30);
    row->addWidget(plain_l_label_);
    QObject::connect(plain_l_, &QSlider::valueChanged, this,
                     [this](int mm)
                     {
                         const float L = static_cast<float>(mm) / 1000.f;
                         if (plain_l_label_) plain_l_label_->setText(QString::asprintf("%.2f", L));
                         if (cb_.on_plain_l) cb_.on_plain_l(L);
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

    // ── DRIVE THE TOUR BACKWARDS ──────────────────────────────────────────────────────────────
    // A recorded tour is a sequence, and the direction it is driven in is a real part of the stimulus:
    // the same corners become the opposite handedness, the approach to every object arrives from the
    // other side, and a route the optimiser found easy one way can be tight the other. Re-recording the
    // tour backwards to test that would be a different route, not the same one reversed — which is
    // exactly what makes this worth a toggle rather than a second mission.
    // It reverses the WAYPOINT ORDER for the run only; the recorded mission on disk is untouched.
    reverse_btn_ = new QPushButton("Reverse", frame);
    reverse_btn_->setCheckable(true);
    reverse_btn_->setToolTip(
        QStringLiteral("Drive the selected tour in REVERSE waypoint order.\n"
                       "Applies to the next Run; the recorded mission is not modified.\n"
                       "The run's JSON records it (params.reversed), because a tour driven backwards is a\n"
                       "different stimulus and two runs that differ silently are not comparable."));
    QObject::connect(reverse_btn_, &QPushButton::toggled, this,
                     [this](bool on)
                     {
                         // Amber when armed, so it is visible at a glance next to a green Run — an
                         // unnoticed reverse would silently invalidate a comparison.
                         reverse_btn_->setStyleSheet(
                             on ? "QPushButton { background-color: #f39c12; color: white; font-weight: bold; }"
                                : QString{});
                     });
    row->addWidget(reverse_btn_);

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
                         else if (cb_.on_run)
                             cb_.on_run(loops_ != nullptr ? loops_->value() : 1,
                                        reverse_btn_ != nullptr and reverse_btn_->isChecked());
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

void MissionPanel::set_plain_l(float metres)
{
    if (plain_l_ == nullptr) return;
    const int mm = std::clamp(static_cast<int>(std::lround(metres * 1000.f)), kPlainLMinMm, kPlainLMaxMm);
    // Blocked: this is ADOPTING the control side's value, not asking for a new one. Letting it emit
    // would round-trip the clamped number back as a command and overwrite a config L outside the range.
    { const QSignalBlocker block(plain_l_); plain_l_->setValue(mm); }
    if (plain_l_label_) plain_l_label_->setText(QString::asprintf("%.2f", static_cast<float>(mm) / 1000.f));
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
