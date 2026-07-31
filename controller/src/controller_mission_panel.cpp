/*
 * controller_mission_panel.cpp — see controller_mission_panel.h
 */

#include "controller_mission_panel.h"

namespace rc
{

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
    drive_mode_->addItem("Mission + affordances");
    drive_mode_->addItem("Target");
    drive_mode_->setToolTip(
        QStringLiteral("What is driving the robot.\n"
                       "Affordances — normal operation; the epistemic planner chooses.\n"
                       "Mission — the tour is the sole target source (the clean benchmark).\n"
                       "Mission + affordances — NOT IMPLEMENTED YET; Run will refuse and say why.\n"
                       "Target — a single clicked point. Selected automatically when you click one."));
    row->addWidget(drive_mode_);
    QObject::connect(drive_mode_, &QComboBox::currentIndexChanged, this,
                     [this](int i) { if (cb_.on_drive_mode) cb_.on_drive_mode(i); });

    row->addWidget(new QLabel("Mission:", frame));
    missions_ = new QComboBox(frame);
    missions_->setMinimumWidth(150);
    missions_->setToolTip(QStringLiteral("Recorded missions (etc/missions.toml)."));
    row->addWidget(missions_);
    QObject::connect(missions_, &QComboBox::currentTextChanged, this,
                     [this](const QString &n)
                     { if (cb_.on_select and not n.isEmpty()) cb_.on_select(n.toStdString()); });

    record_btn_ = new QPushButton("New", frame);
    record_btn_->setCheckable(true);
    record_btn_->setToolTip(
        QStringLiteral("Create a NEW mission.\n"
                       "While checked, LEFT-CLICK in the view appends a waypoint (it does not drive).\n"
                       "Ctrl+RIGHT-CLICK removes the last one. Uncheck to name and save.\n"
                       "To change an existing mission, just drag its waypoints with the RIGHT button."));
    record_btn_->setStyleSheet("QPushButton:checked { background-color: #8e44ad; color: white; font-weight: bold; }");
    row->addWidget(record_btn_);
    QObject::connect(record_btn_, &QPushButton::toggled, this,
                     [this](bool checked)
                     {
                         if (checked)
                         {
                             if (cb_.on_record_begin) cb_.on_record_begin();
                             return;
                         }
                         const QString name = QInputDialog::getText(this, QStringLiteral("Save mission"),
                                                                    QStringLiteral("Mission name:"));
                         if (cb_.on_record_finish) cb_.on_record_finish(name.trimmed().toStdString());
                     });

    delete_btn_ = new QPushButton("Delete", frame);
    delete_btn_->setToolTip(QStringLiteral("Delete the selected mission from etc/missions.toml."));
    row->addWidget(delete_btn_);
    QObject::connect(delete_btn_, &QPushButton::clicked, this,
                     [this]()
                     {
                         const std::string name = selected_mission();
                         if (name.empty()) return;
                         // Deleting a recorded tour destroys a baseline that other runs are compared against,
                         // and it is one click away from Run. Confirm it.
                         const auto answer = QMessageBox::question(
                             this, QStringLiteral("Delete mission"),
                             QStringLiteral("Delete mission '%1'?\n\nRuns already recorded in the metrics CSV "
                                            "will no longer have their route on file.")
                                 .arg(QString::fromStdString(name)),
                             QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                         if (answer == QMessageBox::Yes and cb_.on_delete) cb_.on_delete(name);
                     });

    row->addWidget(new QLabel("laps", frame));
    loops_ = new QSpinBox(frame);
    loops_->setRange(1, 99);
    loops_->setValue(1);
    loops_->setToolTip(
        QStringLiteral("Repeat the tour this many times.\n"
                       "Laps 2..N are scored against lap 1 — that difference IS the repeatability number.\n"
                       "Lap 1 starts from standstill, so consider it a warm-up and use N+1."));
    row->addWidget(loops_);

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

    row->addStretch();
}

void MissionPanel::set_missions(const std::vector<std::string> &names, const std::string &selected)
{
    if (missions_ == nullptr) return;
    // Repopulating fires currentTextChanged for every intermediate state, which would round-trip a
    // selection change back into the worker for missions the user never picked. Block it.
    const QSignalBlocker block(missions_);
    missions_->clear();
    for (const auto &n : names)
        missions_->addItem(QString::fromStdString(n));
    if (const int idx = missions_->findText(QString::fromStdString(selected)); idx >= 0)
        missions_->setCurrentIndex(idx);
}

void MissionPanel::apply(const View &view)
{
    running_ = view.running;
    recording_ = view.recording;
    if (drive_btn_ != nullptr and driving_ != view.driving)
    {
        driving_ = view.driving;
        drive_btn_->setText(driving_ ? "Stop" : "Run");
        drive_btn_->setStyleSheet(driving_
            ? "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }"
            : "QPushButton { background-color: #27ae60; color: white; font-weight: bold; }");
    }

    status_ = view.status;

    // The selector REPORTS what is driving, so it has to follow state the user did not set with it —
    // clicking a target switches it to "Target". Blocked, or the echo would be sent back as a fresh
    // mode change and clear the very click target that caused it.
    if (drive_mode_ != nullptr and drive_mode_->currentIndex() != view.mode_index)
    {
        const QSignalBlocker block(drive_mode_);
        drive_mode_->setCurrentIndex(view.mode_index);
    }

    // The drive button is deliberately NOT in this list: it must work in every mode.
    for (QWidget *w : {static_cast<QWidget *>(missions_), static_cast<QWidget *>(record_btn_),
                       static_cast<QWidget *>(delete_btn_), static_cast<QWidget *>(loops_)})
        if (w != nullptr and w->isEnabled() != view.controls_enabled)
            w->setEnabled(view.controls_enabled);

    // Keep Record showing the ACTUAL state: a recording can end from the worker's side (a save that failed
    // validation, a mode change), and a button that still looks armed would send the next click to a
    // recording that no longer exists.
    if (record_btn_ != nullptr and record_btn_->isChecked() != view.recording)
    {
        const QSignalBlocker block(record_btn_);
        record_btn_->setChecked(view.recording);
    }
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
