/*
 * controller_mission_panel.h — the whole mission UI, in one widget.
 *
 * Everything mission-shaped that used to be scattered across custom_widget.h (layout), controller_display.cpp
 * (signal wiring, dialogs) and specificworker.cpp (intent) lives here. The rest of the GUI now embeds one
 * widget and forwards one struct; it does not know what a lap is.
 *
 * The panel owns its own DIALOGS — the name prompt, the delete confirmation, the "cancel the running
 * mission?" question. Those are GUI-thread, blocking, and inherently presentational, so putting them behind
 * a callback would only move the QMessageBox somewhere it has no business being.
 *
 * No Q_OBJECT: the panel talks upward through std::function callbacks, the same way ControllerDisplay does,
 * which keeps it out of AUTOMOC entirely (see the build note in CLAUDE.md about AUTOMOC cycles).
 */

#pragma once

#if Qt5_FOUND
    #include <QtWidgets>
#else
    #include <QtGui>
    #include <QtWidgets>
#endif

#include <functional>
#include <string>
#include <vector>

namespace rc
{

class MissionPanel : public QWidget
{
public:
    struct Callbacks
    {
        std::function<void(int)>         on_drive_mode;    // 0=affordances, 1=mission, 2=mission+affordances
        std::function<void(std::string)> on_select;
        std::function<void()>            on_record_begin;
        // Empty name ⇒ the user cancelled; discard whatever was being recorded or edited.
        std::function<void(std::string)> on_record_finish;
        std::function<void(std::string)> on_delete;
        std::function<void()>            on_smooth;
        std::function<void(int)>         on_run;           // laps
        std::function<void()>            on_stop;
        // Route optimiser: speed (0) <-> safety (1). See RouteOptimizerConfig::safety_bias — it trades
        // PRECISION between the clearance preference and the curvature prior, so it is one number with a
        // measurable frontier rather than two weights to guess at.
        std::function<void(float)>       on_safety_bias;
    };

    // What the panel needs to know each cycle to show the truth. Pushed from the GUI thread in present().
    struct View
    {
        std::string status;
        bool controls_enabled = false;   // false unless a mission mode — the mission widgets mean nothing
        bool running = false;
        bool recording = false;
        bool driving = false;            // is the base allowed to move right now?
        int  mode_index = 0;             // rc::to_index(mode); the selector follows the state, not only clicks
        int  recorded_points = 0;        // points placed so far in the recording being built
        int  laps_remaining = 0;         // countdown incl. the lap in progress; 0 when idle
    };

    explicit MissionPanel(QWidget *parent, Callbacks callbacks);

    // The ONE drive control: Run when halted (green), Stop when driving (red). It is created here so all
    // mission behaviour stays in this class, but it is PLACED in the main toolbar by Custom_widget — it is
    // the primary control and belongs on the top line, not buried in the mission row.
    QPushButton *drive_button() const { return drive_btn_; }

    void set_missions(const std::vector<std::string> &names, const std::string &selected);
    void rebuild_actions(bool recording, int recorded_points);
    void apply(const View &view);

    // Is a mission running RIGHT NOW? Cached from the last apply() so the GUI thread can answer without
    // reaching into control-thread state. Used to decide whether a click needs a confirmation.
    bool mission_running() const { return running_; }
    bool recording() const { return recording_; }

    // Ask whether a click may cancel the running mission. Returns true if the user said yes. Blocking and
    // GUI-thread only. Called on a mouse click, which is already a GUI-thread event.
    bool confirm_supersede();

    std::string selected_mission() const;
    // Latest status line, for the window title (see status_).
    const std::string &status() const { return status_; }

private:
    Callbacks cb_;
    QComboBox   *drive_mode_ = nullptr;
    QComboBox   *missions_ = nullptr;
    // One action menu for everything that acts ON the selected mission, instead of a button per verb.
    // It is a MENU, not a state: it snaps back to its placeholder after firing, so it never claims to
    // show a current value it does not have.
    QComboBox   *actions_ = nullptr;
    QSpinBox    *loops_ = nullptr;
    QLCDNumber  *laps_left_ = nullptr;
    int  laps_left_shown_ = -1;
    QPushButton *drive_btn_ = nullptr;
    QSlider     *safety_ = nullptr;
    QLabel      *safety_label_ = nullptr;

    bool running_ = false;
    bool recording_ = false;
    bool driving_ = false;
    int  recorded_points_ = 0;
    bool recording_was_ = false;
    // Status is NOT shown in this row: a stretchy label with lap/waypoint text forced the whole window
    // wider than the 2D view needed. It goes in the WINDOW TITLE instead, which costs no layout width.
    std::string status_;
};

}  // namespace rc
