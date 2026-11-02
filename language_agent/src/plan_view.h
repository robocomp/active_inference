#pragma once
/*
 * plan_view.h — the "Plan" window: the mission the controller runs, drawn as its text form with live state (2026-10-17).
 *
 *   ┌ tour · in progress · 1:12                          remote · #4 ┐
 *   │ [Plan] [Text] [JSON]                                          │
 *   │  ✓ anchor start                                         0.0 s │
 *   │  ◌ repeat 3                                     iteration 2/3 │   ◌ spins while the repeat runs
 *   │  │ ◉ cross door_1 retry=2               attempt 2/3 · 14.2 s │   ◉ pulses: the leaf that drives
 *   │  │ ○ wait s=10s                                              │
 *   │  ○ explore room_2 budget_s=30s                                │   a bounded leaf fills a bar
 *   ├───────────────────────────────────────────────────────────────┤
 *   │ step 2.1:cross door_1: retry 2/3 after unreachable            │   the last event
 *   └───────────────────────────────────────────────────────────────┘
 *
 * Fed by the chat's 250 ms poll of the controller's `controller_mission` node (mission_watch.h): set_mission() when
 * a new IR appears, set_progress() with each snapshot. A step that changes state flashes in its new colour.
 * The animation timer runs only while the window is shown and something moves. No Q_OBJECT (no moc), as chat_window.h.
 * GUI thread only.
 */

#include <cstdint>
#include <functional>
#include <vector>

#include <QLabel>
#include <QPlainTextEdit>
#include <QScrollArea>
#include <QTabWidget>
#include <QTimer>
#include <QWidget>

#include "plan_model.h"

namespace rc::chat
{

class PlanCanvas;

class PlanView : public QWidget
{
public:
    explicit PlanView(QWidget* parent = nullptr);

    /// A new mission (it resets every row to pending). `ir_json` is the wire form, shown verbatim in the JSON tab.
    void set_mission(const rc::mission::Mission& m, const QString& ir_json);
    void set_progress(const rc::mission::Progress& p);
    [[nodiscard]] bool has_mission() const { return has_mission_; }
    [[nodiscard]] const plan::Model& model() const { return model_; }
    /// The clock the view draws at; tests and the demo pin it, the agent leaves it at the wall clock.
    void set_clock(std::function<std::uint64_t()> now) { now_ = std::move(now); }
    [[nodiscard]] std::uint64_t now() const { return now_(); }

    void showEvent(QShowEvent* e) override;
    void hideEvent(QHideEvent* e) override;

private:
    friend class PlanCanvas;
    void refresh_header();
    void tick();
    [[nodiscard]] bool animating() const;

    plan::Model model_;
    bool has_mission_ = false;
    std::vector<std::uint64_t> flash_at_;   ///< per row: when its state last changed (0 = never)
    int last_running_ = -1;
    std::function<std::uint64_t()> now_;

    QLabel* title_ = nullptr;
    QLabel* who_ = nullptr;
    QLabel* event_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    QScrollArea* scroll_ = nullptr;
    PlanCanvas* canvas_ = nullptr;
    QPlainTextEdit* text_ = nullptr;
    QPlainTextEdit* json_ = nullptr;
    QTimer timer_;
};

}   // namespace rc::chat
