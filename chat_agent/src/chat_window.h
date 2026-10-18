#pragma once
/*
 * chat_window.h — chat_agent's own window: transcript, input line, a confirmation card, and a status bar.
 *
 *   ┌ transcript (QTextBrowser) ─────────────────────────────┐
 *   │ you: cross the door                                     │
 *   │ robot: I can cross door_19 (2.3 m ahead-left).          │
 *   │ ⟶ robot status: approach — offered                      │
 *   ├ card (hidden unless a movement awaits confirmation) ────┤
 *   │ Confirm: cross door_19 …        [Confirm] [Cancel]      │
 *   ├─────────────────────────────────────────────────────────┤
 *   │ [ type here … (Enter sends, ↑↓ past prompts)] thinking… [Plan] [Cancel] │
 *   │ model qwen3:8b · ollama ok · controller present          │
 *   └─────────────────────────────────────────────────────────┘
 *
 * Enter sends; ↑/↓ walk past prompts (the line being typed is kept as the draft); /new, /clear or Ctrl+L
 * start a new conversation. [Plan] opens the plan window (plan_view.h): the mission the controller runs, live; it is
 * enabled once the controller has published one.
 * No Q_OBJECT (no moc): it talks upward through std::function callbacks, like controller_mission_panel.h.
 * GUI thread only.
 */

#include <functional>
#include <vector>

#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QTextBrowser>
#include <QWidget>

#include "plan_view.h"

namespace rc::chat
{

class ChatWindow : public QWidget
{
public:
    struct Callbacks
    {
        std::function<void(const QString&)> on_send;
        std::function<void()> on_cancel;     ///< abort the model's answer in flight
        std::function<void()> on_new;        ///< new conversation (history cleared)
        std::function<void()> on_confirm;    ///< the card's Confirm
        std::function<void()> on_reject;     ///< the card's Cancel
    };

    explicit ChatWindow(Callbacks cb, QWidget* parent = nullptr);

    void append_user(const QString& text);
    void begin_assistant();
    void append_token(const QString& token);
    void end_assistant();
    void append_robot(const QString& text);     ///< a robot status line (from the controller)
    void append_system(const QString& text);    ///< agent notes: validation, errors, sent commands
    void clear_transcript();

    void show_card(const QString& text);
    void hide_card();
    [[nodiscard]] bool card_visible() const;

    /// The controller's mission (mission_watch.h): a new one resets the plan window's rows.
    void set_plan(const rc::mission::Mission& m, const QString& ir_json);
    void set_plan_progress(const rc::mission::Progress& p);

    void set_busy(bool busy);
    void set_status(const QString& model, const QString& ollama, bool ollama_ok, bool controller_present);

    void closeEvent(QCloseEvent* e) override;

private:
    void append_html(const QString& html);
    void send_clicked();
    void history_step(int dir);   ///< −1 = older (↑), +1 = newer (↓)
    friend class HistoryKeys;

    Callbacks cb_;
    QTextBrowser* transcript_ = nullptr;
    QFrame* card_ = nullptr;
    QLabel* card_text_ = nullptr;
    QPushButton* card_confirm_ = nullptr;
    QPushButton* card_cancel_ = nullptr;
    QLineEdit* input_ = nullptr;
    QPushButton* cancel_ = nullptr;
    QPushButton* plan_btn_ = nullptr;
    PlanView* plan_ = nullptr;        ///< a top-level window this one owns
    QLabel* busy_ = nullptr;          ///< "thinking…" while a request is in flight (replaces a greyed Send button)
    QLabel* status_ = nullptr;
    std::vector<QString> history_;    ///< prompts sent, oldest first
    int hist_pos_ = 0;                ///< == history_.size() when editing the draft
    QString draft_;                   ///< the line being typed when ↑ was first pressed
    bool in_assistant_ = false;
};

}   // namespace rc::chat
