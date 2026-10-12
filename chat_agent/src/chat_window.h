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
 *   │ [ type here …                ] [Send] [Cancel] [New]    │
 *   │ model qwen3:8b · ollama ok · controller present          │
 *   └─────────────────────────────────────────────────────────┘
 *
 * No Q_OBJECT (no moc): it talks upward through std::function callbacks, like controller_mission_panel.h.
 * GUI thread only.
 */

#include <functional>

#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QTextBrowser>
#include <QWidget>

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

    void set_busy(bool busy);
    void set_status(const QString& model, const QString& ollama, bool ollama_ok, bool controller_present);

    void closeEvent(QCloseEvent* e) override;

private:
    void append_html(const QString& html);
    void send_clicked();

    Callbacks cb_;
    QTextBrowser* transcript_ = nullptr;
    QFrame* card_ = nullptr;
    QLabel* card_text_ = nullptr;
    QPushButton* card_confirm_ = nullptr;
    QPushButton* card_cancel_ = nullptr;
    QLineEdit* input_ = nullptr;
    QPushButton* send_ = nullptr;
    QPushButton* cancel_ = nullptr;
    QPushButton* new_ = nullptr;
    QLabel* status_ = nullptr;
    bool in_assistant_ = false;
};

}   // namespace rc::chat
