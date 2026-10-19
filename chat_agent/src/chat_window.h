#pragma once
/*
 * chat_window.h — chat_agent's own window: transcript, input line and a status bar.
 *
 *   ┌ transcript (QTextBrowser) ──────────────────────┬ canvas (canvas_panel.h) ──────┐
 *   │ you: show me the room                            │ room_1 ● live        </> × │
 *   │ robot: drawn: room layout                        │                             │
 *   │ ⟶ robot status: approach — offered               │   (what the robot drew)     │
 *   ├──────────────────────────────────────────────────┴─────────────────────────────┤
 *   │ [ type here … (Enter sends, ↑↓ past prompts)]                         thinking… │
 *   │ model qwen3:8b · ollama ok · controller present                                 │
 *   └────────────────────────────────────────────────────────────────────────────────┘
 *
 * Enter sends; ↑/↓ walk past prompts (the line being typed is kept as the draft); /new, /clear or Ctrl+L
 * start a new conversation; Esc stops the model's answer in flight. The canvas, docked in a splitter, is open
 * from the start with a hint (show_canvas); the controller's mission is one of its pages ("show me the plan").
 * No Q_OBJECT (no moc): it talks upward through std::function callbacks, like controller_mission_panel.h.
 * GUI thread only.
 */

#include <functional>
#include <vector>

#include <QFrame>
#include <QSplitter>
#include <QLabel>
#include <QLineEdit>
#include <QString>
#include <QTextBrowser>
#include <QWidget>

#include "canvas_panel.h"

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
        CanvasPanel::Callbacks canvas;       ///< the side canvas' [×] and [Save]
    };

    explicit ChatWindow(Callbacks cb, QWidget* parent = nullptr);

    void append_user(const QString& text);
    void begin_assistant();
    void append_token(const QString& token);
    void end_assistant();
    void append_robot(const QString& text);     ///< a robot status line (from the controller)
    void append_system(const QString& text);    ///< agent notes: validation, errors, sent commands
    void clear_transcript();

    /// The controller's mission (mission_watch.h): a new one resets the plan page's rows. It does not open the canvas.
    void set_plan(const rc::mission::Mission& m, const QString& ir_json);
    void set_plan_progress(const rc::mission::Progress& p);

    [[nodiscard]] CanvasPanel* canvas() const { return canvas_; }
    void show_canvas();    ///< unhide the canvas; the window widens by it (its last width, 560 px at first)
    void hide_canvas();    ///< hide it and give its width back: the window returns to its size without it
    [[nodiscard]] bool canvas_visible() const { return canvas_->isVisible(); }

    void set_busy(bool busy);
    void set_status(const QString& model, const QString& ollama, bool ollama_ok, bool controller_present,
                    const QString& gpu);   ///< gpu: "GPU 8.1/15.9 GB" | "GPU n/a" (gpu_memory.h)

    void closeEvent(QCloseEvent* e) override;

private:
    bool restored_with_canvas_ = false;   ///< the saved geometry was taken with the canvas open
    void append_html(const QString& html);
    void send_clicked();
    void history_step(int dir);   ///< −1 = older (↑), +1 = newer (↓)
    friend class HistoryKeys;

    Callbacks cb_;
    QTextBrowser* transcript_ = nullptr;
    QLineEdit* input_ = nullptr;
    QSplitter* split_ = nullptr;      ///< transcript | canvas
    CanvasPanel* canvas_ = nullptr;
    QLabel* busy_ = nullptr;          ///< "thinking…" while a request is in flight (replaces a greyed Send button)
    QLabel* status_ = nullptr;
    std::vector<QString> history_;    ///< prompts sent, oldest first
    int hist_pos_ = 0;                ///< == history_.size() when editing the draft
    QString draft_;                   ///< the line being typed when ↑ was first pressed
    bool in_assistant_ = false;
};

}   // namespace rc::chat
