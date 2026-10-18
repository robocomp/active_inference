#include "chat_window.h"
#include <algorithm>
#include <QKeySequence>
#include <QKeyEvent>
#include <QShortcut>

#include <QCloseEvent>
#include <QHBoxLayout>
#include <QScrollBar>
#include <QSettings>
#include <QTextCursor>
#include <QVBoxLayout>

namespace rc::chat
{

// ↑/↓ on the input line walk the prompt history. A plain QObject with an eventFilter override: no signals, so no moc.
class HistoryKeys : public QObject
{
public:
    explicit HistoryKeys(ChatWindow* w) : QObject(w), w_(w) {}
    bool eventFilter(QObject* obj, QEvent* e) override
    {
        if (e->type() == QEvent::KeyPress)
        {
            const auto* k = static_cast<QKeyEvent*>(e);
            if (k->key() == Qt::Key_Up)   { w_->history_step(-1); return true; }
            if (k->key() == Qt::Key_Down) { w_->history_step(+1); return true; }
        }
        return QObject::eventFilter(obj, e);
    }
private:
    ChatWindow* w_;
};


ChatWindow::ChatWindow(Callbacks cb, QWidget* parent) : QWidget(parent), cb_(std::move(cb))
{
    setWindowTitle(QStringLiteral("Robot chat"));
    auto* lay = new QVBoxLayout(this);

    split_ = new QSplitter(Qt::Horizontal, this);
    transcript_ = new QTextBrowser(split_);
    transcript_->setOpenExternalLinks(false);
    transcript_->setMinimumSize(480, 360);
    canvas_ = new CanvasPanel(cb_.canvas, split_);
    canvas_->setVisible(false);
    split_->addWidget(transcript_);
    split_->addWidget(canvas_);
    split_->setStretchFactor(0, 1);
    split_->setStretchFactor(1, 1);
    lay->addWidget(split_, 1);

    auto* row = new QHBoxLayout();
    input_ = new QLineEdit(this);
    input_->setPlaceholderText(QStringLiteral("Ask the robot… (\"stop\" stops it at once)"));
    input_->setToolTip(QStringLiteral("Enter sends · ↑/↓ past prompts · Esc stops the answer · /new or Ctrl+L: new conversation"));
    busy_ = new QLabel(QStringLiteral("<i>thinking… (Esc stops)</i>"), this);
    busy_->setVisible(false);
    row->addWidget(input_, 1);
    row->addWidget(busy_);
    input_->installEventFilter(new HistoryKeys(this));
    auto* clear = new QShortcut(QKeySequence(QStringLiteral("Ctrl+L")), this);
    QObject::connect(clear, &QShortcut::activated, this, [this]() { if (cb_.on_new) cb_.on_new(); });
    // Esc aborts the model's answer (not the robot: "stop" does that). It replaces the Cancel button (2026-10-18).
    auto* esc = new QShortcut(QKeySequence(Qt::Key_Escape), this);
    QObject::connect(esc, &QShortcut::activated, this, [this]() { if (cb_.on_cancel) cb_.on_cancel(); });
    lay->addLayout(row);

    status_ = new QLabel(this);
    status_->setTextFormat(Qt::RichText);
    lay->addWidget(status_);

    QObject::connect(input_, &QLineEdit::returnPressed, this, [this]() { send_clicked(); });

    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    if (const auto g = settings.value(QStringLiteral("chat_window/geometry")).toByteArray(); not g.isEmpty())
        restoreGeometry(g);
    else
        resize(640, 560);
}

void ChatWindow::closeEvent(QCloseEvent* e)
{
    // The geometry is saved WITHOUT the canvas: the next start opens it again and widens the window by it, so a
    // window saved with the canvas would grow by one canvas on every run.
    hide_canvas();
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    settings.setValue(QStringLiteral("chat_window/geometry"), saveGeometry());
    QWidget::closeEvent(e);
}

void ChatWindow::send_clicked()
{
    const QString t = input_->text().trimmed();
    if (t.isEmpty()) return;
    input_->clear();
    if (history_.empty() or history_.back() != t) history_.push_back(t);
    hist_pos_ = static_cast<int>(history_.size());
    draft_.clear();
    const QString low = t.toLower();
    if (low == QStringLiteral("/new") or low == QStringLiteral("/clear") or low == QStringLiteral("nueva conversación")
        or low == QStringLiteral("nueva conversacion") or low == QStringLiteral("new conversation"))
    { if (cb_.on_new) cb_.on_new(); return; }
    if (cb_.on_send) cb_.on_send(t);
}

void ChatWindow::history_step(int dir)
{
    if (history_.empty()) return;
    const int n = static_cast<int>(history_.size());
    if (hist_pos_ == n) draft_ = input_->text();          // leaving the draft: keep what was being typed
    hist_pos_ = std::clamp(hist_pos_ + dir, 0, n);
    input_->setText(hist_pos_ == n ? draft_ : history_[static_cast<std::size_t>(hist_pos_)]);
    input_->end(false);
}

void ChatWindow::append_html(const QString& html)
{
    if (in_assistant_) end_assistant();
    transcript_->append(html);
    transcript_->verticalScrollBar()->setValue(transcript_->verticalScrollBar()->maximum());
}

void ChatWindow::append_user(const QString& text)
{
    append_html(QStringLiteral("<p><b style='color:#1565c0'>you:</b> %1</p>").arg(text.toHtmlEscaped()));
}

void ChatWindow::begin_assistant()
{
    if (in_assistant_) end_assistant();
    transcript_->append(QStringLiteral("<b style='color:#2e7d32'>robot:</b> "));
    in_assistant_ = true;
}

void ChatWindow::append_token(const QString& token)
{
    if (not in_assistant_) begin_assistant();
    QTextCursor c = transcript_->textCursor();
    c.movePosition(QTextCursor::End);
    QTextCharFormat plain;
    c.insertText(token, plain);
    transcript_->verticalScrollBar()->setValue(transcript_->verticalScrollBar()->maximum());
}

void ChatWindow::end_assistant() { in_assistant_ = false; }

void ChatWindow::append_robot(const QString& text)
{
    append_html(QStringLiteral("<p style='color:#6a1b9a'>&#10230; <i>robot status:</i> %1</p>").arg(text.toHtmlEscaped()));
}

void ChatWindow::append_system(const QString& text)
{
    append_html(QStringLiteral("<p style='color:#757575'><small>%1</small></p>").arg(text.toHtmlEscaped()));
}

void ChatWindow::clear_transcript()
{
    in_assistant_ = false;
    transcript_->clear();
}

void ChatWindow::set_plan(const rc::mission::Mission& m, const QString& ir_json)
{
    canvas_->plan()->set_mission(m, ir_json);
}

void ChatWindow::set_plan_progress(const rc::mission::Progress& p)
{
    canvas_->plan()->set_progress(p);
}

// Opening the canvas WIDENS the window by the canvas and closing it ([×]) gives exactly that width back, so the
// transcript keeps its size either way and the window returns to what it was. The canvas width is remembered.
void ChatWindow::show_canvas()
{
    if (canvas_->isVisible()) return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    const int canvas_w = std::max(320, settings.value(QStringLiteral("chat_window/canvas_width"), 560).toInt());
    const int transcript_w = transcript_->width();
    const int added = canvas_w + split_->handleWidth();
    canvas_->setVisible(true);
    resize(width() + added, height());
    split_->setSizes({transcript_w, canvas_w});
}

void ChatWindow::hide_canvas()
{
    if (not canvas_->isVisible()) return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    settings.setValue(QStringLiteral("chat_window/canvas_width"), canvas_->width());
    // what the canvas takes NOW (the person may have dragged the splitter or resized the window since it opened)
    const int added = canvas_->width() + split_->handleWidth();
    canvas_->setVisible(false);
    resize(std::max(minimumSizeHint().width(), width() - added), height());
}

void ChatWindow::set_busy(bool busy)
{
    busy_->setVisible(busy);   // typing while busy is still allowed: "stop" must always get through
}

void ChatWindow::set_status(const QString& model, const QString& ollama, bool ollama_ok, bool controller_present)
{
    const auto dot = [](bool ok) { return ok ? QStringLiteral("<span style='color:#2e7d32'>&#9679;</span>")
                                             : QStringLiteral("<span style='color:#c62828'>&#9679;</span>"); };
    status_->setText(QStringLiteral("<small>model <b>%1</b> &nbsp;·&nbsp; %2 %3 &nbsp;·&nbsp; %4 controller %5</small>")
                         .arg(model.toHtmlEscaped(), dot(ollama_ok), ollama.toHtmlEscaped(), dot(controller_present),
                              controller_present ? QStringLiteral("present") : QStringLiteral("absent")));
}

}   // namespace rc::chat
