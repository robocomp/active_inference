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

    transcript_ = new QTextBrowser(this);
    transcript_->setOpenExternalLinks(false);
    transcript_->setMinimumSize(480, 360);
    lay->addWidget(transcript_, 1);

    card_ = new QFrame(this);
    card_->setFrameShape(QFrame::StyledPanel);
    card_->setStyleSheet(QStringLiteral("QFrame { background: #fff4d6; border: 1px solid #d9a400; border-radius: 4px; }"
                                        "QLabel { color: #3a2e00; border: none; }"));
    auto* card_lay = new QHBoxLayout(card_);
    card_text_ = new QLabel(card_);
    card_text_->setWordWrap(true);
    card_confirm_ = new QPushButton(QStringLiteral("Confirm"), card_);
    card_cancel_ = new QPushButton(QStringLiteral("Cancel"), card_);
    card_confirm_->setStyleSheet(QStringLiteral("background: #2e7d32; color: white; font-weight: bold; padding: 4px 12px;"));
    card_lay->addWidget(card_text_, 1);
    card_lay->addWidget(card_confirm_);
    card_lay->addWidget(card_cancel_);
    card_->hide();
    lay->addWidget(card_);

    auto* row = new QHBoxLayout();
    input_ = new QLineEdit(this);
    input_->setPlaceholderText(QStringLiteral("Ask the robot… (\"stop\" stops it at once)"));
    input_->setToolTip(QStringLiteral("Enter sends · ↑/↓ past prompts · /new or Ctrl+L: new conversation"));
    busy_ = new QLabel(QStringLiteral("<i>thinking…</i>"), this);
    busy_->setVisible(false);
    cancel_ = new QPushButton(QStringLiteral("Cancel"), this);
    cancel_->setToolTip(QStringLiteral("Stop the model's answer (does not stop the robot — type \"stop\" for that)"));
    cancel_->setEnabled(false);
    plan_ = new PlanView(this);
    plan_btn_ = new QPushButton(QStringLiteral("Plan"), this);
    plan_btn_->setToolTip(QStringLiteral("The mission the robot runs, step by step (none yet)"));
    plan_btn_->setEnabled(false);
    row->addWidget(input_, 1);
    row->addWidget(busy_);
    row->addWidget(plan_btn_);
    row->addWidget(cancel_);
    input_->installEventFilter(new HistoryKeys(this));
    auto* clear = new QShortcut(QKeySequence(QStringLiteral("Ctrl+L")), this);
    QObject::connect(clear, &QShortcut::activated, this, [this]() { if (cb_.on_new) cb_.on_new(); });
    lay->addLayout(row);

    status_ = new QLabel(this);
    status_->setTextFormat(Qt::RichText);
    lay->addWidget(status_);

    QObject::connect(input_, &QLineEdit::returnPressed, this, [this]() { send_clicked(); });
    QObject::connect(plan_btn_, &QPushButton::clicked, this, [this]()
    {
        plan_->show();
        plan_->raise();
        plan_->activateWindow();
    });
    QObject::connect(cancel_, &QPushButton::clicked, this, [this]() { if (cb_.on_cancel) cb_.on_cancel(); });
    QObject::connect(card_confirm_, &QPushButton::clicked, this, [this]() { if (cb_.on_confirm) cb_.on_confirm(); });
    QObject::connect(card_cancel_, &QPushButton::clicked, this, [this]() { if (cb_.on_reject) cb_.on_reject(); });

    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    if (const auto g = settings.value(QStringLiteral("chat_window/geometry")).toByteArray(); not g.isEmpty())
        restoreGeometry(g);
    else
        resize(640, 560);
}

void ChatWindow::closeEvent(QCloseEvent* e)
{
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

void ChatWindow::show_card(const QString& text)
{
    card_text_->setText(text);
    card_->show();
    card_confirm_->setFocus();
}

void ChatWindow::hide_card() { card_->hide(); input_->setFocus(); }

bool ChatWindow::card_visible() const { return card_->isVisible(); }

void ChatWindow::set_plan(const rc::mission::Mission& m, const QString& ir_json)
{
    plan_->set_mission(m, ir_json);
    plan_btn_->setEnabled(true);
    plan_btn_->setToolTip(QStringLiteral("The mission the robot runs, step by step"));
}

void ChatWindow::set_plan_progress(const rc::mission::Progress& p)
{
    plan_->set_progress(p);
    // the button says whether something runs, so a glance at the chat tells it without opening the window
    plan_btn_->setText(p.status == "in_progress" ? QStringLiteral("Plan \u25CF") : QStringLiteral("Plan"));
    plan_btn_->setStyleSheet(p.status == "in_progress" ? QStringLiteral("color: #1565c0; font-weight: bold;") : QString());
}

void ChatWindow::set_busy(bool busy)
{
    cancel_->setEnabled(busy);
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
