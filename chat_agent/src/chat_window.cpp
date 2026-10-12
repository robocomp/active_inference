#include "chat_window.h"

#include <QCloseEvent>
#include <QHBoxLayout>
#include <QScrollBar>
#include <QSettings>
#include <QTextCursor>
#include <QVBoxLayout>

namespace rc::chat
{

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
    send_ = new QPushButton(QStringLiteral("Send"), this);
    cancel_ = new QPushButton(QStringLiteral("Cancel"), this);
    new_ = new QPushButton(QStringLiteral("New conversation"), this);
    cancel_->setEnabled(false);
    row->addWidget(input_, 1);
    row->addWidget(send_);
    row->addWidget(cancel_);
    row->addWidget(new_);
    lay->addLayout(row);

    status_ = new QLabel(this);
    status_->setTextFormat(Qt::RichText);
    lay->addWidget(status_);

    QObject::connect(send_, &QPushButton::clicked, this, [this]() { send_clicked(); });
    QObject::connect(input_, &QLineEdit::returnPressed, this, [this]() { send_clicked(); });
    QObject::connect(cancel_, &QPushButton::clicked, this, [this]() { if (cb_.on_cancel) cb_.on_cancel(); });
    QObject::connect(new_, &QPushButton::clicked, this, [this]() { if (cb_.on_new) cb_.on_new(); });
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
    if (cb_.on_send) cb_.on_send(t);
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

void ChatWindow::set_busy(bool busy)
{
    cancel_->setEnabled(busy);
    send_->setEnabled(true);   // sending while busy is still allowed: "stop" must always get through
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
