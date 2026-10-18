#include "canvas_panel.h"

#include "../../common/viewers/gl_mesh_viewer.h"

#include <QFontDatabase>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QPixmap>
#include <QVBoxLayout>

namespace rc::chat
{

CanvasPanel::CanvasPanel(Callbacks cb, QWidget* parent) : QWidget(parent), cb_(std::move(cb))
{
    auto* lay = new QVBoxLayout(this);
    lay->setContentsMargins(0, 0, 0, 0);

    auto* head = new QHBoxLayout();
    title_ = new QLabel(this);
    title_->setTextFormat(Qt::RichText);
    code_btn_ = new QPushButton(QStringLiteral("</>"), this);
    code_btn_->setCheckable(true);
    code_btn_->setToolTip(QStringLiteral("Show the script that drew this"));
    save_btn_ = new QPushButton(QStringLiteral("Save"), this);
    save_btn_->setToolTip(QStringLiteral("Keep this script in the canvas library, so it can be asked for by name"));
    auto* close = new QPushButton(QStringLiteral("×"), this);
    close->setToolTip(QStringLiteral("Close the canvas"));
    for (auto* b : {code_btn_, save_btn_, close}) b->setFlat(true);
    head->addWidget(title_, 1);
    head->addWidget(code_btn_);
    head->addWidget(save_btn_);
    head->addWidget(close);
    lay->addLayout(head);

    stack_ = new QStackedWidget(this);
    picture_ = new QLabel(this);
    picture_->setAlignment(Qt::AlignCenter);
    picture_->setMinimumSize(160, 120);
    picture_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);   // the pixmap must not grow the panel
    picture_->setWordWrap(true);
    stack_->addWidget(picture_);
    plan_ = new PlanView();          // a top-level window by default; adding it to the stack reparents it as a page
    stack_->addWidget(plan_);
    model_ = new rc::viewers::GLMeshViewer();   // the same: a page, not a window
    model_->setMinimumSize(160, 120);
    stack_->addWidget(model_);
    lay->addWidget(stack_, 1);
    caption_ = new QLabel(this);
    caption_->setWordWrap(true);
    caption_->setVisible(false);
    lay->addWidget(caption_);

    code_ = new QPlainTextEdit(this);
    code_->setReadOnly(true);
    code_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    code_->setVisible(false);
    lay->addWidget(code_, 1);

    QObject::connect(code_btn_, &QPushButton::toggled, this, [this](bool on) { code_->setVisible(on); });
    QObject::connect(close, &QPushButton::clicked, this, [this]() { if (cb_.on_close) cb_.on_close(); });
    QObject::connect(save_btn_, &QPushButton::clicked, this, [this]()
    {
        bool ok = false;
        const QString name = QInputDialog::getText(this, QStringLiteral("Save drawing script"),
                                                   QStringLiteral("Name (letters, digits, _):"), QLineEdit::Normal,
                                                   QString(), &ok).trimmed();
        if (ok and not name.isEmpty() and cb_.on_save) cb_.on_save(name);
    });
    save_btn_->setEnabled(false);
    code_btn_->setEnabled(false);
}

void CanvasPanel::set_title(const QString& title, bool live)
{
    caption_->setVisible(false);   // only the 3-D page has one; show_model sets it after this
    title_text_ = title;
    live_ = live;
    title_->setText(QStringLiteral("<b>%1</b>%2").arg(title.toHtmlEscaped(),
        live ? QStringLiteral(" &nbsp;<span style='color:#c62828'>&#9679;</span> <small>live</small>") : QString()));
}

void CanvasPanel::set_live(bool live) { set_title(title_text_, live); }

void CanvasPanel::show_drawing(const QString& title, const QImage& image, bool live, const QString& source, bool savable)
{
    set_title(title, live);
    code_->setPlainText(source);
    code_btn_->setEnabled(not source.isEmpty());
    save_btn_->setEnabled(savable);
    stack_->setCurrentWidget(picture_);
    update_image(image);
}

void CanvasPanel::update_image(const QImage& image)
{
    image_ = image;
    rescale();
}

void CanvasPanel::show_message(const QString& title, const QString& text)
{
    set_title(title, false);
    image_ = QImage();
    stack_->setCurrentWidget(picture_);
    picture_->setPixmap(QPixmap());
    picture_->setText(QStringLiteral("<span style='color:#757575'>%1</span>").arg(text.toHtmlEscaped().replace('\n', "<br>")));
}

void CanvasPanel::show_plan()
{
    set_title(QStringLiteral("Mission plan"), true);
    code_->setPlainText(QString());
    code_btn_->setChecked(false);
    code_btn_->setEnabled(false);
    save_btn_->setEnabled(false);
    stack_->setCurrentWidget(plan_);
}

void CanvasPanel::show_model(const QString& title, std::span<const QVector3D> triangles, const QString& caption)
{
    set_title(title, false);
    code_->setPlainText(QString());
    code_btn_->setChecked(false);
    code_btn_->setEnabled(false);
    save_btn_->setEnabled(false);
    stack_->setCurrentWidget(model_);
    model_->set_mesh(triangles);
    caption_->setText(QStringLiteral("<small>%1 &nbsp;·&nbsp; drag to orbit, wheel to zoom</small>").arg(caption.toHtmlEscaped()));
    caption_->setVisible(true);
}

bool CanvasPanel::showing_plan() const { return stack_->currentWidget() == plan_; }

QSize CanvasPanel::drawing_size() const { return stack_->size(); }

void CanvasPanel::rescale()
{
    if (image_.isNull()) return;
    picture_->setText(QString());
    picture_->setPixmap(QPixmap::fromImage(image_).scaled(picture_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void CanvasPanel::resizeEvent(QResizeEvent* e)
{
    QWidget::resizeEvent(e);
    rescale();
}

}   // namespace rc::chat
