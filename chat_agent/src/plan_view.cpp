#include "plan_view.h"

#include <chrono>
#include <cmath>
#include <numbers>

#include <QFontDatabase>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QPainter>
#include <QPainterPath>
#include <QSettings>
#include <QVBoxLayout>

namespace rc::chat
{

namespace
{
using SS = rc::mission::Interpreter::StepState;

constexpr int kRowH = 30, kIndent = 22, kMargin = 12, kDot = 14;
constexpr std::uint64_t kFlashMs = 700;    ///< how long a state change stays highlighted
constexpr double kPulsePeriodS = 1.2;      ///< the running leaf's halo
constexpr double kSpinPeriodS = 1.6;       ///< a running repeat's arc

QColor state_color(SS s)
{
    switch (s)
    {
        case SS::Pending: return QColor(0x9e, 0x9e, 0x9e);
        case SS::Running: return QColor(0x15, 0x65, 0xc0);
        case SS::Satisfied: return QColor(0x2e, 0x7d, 0x32);
        case SS::Failed: return QColor(0xc6, 0x28, 0x28);
        case SS::Skipped: return QColor(0x8d, 0x6e, 0x63);
        case SS::Cancelled: return QColor(0x75, 0x75, 0x75);
    }
    return Qt::gray;
}

QColor headline_color(const std::string& status)
{
    if (status == "in_progress") return state_color(SS::Running);
    if (status == "completed") return state_color(SS::Satisfied);
    if (status.starts_with("failed")) return state_color(SS::Failed);
    return state_color(SS::Cancelled);
}

QString qs(const std::string& s) { return QString::fromStdString(s); }
}   // namespace

// ── the painted rows ────────────────────────────────────────────────────────────────────────────────────────────
class PlanCanvas : public QWidget
{
public:
    explicit PlanCanvas(PlanView* view) : QWidget(view), view_(view) { setAutoFillBackground(false); }

    QSize sizeHint() const override { return {560, std::max(1, int(view_->model_.rows.size())) * kRowH + 2 * kMargin}; }
    void sync_size() { setMinimumHeight(sizeHint().height()); updateGeometry(); }
    QRect row_rect(int i) const { return {0, kMargin + i * kRowH, width(), kRowH}; }

    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), palette().base());
        const auto& md = view_->model_;
        const std::uint64_t now = view_->now();
        const double t = double(now % 3'600'000) / 1000.0;
        QFont mono = QFontDatabase::systemFont(QFontDatabase::FixedFont);
        QFont small = font();
        small.setPointSizeF(small.pointSizeF() * 0.9);

        // composite brackets first, so rows paint over them
        for (std::size_t i = 0; i < md.rows.size(); ++i)
        {
            const auto& r = md.rows[i];
            if (not r.composite) continue;
            std::size_t last = i;
            while (last + 1 < md.rows.size() and md.rows[last + 1].depth > r.depth) ++last;
            if (last == i) continue;
            const int x = kMargin + r.depth * kIndent + kDot / 2;
            QColor c = state_color(r.state);
            c.setAlpha(110);
            p.setPen(QPen(c, 2));
            p.drawLine(x, row_rect(int(i)).center().y() + kDot / 2 + 2, x, row_rect(int(last)).center().y());
        }

        for (std::size_t i = 0; i < md.rows.size(); ++i)
        {
            const auto& r = md.rows[i];
            const QRect rr = row_rect(int(i));
            const QColor c = state_color(r.state);
            const bool driving = int(i) == md.running_row();

            // background: the driving row, then a fading flash of a state change
            if (driving) p.fillRect(rr, QColor(c.red(), c.green(), c.blue(), 28));
            if (i < view_->flash_at_.size() and view_->flash_at_[i] != 0 and now - view_->flash_at_[i] < kFlashMs)
            {
                const double k = 1.0 - double(now - view_->flash_at_[i]) / double(kFlashMs);
                p.fillRect(rr, QColor(c.red(), c.green(), c.blue(), int(110 * k)));
            }

            const int x = kMargin + r.depth * kIndent;
            const QPointF ctr(x + kDot / 2.0, rr.center().y());
            draw_dot(p, r, ctr, c, driving, t);

            // the step's own line of the text form
            QFont f = mono;
            f.setBold(r.state == SS::Running);
            p.setFont(f);
            p.setPen(r.state == SS::Pending ? palette().color(QPalette::PlaceholderText) : palette().color(QPalette::Text));
            const QString ann = qs(md.annotation(r, now));
            const int ann_w = QFontMetrics(small).horizontalAdvance(ann) + 2 * kMargin;
            const QRect text_r(x + kDot + 10, rr.top(), std::max(40, rr.width() - (x + kDot + 10) - ann_w), rr.height());
            p.drawText(text_r, Qt::AlignVCenter | Qt::AlignLeft,
                       QFontMetrics(f).elidedText(qs(r.line), Qt::ElideRight, text_r.width()));

            p.setFont(small);
            p.setPen(r.state == SS::Pending ? palette().color(QPalette::PlaceholderText) : c);
            p.drawText(QRect(rr.right() - ann_w, rr.top(), ann_w - kMargin, rr.height()), Qt::AlignVCenter | Qt::AlignRight, ann);

            // a bounded running leaf fills a bar along the bottom of its row
            if (const auto fr = md.fraction(r, now); fr.has_value())
            {
                const QRectF track(text_r.left(), rr.bottom() - 4, rr.width() - text_r.left() - kMargin, 3);
                p.fillRect(track, QColor(c.red(), c.green(), c.blue(), 40));
                p.fillRect(QRectF(track.left(), track.top(), track.width() * *fr, track.height()), c);
            }
        }
        if (md.rows.empty())
        {
            p.setPen(palette().color(QPalette::PlaceholderText));
            p.drawText(rect(), Qt::AlignCenter, QStringLiteral("No mission yet"));
        }
    }

private:
    static void draw_dot(QPainter& p, const plan::Row& r, QPointF c, QColor col, bool driving, double t)
    {
        const double rad = kDot / 2.0;
        switch (r.state)
        {
            case SS::Pending:
                p.setPen(QPen(col, 1.6));
                p.setBrush(Qt::NoBrush);
                p.drawEllipse(c, rad - 1, rad - 1);
                return;
            case SS::Running:
                if (r.composite)
                {
                    // a spinning arc: the loop is turning
                    p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 60), 2.2));
                    p.setBrush(Qt::NoBrush);
                    p.drawEllipse(c, rad - 1, rad - 1);
                    p.setPen(QPen(col, 2.2, Qt::SolidLine, Qt::RoundCap));
                    const int start = int(-360.0 * 16 * std::fmod(t / kSpinPeriodS, 1.0));
                    p.drawArc(QRectF(c.x() - rad + 1, c.y() - rad + 1, 2 * rad - 2, 2 * rad - 2), start, 110 * 16);
                    return;
                }
                if (driving)
                {
                    const double ph = std::fmod(t / kPulsePeriodS, 1.0);   // a halo that grows and fades
                    p.setPen(Qt::NoPen);
                    p.setBrush(QColor(col.red(), col.green(), col.blue(), int(120 * (1.0 - ph))));
                    p.drawEllipse(c, rad + 7 * ph, rad + 7 * ph);
                }
                p.setPen(Qt::NoPen);
                p.setBrush(col);
                p.drawEllipse(c, rad - 1, rad - 1);
                return;
            default: break;
        }
        p.setPen(Qt::NoPen);
        p.setBrush(col);
        p.drawEllipse(c, rad - 1, rad - 1);
        p.setPen(QPen(Qt::white, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        p.setBrush(Qt::NoBrush);
        const double k = rad * 0.45;
        QPainterPath g;
        switch (r.state)
        {
            case SS::Satisfied:   // ✓
                g.moveTo(c + QPointF(-k, 0));
                g.lineTo(c + QPointF(-k * 0.25, k * 0.7));
                g.lineTo(c + QPointF(k, -k * 0.6));
                break;
            case SS::Failed:      // ✗
                g.moveTo(c + QPointF(-k, -k));
                g.lineTo(c + QPointF(k, k));
                g.moveTo(c + QPointF(k, -k));
                g.lineTo(c + QPointF(-k, k));
                break;
            case SS::Skipped:     // ≫
                g.moveTo(c + QPointF(-k, -k));
                g.lineTo(c + QPointF(0, 0));
                g.lineTo(c + QPointF(-k, k));
                g.moveTo(c + QPointF(0, -k));
                g.lineTo(c + QPointF(k, 0));
                g.lineTo(c + QPointF(0, k));
                break;
            default:              // cancelled: —
                g.moveTo(c + QPointF(-k, 0));
                g.lineTo(c + QPointF(k, 0));
                break;
        }
        p.drawPath(g);
    }

    PlanView* view_;
};

// ── the window ──────────────────────────────────────────────────────────────────────────────────────────────────
PlanView::PlanView(QWidget* parent) : QWidget(parent, Qt::Window)
{
    now_ = []
    {
        return std::uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch()).count());
    };
    setWindowTitle(QStringLiteral("Mission plan"));
    auto* lay = new QVBoxLayout(this);

    auto* head = new QHBoxLayout();
    title_ = new QLabel(this);
    title_->setTextFormat(Qt::RichText);
    who_ = new QLabel(this);
    who_->setTextFormat(Qt::RichText);
    head->addWidget(title_, 1);
    head->addWidget(who_);
    lay->addLayout(head);

    tabs_ = new QTabWidget(this);
    canvas_ = new PlanCanvas(this);
    scroll_ = new QScrollArea(this);
    scroll_->setWidget(canvas_);
    scroll_->setWidgetResizable(true);
    scroll_->setFrameShape(QFrame::NoFrame);
    tabs_->addTab(scroll_, QStringLiteral("Plan"));
    const auto read_only = [this]
    {
        auto* e = new QPlainTextEdit(this);
        e->setReadOnly(true);
        e->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        e->setLineWrapMode(QPlainTextEdit::NoWrap);
        return e;
    };
    text_ = read_only();
    json_ = read_only();
    tabs_->addTab(text_, QStringLiteral("Text"));
    tabs_->addTab(json_, QStringLiteral("JSON"));
    lay->addWidget(tabs_, 1);

    event_ = new QLabel(this);
    event_->setWordWrap(true);
    event_->setTextFormat(Qt::PlainText);
    lay->addWidget(event_);

    timer_.setInterval(33);   // ~30 fps, only while something moves (animating())
    QObject::connect(&timer_, &QTimer::timeout, this, [this] { tick(); });

    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    if (const auto g = settings.value(QStringLiteral("plan_view/geometry")).toByteArray(); not g.isEmpty())
        restoreGeometry(g);
    else
        resize(620, 420);
    refresh_header();
}

void PlanView::set_mission(const rc::mission::Mission& m, const QString& ir_json)
{
    model_ = plan::Model::from(m);
    has_mission_ = true;
    flash_at_.assign(model_.rows.size(), 0);
    last_running_ = -1;
    text_->setPlainText(qs(rc::mission::print_text(m)));
    const auto doc = QJsonDocument::fromJson(ir_json.toUtf8());
    json_->setPlainText(doc.isNull() ? ir_json : QString::fromUtf8(doc.toJson(QJsonDocument::Indented)));
    canvas_->sync_size();
    refresh_header();
    canvas_->update();
    if (isVisible() and animating()) timer_.start();
}

void PlanView::set_progress(const rc::mission::Progress& p)
{
    const std::uint64_t now = now_();
    for (const std::size_t i : model_.apply(p))
        if (i < flash_at_.size()) flash_at_[i] = now;
    if (const int r = model_.running_row(); r != last_running_ and r >= 0)
    {
        const QRect rr = canvas_->row_rect(r);
        scroll_->ensureVisible(rr.center().x(), rr.center().y(), 0, kRowH);
    }
    last_running_ = model_.running_row();
    refresh_header();
    canvas_->update();
    if (isVisible() and animating() and not timer_.isActive()) timer_.start();
}

void PlanView::refresh_header()
{
    const std::uint64_t now = now_();
    const QString name = model_.title.empty() ? QStringLiteral("(untitled mission)") : qs(model_.title).toHtmlEscaped();
    title_->setText(QStringLiteral("<b>%1</b> &nbsp;<span style='color:%2'>%3</span>")
                        .arg(name, headline_color(model_.status).name(), qs(model_.headline(now)).toHtmlEscaped()));
    who_->setText(model_.serial == 0 ? QString()
                                     : QStringLiteral("<small>%1 · #%2</small>")
                                           .arg(qs(model_.source).toHtmlEscaped(), QString::number(model_.serial)));
    // "<step>|<state>|<last event>": the last event is what a person wants to read
    const auto bar = model_.line.rfind('|');
    event_->setText(qs(bar == std::string::npos ? model_.line : model_.line.substr(bar + 1)));
}

bool PlanView::animating() const
{
    if (model_.status == "in_progress") return true;
    const std::uint64_t now = now_();
    return std::ranges::any_of(flash_at_, [now](std::uint64_t f) { return f != 0 and now - f < kFlashMs; });
}

void PlanView::tick()
{
    refresh_header();   // the elapsed clock
    canvas_->update();
    if (not animating()) timer_.stop();
}

void PlanView::showEvent(QShowEvent* e)
{
    QWidget::showEvent(e);
    if (animating()) timer_.start();
}

void PlanView::hideEvent(QHideEvent* e)
{
    timer_.stop();
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chat_agent"));
    settings.setValue(QStringLiteral("plan_view/geometry"), saveGeometry());
    QWidget::hideEvent(e);
}

}   // namespace rc::chat
