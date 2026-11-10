#include "thumbnail_view.h"
#include "../../common/thumbs/thumb_select.h"

#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <ranges>
#include <utility>
#include <vector>

namespace ltom
{
namespace
{
using rc::thumbs::bin_centre_rad;
using rc::thumbs::kViewBins;

constexpr int kSlot = 96;           // the side of the box a view is shown in
constexpr double kViewRing = 224.;  // distance of a view from the centre: the least at which twelve kSlot boxes do not touch
constexpr double kRose = 128.;      // radius of a wedge whose view is the object's best
constexpr int kHalfCanvas = 284;
constexpr double kWedgeDeg = 360. / kViewBins;

double to_deg(float rad) { return static_cast<double>(rad) * 180. / std::numbers::pi; }

/// A direction on screen: 0 (the object's front) is up, positive angles go counter-clockwise, as in the atlas viewer.
QPointF on_screen(double centre_deg, double radius)
{
    const double t = centre_deg * std::numbers::pi / 180.;
    return {-radius * std::sin(t), -radius * std::cos(t)};
}

struct Wedge { int bin; float score; };

class RoseCanvas : public QWidget
{
public:
    RoseCanvas(std::vector<Wedge> wedges, QWidget* parent) : QWidget(parent), wedges_(std::move(wedges))
    {
        setFixedSize(2 * kHalfCanvas, 2 * kHalfCanvas);
        for (const auto& w : wedges_) best_ = std::max(best_, w.score);
    }
    void select(int bin) { selected_ = bin; update(); }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.translate(rect().center());
        const QColor ink = palette().color(QPalette::WindowText);
        QColor faint = ink;
        faint.setAlphaF(0.25);

        p.setPen(QPen(faint, 1));
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(QPointF(0, 0), kRose, kRose);
        p.drawEllipse(QPointF(0, 0), kRose / 2., kRose / 2.);       // a quarter of the best view's area
        for (int bin : std::views::iota(0, kViewBins))
            p.drawLine(QPointF(0, 0), on_screen(to_deg(bin_centre_rad(bin, kViewBins)) - kWedgeDeg / 2., kRose));

        // Qt angles: 0 at three o'clock, counter-clockwise; ours: 0 at twelve.
        p.setPen(QPen(ink, 0.8));
        for (const auto& w : wedges_)
        {
            if (not (w.score > 0.f and best_ > 0.f)) continue;
            const double r = kRose * std::sqrt(static_cast<double>(w.score / best_));     // area in proportion to the score
            const QRectF box(-r, -r, 2. * r, 2. * r);
            p.setBrush(w.bin == selected_ ? QColor(255, 120, 0, 230) : QColor(255, 170, 0, 170));
            p.drawPie(box, static_cast<int>(std::lround((90. + to_deg(bin_centre_rad(w.bin, kViewBins)) - kWedgeDeg / 2.) * 16.)),
                      static_cast<int>(std::lround(kWedgeDeg * 16.)));
        }

        const QColor front(220, 40, 40);
        p.setPen(front);
        p.drawText(QRectF(-30., -kRose - 18., 60., 16.), Qt::AlignCenter, "front");
    }

private:
    std::vector<Wedge> wedges_;
    float best_ = 0.f;
    int selected_ = -1;
};

class ViewPicture : public QLabel
{
public:
    ViewPicture(std::function<void()> on_click, QWidget* parent) : QLabel(parent), on_click_(std::move(on_click)) {}

protected:
    void mousePressEvent(QMouseEvent*) override { on_click_(); }

private:
    std::function<void()> on_click_;
};

QString numbers_of(int bin, const Thumbnail& t)
{
    const int centre = static_cast<int>(std::lround(to_deg(bin_centre_rad(bin, kViewBins))));
    const int half = static_cast<int>(std::lround(kWedgeDeg / 2.));
    return QString("%1° to %2° from the front: best view shows %3 pixels of the object")
        .arg(centre - half).arg(centre + half).arg(static_cast<double>(t.score), 0, 'f', 0);
}
}   // namespace

QWidget* make_thumbnail_view(const Entry& e, QWidget* parent)
{
    auto* view = new QWidget(parent);
    auto* column = new QVBoxLayout(view);
    column->addWidget(new QLabel(QString("<b>%1</b>   uid %2").arg(QString::fromStdString(e.name)).arg(e.uid), view));

    std::vector<std::pair<int, QPixmap>> views;
    std::vector<Wedge> wedges;
    for (const auto& [bin, thumb] : e.thumbs)
    {
        QPixmap image;
        if (not image.loadFromData(thumb.jpeg.data(), static_cast<uint>(thumb.jpeg.size()), "JPG")) continue;
        if (std::max(image.width(), image.height()) > kSlot) image = image.scaled(kSlot, kSlot, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        views.emplace_back(bin, std::move(image));
        wedges.push_back({bin, thumb.score});
    }
    if (views.empty())
    {
        column->addWidget(new QLabel("No thumbnails retained for this object yet.", view));
        return view;
    }

    auto* rose = new RoseCanvas(std::move(wedges), view);
    column->addWidget(rose, 0, Qt::AlignHCenter);
    auto* readout = new QLabel("Each wedge is one direction, its area the size of the best view from there. Click a view for its numbers.", view);
    column->addWidget(readout);

    for (auto& [bin, image] : views)
    {
        auto* picture = new ViewPicture([rose, readout, bin = bin, numbers = numbers_of(bin, e.thumbs.at(bin))]
                                        { rose->select(bin); readout->setText(numbers); }, rose);
        picture->setPixmap(image);
        picture->setProperty("bin", bin);
        picture->resize(image.size());
        const QPointF at = QPointF(rose->rect().center()) + on_screen(to_deg(bin_centre_rad(bin, kViewBins)), kViewRing);
        QRect placed(picture->rect());
        placed.moveCenter(at.toPoint());
        picture->setGeometry(placed);
    }
    return view;
}
}   // namespace ltom
