// What "View data" shows for a retained object: its thumbnails, or a plain statement that it has none. Offscreen Qt.
#include "../src/thumbnail_view.h"

#include <QApplication>
#include <QBuffer>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>

#include <cstdlib>
#include <print>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

static std::vector<std::uint8_t> jpeg_of(int w, int h)
{
    QImage img(w, h, QImage::Format_RGB888);
    img.fill(Qt::red);
    QByteArray b;
    QBuffer buf(&b);
    buf.open(QIODevice::WriteOnly);
    img.save(&buf, "JPG");
    return {b.begin(), b.end()};
}
static int shown_images(QWidget* w)
{
    int n = 0;
    for (auto* l : w->findChildren<QLabel*>()) n += not l->pixmap().isNull();
    return n;
}
static bool says(QWidget* w, const QString& text)
{
    for (auto* l : w->findChildren<QLabel*>()) if (l->text().contains(text)) return true;
    return false;
}

static QLabel* picture_of(QWidget* w, int bin)
{
    for (auto* l : w->findChildren<QLabel*>()) if (l->property("bin").isValid() and l->property("bin").toInt() == bin) return l;
    return nullptr;
}
// Where a view sits on the rose, relative to the rose's centre (screen axes: +x right, +y down).
static QPoint offset_of(QLabel* l) { return l->geometry().center() - l->parentWidget()->rect().center(); }
static void click(QWidget* w)
{
    const QPointF at(5, 5);
    QMouseEvent press(QEvent::MouseButtonPress, at, w->mapToGlobal(at), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(w, &press);
}

static void each_view_sits_at_its_direction()
{
    ltom::Entry e;
    e.name = "chair_9";
    e.thumbs[6]  = {10.f, jpeg_of(96, 60)};     //   0..30 deg: front, to the left
    e.thumbs[5]  = {10.f, jpeg_of(96, 60)};     // -30..0  deg: front, to the right
    e.thumbs[11] = {10.f, jpeg_of(96, 60)};     // 150..180: behind, to the left
    e.thumbs[0]  = {10.f, jpeg_of(96, 60)};     // -180..-150: behind, to the right
    e.thumbs[9]  = {10.f, jpeg_of(96, 60)};     //  90..120: left side
    auto* v = ltom::make_thumbnail_view(e, nullptr);
    const auto at = [&](int bin) { return offset_of(picture_of(v, bin)); };
    CHECK(picture_of(v, 6) and picture_of(v, 5) and picture_of(v, 11) and picture_of(v, 0) and picture_of(v, 9));
    CHECK(at(6).x() < 0 and at(6).y() < 0);
    CHECK(at(5).x() > 0 and at(5).y() < 0);
    CHECK(at(11).x() < 0 and at(11).y() > 0);
    CHECK(at(0).x() > 0 and at(0).y() > 0);
    CHECK(at(9).x() < 0 and std::abs(at(9).y()) < std::abs(at(9).x()));
    delete v;
}

static void no_two_directions_overlap()
{
    ltom::Entry e;
    e.name = "chair_9";
    for (int b = 0; b < 12; ++b) e.thumbs[b] = {10.f, jpeg_of(96, 96)};
    auto* v = ltom::make_thumbnail_view(e, nullptr);
    CHECK(shown_images(v) == 12);
    for (int a = 0; a < 12; ++a)
        for (int b = a + 1; b < 12; ++b)
            CHECK(not picture_of(v, a)->geometry().intersects(picture_of(v, b)->geometry()));
    delete v;
}

static void clicking_a_view_gives_its_numbers()
{
    ltom::Entry e;
    e.name = "chair_9";
    e.thumbs[6] = {2310.f, jpeg_of(96, 60)};
    e.thumbs[9] = {640.f, jpeg_of(96, 60)};
    auto* v = ltom::make_thumbnail_view(e, nullptr);
    CHECK(not says(v, "2310"));
    click(picture_of(v, 6));
    CHECK(says(v, "0° to 30°"));
    CHECK(says(v, "2310"));
    click(picture_of(v, 9));
    CHECK(says(v, "90° to 120°"));
    CHECK(says(v, "640"));
    CHECK(not says(v, "2310"));
    delete v;
}

int main(int argc, char** argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    ltom::Entry e;
    e.uid = 42; e.cls = "chair"; e.name = "chair_9";
    e.thumbs[1] = {10.f, jpeg_of(96, 60)};
    e.thumbs[5] = {20.f, jpeg_of(60, 96)};
    auto* full = ltom::make_thumbnail_view(e, nullptr);
    CHECK(shown_images(full) == 2);
    CHECK(says(full, "chair_9"));

    e.thumbs[6] = {30.f, {1, 2, 3}};      // damaged image: must not take the popup down, and is not shown
    auto* damaged = ltom::make_thumbnail_view(e, nullptr);
    CHECK(shown_images(damaged) == 2);

    e.thumbs.clear();
    auto* empty = ltom::make_thumbnail_view(e, nullptr);
    CHECK(shown_images(empty) == 0);
    CHECK(says(empty, "No thumbnails"));

    delete full; delete damaged; delete empty;
    each_view_sits_at_its_direction();
    no_two_directions_overlap();
    clicking_a_view_gives_its_numbers();

    std::println("{}", failures == 0 ? "ALL PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
