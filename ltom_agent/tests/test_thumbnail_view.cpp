// What "View data" shows for a retained object: its thumbnails, or a plain statement that it has none. Offscreen Qt.
#include "../src/thumbnail_view.h"

#include <QApplication>
#include <QBuffer>
#include <QImage>
#include <QLabel>

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
    std::println("{}", failures == 0 ? "ALL PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
