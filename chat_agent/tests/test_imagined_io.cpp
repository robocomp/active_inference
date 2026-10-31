// Offline check of imagined_io.h: a 3-sample set round-trips through imagined.json, the PNG has the render size and the
// per-pixel frequencies, and two writes of the same inputs are byte-identical.
#include "../src/imagined_io.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <QJsonArray>
#include <QJsonDocument>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL  %s  (line %d)\n", #c, __LINE__); ++fails; } } while (0)

static std::string slurp(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main()
{
    using namespace rc::card;
    SampleSet set;
    set.seed = 18446744073709551615ull;
    set.drawn = 5; set.rejected = 2;
    for (int i = 0; i < 3; ++i)
    {
        Sample s;
        s.x = 1.0f + i; s.y = 0.5f; s.yaw = 0.1f * i; s.state["w"] = 0.45f + 0.01f * i;
        s.parts.boxes.push_back(rc::nbv::Obstacle{s.x, s.y, 0.5f, 0.5f, s.yaw, 0.0f, 0.9f});
        set.samples.push_back(s);
    }
    std::vector<ImaginedImage> imgs(3);
    for (int i = 0; i < 3; ++i)
    {
        imgs[i].w = 4; imgs[i].h = 2;
        imgs[i].visible.assign(8, 0); imgs[i].occluded.assign(8, 0);
        imgs[i].visible[0] = 1;                       // pixel 0: visible in all 3
        if (i < 2) imgs[i].visible[1] = 1;            // pixel 1: 2 of 3
        if (i == 0) imgs[i].occluded[2] = 1;          // pixel 2: occluded in 1 of 3
        imgs[i].visible_frac = 0.5f; imgs[i].fill_max = 0.25f;
    }
    std::array<std::vector<ImaginedScan>, 2> scans;
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < 3; ++i)
        {
            ImaginedScan sc;
            sc.on_target.push_back({1.23456f + i, 2.0f, 0.5f});
            sc.on_other.push_back({0.0f, 0.0f, 1.0f});
            sc.ring_elev_room_deg = {-10.f, 5.f};
            sc.modelled = true;
            scans[k].push_back(sc);
        }
    McResult mc; mc.p_notice_mean = 0.8f; mc.n = 3;

    const auto dir = std::filesystem::temp_directory_path() / "imagined_io_test_a";
    const auto dir2 = std::filesystem::temp_directory_path() / "imagined_io_test_b";
    CHECK(rc::chat::write_imagined(dir, set, imgs, scans, mc, "a chair"));
    CHECK(rc::chat::write_imagined(dir2, set, imgs, scans, mc, "a chair"));
    const std::string j = slurp(dir / "imagined.json");
    CHECK(not j.empty());
    CHECK(j == slurp(dir2 / "imagined.json"));        // Review Focus 5: byte-identical
    CHECK(slurp(dir / "imagined_zed.png") == slurp(dir2 / "imagined_zed.png"));

    const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(j));
    CHECK(doc.isObject());
    const auto o = doc.object();
    CHECK(o["caption"].toString() == "a chair");
    CHECK(o["seed"].toString() == "18446744073709551615");
    CHECK(o["rejected"].toInt() == 2);
    CHECK(o["samples"].toArray().size() == 3);
    CHECK(std::abs(o["samples"].toArray()[2].toObject()["x"].toDouble() - 3.0) < 1e-9);
    CHECK(o["samples"].toArray()[1].toObject()["parts"].toArray().size() == 1);
    CHECK(o["lidar"].toArray().size() == 2);
    const auto pt = o["lidar"].toArray()[0].toArray()[0].toObject()["on_target"].toArray()[0].toArray();
    CHECK(pt.size() == 3 and std::abs(pt[0].toDouble() - 1.235) < 1e-9);   // rounded to mm
    CHECK(std::abs(o["mc"].toObject()["p_notice_mean"].toDouble() - 0.8) < 1e-9);
    CHECK(o["image"].toObject()["w"].toInt() == 4);

    QImage png(QString::fromStdString((dir / "imagined_zed.png").string()));
    CHECK(png.width() == 4 and png.height() == 2);
    const QRgb p0 = png.pixel(0, 0), p1 = png.pixel(1, 0), p2 = png.pixel(2, 0), p3 = png.pixel(3, 0);
    CHECK(qRed(p0) == 255 and qGreen(p0) == 0 and qBlue(p0) == 0);
    CHECK(qRed(p1) == 170 and qGreen(p1) == 0);                            // 2/3 of 255
    CHECK(qRed(p2) == 0 and qGreen(p2) == 85);                             // 1/3 of 255
    CHECK(qRed(p3) == 0 and qGreen(p3) == 0);

    std::filesystem::remove_all(dir); std::filesystem::remove_all(dir2);
    std::puts(fails == 0 ? "all imagined_io checks passed" : "imagined_io checks FAILED");
    return fails ? 1 : 0;
}
