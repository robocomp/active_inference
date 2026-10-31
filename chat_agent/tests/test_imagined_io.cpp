// Offline check of imagined_io.h: a 3-sample set round-trips through imagined.json, the PNG has the render size and the
// per-pixel frequencies, and two writes of the same inputs are byte-identical.
#include "../src/imagined_io.h"

#include <clocale>
#include <cstdio>
#include <limits>
#include <fstream>
#include <iostream>
#include <sstream>

#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>

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
    std::setlocale(LC_ALL, "");   // the agents run under es_ES: the writer must not care
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
            sc.modelled = k == 0;
            sc.mount_xyz = {0.1f, 0.2f, 1.075f}; sc.bearing_rad = 0.5f;
            if (k == 1) sc.notes = {"sensor inside a part: no returns"};
            if (k == 0 and i == 2)   // 1200 finite target points + 2 non-finite: capped, counted, skipped
            {
                sc.on_target.clear();
                for (int n = 0; n < 1200; ++n) sc.on_target.push_back({0.001f * n, 1.0f, 0.5f});
                sc.on_target.push_back({std::numeric_limits<float>::quiet_NaN(), 0, 0});
                sc.on_target.push_back({0, std::numeric_limits<float>::infinity(), 0});
            }
            scans[k].push_back(sc);
        }
    McResult mc; mc.p_notice_mean = 0.8f; mc.p_recognise_mean = 0.25f; mc.p_recognise_sd = 0.05f; mc.n = 3;
    set.notes = {"a note"};

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
    CHECK(std::abs(o["mc"].toObject()["p_recognise_mean"].toDouble() - 0.25) < 1e-9 and o["mc"].toObject()["n"].toInt() == 3);
    CHECK(o["notes"].toArray().size() == 1 and o["notes"].toArray()[0].toString() == "a note");
    {
        const auto s0 = o["lidar"].toArray()[0].toArray(), s1 = o["lidar"].toArray()[1].toArray();
        CHECK(s0[0].toObject()["modelled"].toBool() and not s1[0].toObject()["modelled"].toBool());
        CHECK(s1[0].toObject()["notes"].toArray()[0].toString() == "sensor inside a part: no returns");
        const auto big = s0[2].toObject();
        CHECK(big["on_target"].toArray().size() <= 500 and big["on_target"].toArray().size() > 100);   // the cap
        CHECK(big["n_total_target"].toInt() == 1200 and big["n_total_other"].toInt() == 1);
        CHECK(big["notes"].toArray().size() == 1 and big["notes"].toArray()[0].toString().startsWith("2 non-finite"));
        CHECK(std::abs(big["mount_xyz"].toArray()[2].toDouble() - 1.075) < 1e-9 and std::abs(big["bearing_rad"].toDouble() - 0.5) < 1e-9);
    }
    CHECK(not QRegularExpression("\":-?[0-9]+,[0-9]").match(QString::fromStdString(j)).hasMatch());   // no comma decimal separator
    CHECK(o["image"].toObject()["w"].toInt() == 4);

    QImage png(QString::fromStdString((dir / "imagined_zed.png").string()));
    CHECK(png.width() == 4 and png.height() == 2);
    const QRgb p0 = png.pixel(0, 0), p1 = png.pixel(1, 0), p2 = png.pixel(2, 0), p3 = png.pixel(3, 0);
    CHECK(qRed(p0) == 255 and qGreen(p0) == 0 and qBlue(p0) == 0);
    CHECK(qRed(p1) == 170 and qGreen(p1) == 0);                            // 2/3 of 255
    CHECK(qRed(p2) == 0 and qGreen(p2) == 85);                             // 1/3 of 255
    CHECK(qRed(p3) == 0 and qGreen(p3) == 0);

    // no images: the earlier PNG must be removed, not left stale
    CHECK(std::filesystem::exists(dir / "imagined_zed.png"));
    CHECK(rc::chat::write_imagined(dir, set, {}, scans, mc, "a chair"));
    CHECK(not std::filesystem::exists(dir / "imagined_zed.png"));

    std::filesystem::remove_all(dir); std::filesystem::remove_all(dir2);
    std::puts(fails == 0 ? "all imagined_io checks passed" : "imagined_io checks FAILED");
    return fails ? 1 : 0;
}
