// test_mask_dump.cpp — the [ZedDump] mask dump (mask_dump_csv.h + ZedFrameDump::maybe_save_masks).
//   1. CSV round-trip of a synthetic mask set, parsed under the user's locale (es_ES here: comma
//      decimal separator) — catches any strtof/atof creeping into the reader or a locale-formatted writer.
//   2. Gate off (enabled=false) writes NOTHING, not even the directory.
//   3. enabled=true, dump_masks=false writes the JPEG + poses.csv but no mask file.
//   4. Gate on: masks are written only for the stamp the ZedDump KEPT, once.
// Build+run: retina/tests/run_mask_dump_test.sh   (or cmake -DRETINA_BUILD_TESTS=ON … && ctest)
#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <string>

#include "../src/mask_dump_csv.h"
#include "../src/zed_frame_dump.h"

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (not (c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

namespace fs = std::filesystem;
using rc::diag::MaskDumpEntry;
using rc::diag::MaskDumpRecord;

namespace
{
bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

fs::path fresh_dir(const char* name)
{
    const auto d = fs::temp_directory_path() / std::format("retina_mask_dump_test_{}", name);
    fs::remove_all(d);
    return d;
}

std::size_t count_files(const fs::path& d, std::string_view needle = "")
{
    if (not fs::exists(d))
        return 0;
    std::size_t n = 0;
    for (const auto& e : fs::directory_iterator(d))
        if (needle.empty() or e.path().filename().string().find(needle) != std::string::npos)
            ++n;
    return n;
}

MaskDumpRecord synthetic(std::uint64_t stamp)
{
    MaskDumpRecord r{.stamp_ms = stamp, .frame_id = 4242, .img_w = 640, .img_h = 480, .pixel_stride = 2};
    r.masks.push_back(MaskDumpEntry{.label = "chair", .label_id = 56, .confidence = 0.8731f,
                                    .segmenter = rc::MaskSegmenter::yolo,
                                    .pixels_xy = {10, 20, 12, 20, 638, 478},
                                    .points_cam = {-0.25f, 2.5f, 0.125f, 0.0001f, 9.875f, -1.5f}});
    r.masks.push_back(MaskDumpEntry{.label = "chest of drawers", .label_id = 1044, .confidence = 0.5f,
                                    .segmenter = rc::MaskSegmenter::semantic,
                                    .pixels_xy = {0, 0}, .points_cam = {1.0f, 3.0f, -0.0625f}});
    // A label carrying the CSV/DSR separators must not break the row; no pixels, no points.
    r.masks.push_back(MaskDumpEntry{.label = "a,b|c", .label_id = 1003, .confidence = 0.25f,
                                    .segmenter = rc::MaskSegmenter::semantic});
    return r;
}

Mat::RTMat cam_pose(double x, double y)
{
    Mat::RTMat T = Mat::RTMat::Identity();
    T.translation() = Eigen::Vector3d(x, y, 0.95);   // never Identity: the dump rejects that as "no pose"
    return T;
}
} // namespace

int main()
{
    // Behave like the agent: Qt calls setlocale(LC_ALL, "") at startup. On these machines that is es_ES,
    // whose decimal separator is a COMMA. If the environment has no such locale, try it explicitly.
    std::setlocale(LC_ALL, "");
    if (std::localeconv()->decimal_point[0] != ',')
        std::setlocale(LC_ALL, "es_ES.UTF-8");
    std::printf("LC_NUMERIC decimal point: '%s'%s\n", std::localeconv()->decimal_point,
                std::localeconv()->decimal_point[0] == ',' ? "" : "  (no comma locale available — weaker test)");

    // ── 1. round-trip ──────────────────────────────────────────────────────────────────────────
    {
        const auto d = fresh_dir("roundtrip");
        fs::create_directories(d);
        const auto in = synthetic(1700000000123ULL);
        CHECK(rc::diag::write_mask_dump(d.string(), in));
        CHECK(fs::exists(rc::diag::mask_dump_path(d.string(), in.stamp_ms)));
        CHECK(fs::exists(rc::diag::mask_dump_path(d.string(), in.stamp_ms, ".pix")));
        CHECK(fs::exists(rc::diag::mask_dump_path(d.string(), in.stamp_ms, ".pts")));

        const auto out = rc::diag::read_mask_dump(d.string(), in.stamp_ms);
        CHECK(out.has_value());
        if (out)
        {
            CHECK(out->stamp_ms == in.stamp_ms);
            CHECK(out->frame_id == 4242);
            CHECK(out->img_w == 640 and out->img_h == 480 and out->pixel_stride == 2);
            CHECK(out->masks.size() == 3);
            if (out->masks.size() == 3)
            {
                for (std::size_t m = 0; m < 3; ++m)
                {
                    const auto& a = in.masks[m];
                    const auto& b = out->masks[m];
                    CHECK(a.label_id == b.label_id);
                    CHECK(near(a.confidence, b.confidence, 1e-4f));
                    CHECK(a.segmenter == b.segmenter);
                    CHECK(a.pixels_xy == b.pixels_xy);   // integers, exact
                    CHECK(a.points_cam.size() == b.points_cam.size());
                    for (std::size_t i = 0; i < std::min(a.points_cam.size(), b.points_cam.size()); ++i)
                        CHECK(near(a.points_cam[i], b.points_cam[i], 1e-4f));   // written at 0.1 mm
                }
                CHECK(out->masks[0].label == "chair");
                CHECK(out->masks[1].label == "chest of drawers");
                CHECK(out->masks[2].label == "a b c");
                // The truncation bug this guards against: "0.8731" read as 0, "-0.25" read as -0.
                CHECK(out->masks[0].confidence > 0.87f);
                CHECK(out->masks[0].points_cam[0] < -0.2f);
            }
        }
        // The file itself must carry decimal POINTS, never commas inside numbers.
        {
            std::FILE* f = std::fopen(rc::diag::mask_dump_path(d.string(), in.stamp_ms, ".pts").c_str(), "r");
            CHECK(f != nullptr);
            if (f)
            {
                char line[256];
                CHECK(std::fgets(line, sizeof line, f) != nullptr);   // header
                CHECK(std::fgets(line, sizeof line, f) != nullptr);
                CHECK(std::string(line) == "0,-0.2500,2.5000,0.1250\n");
                std::fclose(f);
            }
        }
        // A missing index is reported, not invented.
        CHECK(not rc::diag::read_mask_dump(d.string(), 1).has_value());
        fs::remove_all(d);
    }

    const cv::Mat bgr(48, 64, CV_8UC3, cv::Scalar(30, 60, 90));
    const rc::diag::ZedDumpGroundTruth gt{};

    // ── 2. gate OFF: nothing at all ────────────────────────────────────────────────────────────
    {
        const auto d = fresh_dir("off");
        rc::diag::ZedFrameDump dump;
        dump.configure(false, d.string(), 92, 0.05f, 0.05f, /*dump_masks=*/true);
        CHECK(not dump.enabled());
        CHECK(not dump.wants_masks());
        CHECK(not dump.maybe_save(1000, bgr, cam_pose(1, 2), std::nullopt, gt));
        CHECK(not dump.maybe_save_masks(synthetic(1000)));
        CHECK(not fs::exists(d));
        // Default-constructed (never configured) too.
        rc::diag::ZedFrameDump never;
        CHECK(not never.wants_masks());
        CHECK(not never.maybe_save_masks(synthetic(1000)));
    }

    // ── 3. enabled, dump_masks=false: images yes, masks no ────────────────────────────────────
    {
        const auto d = fresh_dir("nomasks");
        rc::diag::ZedFrameDump dump;
        dump.configure(true, d.string(), 92, 0.05f, 0.05f, /*dump_masks=*/false);
        CHECK(dump.enabled() and not dump.wants_masks());
        CHECK(dump.maybe_save(2000, bgr, cam_pose(1, 2), std::nullopt, gt));
        CHECK(not dump.maybe_save_masks(synthetic(2000)));
        CHECK(fs::exists(d / "2000.jpg"));
        CHECK(fs::exists(d / "poses.csv"));
        CHECK(count_files(d, ".masks") == 0);
        fs::remove_all(d);
    }

    // ── 4. gate ON: same keep decision, same stamp, once ─────────────────────────────────────
    {
        const auto d = fresh_dir("on");
        rc::diag::ZedFrameDump dump;
        dump.configure(true, d.string(), 92, 0.05f, 0.05f, /*dump_masks=*/true);
        CHECK(dump.wants_masks());
        CHECK(not dump.maybe_save_masks(synthetic(3000)));               // nothing kept yet
        CHECK(dump.maybe_save(3000, bgr, cam_pose(1, 2), std::nullopt, gt));
        CHECK(not dump.maybe_save_masks(synthetic(2999)));               // stale record: other stamp
        CHECK(dump.maybe_save_masks(synthetic(3000)));                   // the kept frame
        CHECK(not dump.maybe_save_masks(synthetic(3000)));               // only once
        CHECK(fs::exists(rc::diag::mask_dump_path(d.string(), 3000)));
        // Parked camera ⇒ frame NOT kept ⇒ its masks are not written either.
        CHECK(not dump.maybe_save(3100, bgr, cam_pose(1, 2), std::nullopt, gt));
        CHECK(not dump.maybe_save_masks(synthetic(3100)));
        CHECK(not fs::exists(rc::diag::mask_dump_path(d.string(), 3100)));
        // Moved ⇒ kept ⇒ written; an empty publish still leaves a header-only index.
        CHECK(dump.maybe_save(3200, bgr, cam_pose(1.5, 2), std::nullopt, gt));
        CHECK(dump.maybe_save_masks(MaskDumpRecord{.stamp_ms = 3200, .frame_id = 7}));
        const auto empty = rc::diag::read_mask_dump(d.string(), 3200);
        CHECK(empty.has_value() and empty->masks.empty());
        CHECK(count_files(d, ".jpg") == 2);
        CHECK(count_files(d, ".masks.csv") == 2);
        fs::remove_all(d);
    }

    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
