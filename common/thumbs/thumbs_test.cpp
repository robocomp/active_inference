// g++ -std=c++23 -O2 $(pkg-config --cflags opencv4) common/thumbs/thumbs_test.cpp -o t $(pkg-config --libs opencv4) && ./t
#include "thumb_crop.h"
#include "thumb_select.h"
#include "thumb_wire.h"

#include <cmath>
#include <cstdio>
#include <numbers>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
using namespace rc::thumbs;

static void wire_round_trip()
{
    ThumbFrame f;
    f.frame = 4711;
    f.thumbs.push_back({.key = 21990232555545ULL, .bin = 3, .score = 12.5f, .jpeg = {1, 2, 3}});
    f.thumbs.push_back({.key = 2, .bin = -1, .score = 0.f, .jpeg = {}});
    const auto got = decode(encode(f));
    CHECK(got.has_value());
    if (not got) return;
    CHECK(got->frame == 4711);
    CHECK(got->thumbs.size() == 2);
    CHECK(got->thumbs[0].key == 21990232555545ULL and got->thumbs[0].bin == 3 and got->thumbs[0].score == 12.5f);
    CHECK((got->thumbs[0].jpeg == std::vector<std::uint8_t>{1, 2, 3}));
    CHECK(got->thumbs[1].jpeg.empty());
}

static void wire_rejects_damage()
{
    ThumbFrame f;
    f.thumbs.push_back({.key = 1, .jpeg = {9, 9, 9, 9}});
    auto b = encode(f);
    CHECK(not decode({}).has_value());
    auto cut = b; cut.pop_back();
    CHECK(not decode(cut).has_value());
    auto extra = b; extra.push_back(0);
    CHECK(not decode(extra).has_value());
    auto bad = b; bad[0] ^= 0xFF;
    CHECK(not decode(bad).has_value());
}

static void crop_is_small_jpeg_of_the_box()
{
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    img(cv::Rect(100, 100, 200, 100)).setTo(cv::Scalar(0, 255, 0));
    const auto jpeg = crop_thumbnail(img, 100, 100, 299, 199, 96, 80);
    CHECK(not jpeg.empty());
    const cv::Mat back = cv::imdecode(jpeg, cv::IMREAD_COLOR);
    CHECK(not back.empty() and std::max(back.cols, back.rows) == 96);
    CHECK(not back.empty() and back.cols > back.rows);
    if (not back.empty()) CHECK(back.at<cv::Vec3b>(back.rows / 2, back.cols / 2)[1] > 200);
    CHECK(jpeg.size() < 8000);
}

static void crop_survives_the_image_border_and_nonsense()
{
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(10, 20, 30));
    CHECK(not crop_thumbnail(img, -30, -30, 200, 200, 64, 80).empty());
    CHECK(crop_thumbnail(img, 50, 50, 40, 40, 64, 80).empty());
    CHECK(crop_thumbnail(img, 300, 300, 400, 400, 64, 80).empty());
    CHECK(crop_thumbnail(cv::Mat(), 0, 0, 10, 10, 64, 80).empty());
}

static void the_viewing_angle_picks_the_bin()
{
    constexpr int n = 8;
    const float pi = std::numbers::pi_v<float>;
    CHECK(view_bin(1.f, 0.f, 0.f, 0.f, 0.f, n) == view_bin(5.f, 0.f, 0.f, 0.f, 0.f, n));   // range does not matter
    CHECK(view_bin(1.f, 0.f, 0.f, 0.f, 0.f, n) != view_bin(-1.f, 0.f, 0.f, 0.f, 0.f, n));  // the opposite side does
    CHECK(view_bin(1.f, 0.f, 0.f, 0.f, pi / 2, n) == view_bin(0.f, -1.f, 0.f, 0.f, 0.f, n));   // relative to the chair's yaw
    CHECK(view_bin(1.f, 1.f, 1.f, 1.f, 0.f, n) >= 0);
    for (float a = -10.f; a < 10.f; a += 0.37f)
    {
        const int b = view_bin(std::cos(a), std::sin(a), 0.f, 0.f, 1.3f, n);
        CHECK(b >= 0 and b < n);
    }
}

static void the_bins_are_the_atlas_bins()
{
    constexpr int n = kViewBins;
    const float pi = std::numbers::pi_v<float>;
    const float d = 0.05f;
    CHECK(n == 12);
    CHECK(view_bin(std::cos(d), std::sin(d), 0.f, 0.f, 0.f, n) == 6);      // just left of the front
    CHECK(view_bin(std::cos(d), -std::sin(d), 0.f, 0.f, 0.f, n) == 5);     // just right of it
    CHECK(view_bin(-std::cos(d), std::sin(d), 0.f, 0.f, 0.f, n) == 11);    // just short of behind, on the left
    CHECK(view_bin(-std::cos(d), -std::sin(d), 0.f, 0.f, 0.f, n) == 0);    // and on the right
    for (int b = 0; b < n; ++b)
    {
        const float a = bin_centre_rad(b, n);
        CHECK(std::abs(a - (-pi + (static_cast<float>(b) + .5f) * 2.f * pi / static_cast<float>(n))) < 1e-5f);
        CHECK(view_bin(std::cos(a), std::sin(a), 0.f, 0.f, 0.f, n) == b);
    }
}

static void a_bin_keeps_its_best_view()
{
    ThumbSet s;
    CHECK(s.offer(2, 10.f, {1}));
    CHECK(not s.offer(2, 5.f, {2}));
    CHECK(s.offer(2, 20.f, {3}));
    CHECK(s.offer(5, 1.f, {4}));
    CHECK(s.bins().size() == 2);
    CHECK((s.bins().at(2).jpeg == std::vector<std::uint8_t>{3}));
    CHECK(not s.offer(1, 9.f, {}));   // no image, nothing to keep
    CHECK(s.bins().size() == 2);
}

int main()
{
    wire_round_trip();
    wire_rejects_damage();
    crop_is_small_jpeg_of_the_box();
    crop_survives_the_image_border_and_nonsense();
    the_viewing_angle_picks_the_bin();
    the_bins_are_the_atlas_bins();
    a_bin_keeps_its_best_view();
    std::printf("%s: %d failure(s)\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
