/*
 * view_crop_test.cpp — crop geometry, background blanking, colour histogram, frame history. OpenCV only.
 */
#include "frame_history.h"
#include "view_crop.h"

#include <cstdio>

using namespace rc::atlas;
static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}
static std::vector<Eigen::Vector2f> rect_px(int x0, int y0, int x1, int y1)
{
    std::vector<Eigen::Vector2f> v;
    for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) v.emplace_back(float(x), float(y));
    return v;
}

int main()
{
    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255));          // red background
    cv::rectangle(img, {600, 300}, {700, 400}, cv::Scalar(255, 0, 0), cv::FILLED);   // blue object

    std::printf("crop\n");
    auto c = make_crop(img, rect_px(600, 300, 700, 400));
    check(c.has_value(), "a normal mask yields a crop");
    check(c->rgb224.size() == cv::Size(224, 224) and c->rgb224.type() == CV_8UC3, "crop is 224x224 RGB");
    check(c->mask_px == 100 * 100, "mask pixel count");
    const cv::Vec3b corner = c->rgb224.at<cv::Vec3b>(2, 2), centre = c->rgb224.at<cv::Vec3b>(112, 112);
    check(corner == cv::Vec3b(124, 116, 104), "background blanked to ImageNet mean (RGB order)");
    check(centre[2] > 200 and centre[0] < 50, "object pixels kept, converted to RGB (blue in channel 2)");
    check(c->trunc_frac == 0.f, "interior mask is not truncated");

    auto e = make_crop(img, rect_px(1200, 300, 1280, 400));
    check(e.has_value() and e->rgb224.size() == cv::Size(224, 224), "mask at the right border: clipped crop");
    check(e->trunc_frac > 0.f, "border mask reports truncation");
    check(not make_crop(img, {}).has_value(), "empty mask: no crop");
    auto one = make_crop(img, {Eigen::Vector2f(10.f, 10.f)});
    check(one.has_value() and one->mask_px == 1, "1-pixel mask: crop, no throw");
    check(not make_crop(img, {Eigen::Vector2f(-5.f, 3.f), Eigen::Vector2f(5000.f, 3.f)}).has_value(),
          "all mask pixels outside the image: no crop");

    std::printf("colour histogram\n");
    const Eigen::VectorXf h = colour_hist(c->bgr_box, c->mask_box);
    check(h.size() == 64, "64 bins");
    check(std::fabs(h.norm() - 1.f) < 1e-5f, "unit length");
    int argmax; h.maxCoeff(&argmax);
    check(argmax == (120 * 16 / 180) * 4 + 255 * 4 / 256, "dominant bin is blue (hue 120, saturation 255)");
    check(colour_hist(c->bgr_box, cv::Mat::zeros(c->bgr_box.size(), CV_8U)).norm() == 0.f, "empty mask: zero histogram");

    std::printf("frame history\n");
    FrameHistory fh(3);
    for (std::uint64_t s = 1; s <= 4; ++s) fh.push(s * 100, img);
    check(fh.size() == 3, "capacity respected");
    check(fh.at(100) == nullptr, "oldest frame evicted");
    check(fh.at(400) != nullptr, "newest frame present");
    check(fh.at(250) == nullptr, "a stamp never pushed: nothing");
    cv::Mat src = img.clone(); fh.push(500, src); src.setTo(cv::Scalar(0, 0, 0));
    check(fh.at(500)->at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 0, 255), "push deep-copies (caller may reuse its buffer)");

    std::printf("\nview_crop_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
