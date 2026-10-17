// crop_golden — (i) C++ dense crop == Python mirror crop; (ii) stride-2 decimated crop ~ dense crop.
// usage: crop_golden <fixture_dir> <frames_dir> <model.onnx>
#include "../view_crop.h"
#include "../view_encoder.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

using namespace rc::atlas;

static std::vector<Eigen::Vector2f> pixels(const cv::Mat& m, int s, int ax, int ay)
{
    std::vector<Eigen::Vector2f> v;
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x)
            if (m.at<std::uint8_t>(y, x) and (x - ax) % s == 0 and (y - ay) % s == 0 and x >= ax and y >= ay)
                v.emplace_back(float(x), float(y));
    return v;
}
static float iou(const cv::Mat& a, const cv::Mat& b)
{
    if (a.size() != b.size()) return 0.f;
    const int i = cv::countNonZero(cv::Mat(a & b)), u = cv::countNonZero(cv::Mat(a | b));
    return u ? float(i) / float(u) : 1.f;
}
// The blanked 224² crop encodes the mask: background pixels equal the ImageNet-mean fill exactly.
static cv::Mat crop_mask(const cv::Mat& rgb224)
{
    cv::Mat m;
    cv::inRange(rgb224, cv::Scalar(124, 116, 104), cv::Scalar(124, 116, 104), m);
    return 255 - m;
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: %s fixture_dir frames_dir model.onnx\n", argv[0]); return 2; }
    const std::string fx = argv[1], fr = argv[2];
    EncoderConfig cfg; cfg.model_path = argv[3];
    ViewEncoder enc(cfg);
    if (not enc.ok()) return 2;
    std::ifstream list(fx + "/masks.txt");
    std::ifstream col(fx + "/py_colour.f32", std::ios::binary | std::ios::ate);
    const auto bytes = std::size_t(col.tellg()); col.seekg(0);
    std::vector<float> pyc(bytes / sizeof(float));
    col.read(reinterpret_cast<char*>(pyc.data()), std::streamsize(bytes));
    int k; std::string frame; bool ok = true;
    float worst_px = 0, worst_pyc = 1, worst_iou = 1, worst_dino = 1, worst_col = 1;
    while (list >> k >> frame)
    {
        char name[64];
        const cv::Mat img = cv::imread(fr + "/" + frame);
        std::snprintf(name, sizeof name, "/mask_%02d.png", k);
        const cv::Mat m = cv::imread(fx + name, cv::IMREAD_GRAYSCALE);
        std::snprintf(name, sizeof name, "/py_crop_%02d.png", k);
        cv::Mat py = cv::imread(fx + name), py_rgb;
        if (img.empty() or m.empty() or py.empty()) { std::printf("mask %d: missing input\n", k); return 1; }
        cv::cvtColor(py, py_rgb, cv::COLOR_BGR2RGB);
        const auto d = make_crop(img, pixels(m, 1, 0, 0));
        if (not d) { std::printf("mask %d: dense crop failed\n", k); return 1; }
        cv::Mat diff; cv::absdiff(d->rgb224, py_rgb, diff);
        double mx; cv::minMaxLoc(diff.reshape(1), nullptr, &mx);
        const Eigen::VectorXf hc = colour_hist(d->bgr_box, d->mask_box);
        const Eigen::Map<const Eigen::VectorXf> hp(pyc.data() + std::size_t(k) * 64, 64);
        worst_px = std::max(worst_px, float(mx)); worst_pyc = std::min(worst_pyc, hc.dot(hp));
        const auto ed = enc.encode(d->rgb224);
        cv::Rect bb = cv::boundingRect(m);
        int st[2] = {0, 0};
        float m_iou = 1, m_dino = 1, m_col = 1;
        for (int off = 0; off <= 1; ++off)
        {
            const auto px = pixels(m, 2, bb.x + off, bb.y + off);
            st[off] = infer_stride(px);
            const auto s = make_crop(img, px);
            if (not s or not ed) { std::printf("mask %d: decimated crop failed\n", k); return 1; }
            const auto es = enc.encode(s->rgb224);
            const float a = iou(crop_mask(s->rgb224), crop_mask(d->rgb224)), b = es->dot(*ed),
                        c = colour_hist(s->bgr_box, s->mask_box).dot(hc);
            m_iou = std::min(m_iou, a); m_dino = std::min(m_dino, b); m_col = std::min(m_col, c);
            worst_iou = std::min(worst_iou, a); worst_dino = std::min(worst_dino, b); worst_col = std::min(worst_col, c);
        }
        std::printf("mask %02d: px diff %.0f, colour(py) %.6f | inferred stride off0=%d off1=%d%s | iou %.4f dino %.4f col %.4f\n",
                    k, mx, hc.dot(hp), st[0], st[1], (st[0] != 2 or st[1] != 2) ? "  <-- NOT 2" : "", m_iou, m_dino, m_col);
    }
    const bool i_ok = worst_px <= 1.f and worst_pyc >= 0.9999f;
    const bool ii_ok = worst_iou >= 0.98f and worst_dino >= 0.99f and worst_col >= 0.99f;
    std::printf("(i)  C++ vs Python dense: worst pixel diff %.0f, worst colour cosine %.6f -> %s\n",
                worst_px, worst_pyc, i_ok ? "PASS" : "FAIL");
    std::printf("(ii) stride 2 vs dense: worst mask IoU %.4f, worst DINO cosine %.4f, worst colour cosine %.4f -> %s\n",
                worst_iou, worst_dino, worst_col, ii_ok ? "PASS" : "FAIL");
    ok = i_ok and ii_ok;
    return ok ? 0 : 1;
}
