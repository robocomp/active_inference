// encoder_parity — the C++ encoder must reproduce torch on the fixture crops (cosine >= 0.999).
// usage: encoder_parity <model.onnx> <fixture_dir> [--no-opt]
#include "../view_encoder.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.onnx fixture_dir [--no-opt]\n", argv[0]); return 2; }
    rc::atlas::EncoderConfig cfg; cfg.model_path = argv[1];
    cfg.graph_opt_all = not (argc > 3 and std::string(argv[3]) == "--no-opt");
    rc::atlas::ViewEncoder enc(cfg);
    if (not enc.ok()) return 2;
    const std::string dir = argv[2];
    std::ifstream f(dir + "/expected.f32", std::ios::binary | std::ios::ate);
    if (not f) { std::printf("cannot open %s/expected.f32\n", dir.c_str()); return 2; }
    const auto bytes = std::size_t(f.tellg());
    f.seekg(0);
    std::vector<float> ref(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(bytes));
    const std::size_t n = ref.size() / 384;
    float worst = 1.f;
    for (std::size_t i = 0; i < n; ++i)
    {
        char name[64]; std::snprintf(name, sizeof name, "/crop_%02zu.png", i);
        cv::Mat bgr = cv::imread(dir + name), rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        const auto v = enc.encode(rgb);
        if (not v) { std::printf("crop %zu: encode failed\n", i); return 1; }
        const Eigen::Map<const Eigen::VectorXf> r(ref.data() + i * 384, 384);
        const float cos = v->dot(r);
        worst = std::min(worst, cos);
        std::printf("crop %zu: cosine %.6f\n", i, cos);
    }
    std::printf("encoder_parity (%s): worst cosine %.6f -> %s\n", cfg.graph_opt_all ? "ENABLE_ALL" : "DISABLE_ALL",
                worst, worst >= 0.999f ? "PASS" : "FAIL");
    return worst >= 0.999f ? 0 : 1;
}
