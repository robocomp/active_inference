// look() offline checks: frame shrink + JPEG (image_prep.h) and the request JSON / one-turn image lifetime
// (look_messages.h). Build+run: tests/run_tests.sh
#include "../src/image_prep.h"
#include "../src/look_messages.h"

#include <cassert>
#include <clocale>
#include <deque>
#include <print>

#include <QJsonDocument>

int main()
{
    using namespace rc::chat;
    std::setlocale(LC_ALL, "");
    int checks = 0;

    // ── fit_longest_side: never upscales, keeps the aspect, longest side = max_side ──
    assert(fit_longest_side(1280, 720, 768) == cv::Size(768, 432)); ++checks;
    assert(fit_longest_side(1920, 960, 1280) == cv::Size(1280, 640)); ++checks;
    assert(fit_longest_side(720, 1280, 768) == cv::Size(432, 768)); ++checks;        // portrait
    assert(fit_longest_side(640, 360, 768) == cv::Size(640, 360)); ++checks;         // smaller: unchanged
    assert(fit_longest_side(1280, 720, 0) == cv::Size(1280, 720)); ++checks;         // 0 = no scaling

    // ── prepare_image: a ZED-sized frame becomes a decodable JPEG of the expected size, colours kept ──
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(0, 200, 0));                       // green (BGR)
    cv::rectangle(frame, {0, 0, 640, 720}, cv::Scalar(0, 0, 220), cv::FILLED);       // left half red
    const auto img = prepare_image(frame, 768, 80);
    assert(img.has_value() and img->width == 768 and img->height == 432); ++checks;
    assert(img->jpeg.size() > 100 and img->jpeg[0] == 0xFF and img->jpeg[1] == 0xD8); ++checks;   // JPEG SOI
    const cv::Mat back = cv::imdecode(img->jpeg, cv::IMREAD_COLOR);
    assert(back.cols == 768 and back.rows == 432); ++checks;
    const cv::Vec3b left = back.at<cv::Vec3b>(216, 100), right = back.at<cv::Vec3b>(216, 700);
    assert(left[2] > 180 and left[1] < 60 and right[1] > 160 and right[2] < 60); ++checks;   // red left, green right
    const auto hi = prepare_image(frame, 768, 95), lo = prepare_image(frame, 768, 20);
    assert(hi.has_value() and lo.has_value() and lo->jpeg.size() <= hi->jpeg.size()); ++checks;
    assert(not prepare_image(cv::Mat{}, 768, 80).has_value()); ++checks;               // empty ⇒ nothing
    assert(not prepare_image(cv::Mat(10, 10, CV_32FC1), 768, 80).has_value()); ++checks;   // depth ⇒ refused
    const auto grey = prepare_image(cv::Mat(100, 50, CV_8UC1, cv::Scalar(128)), 768, 80);
    assert(grey.has_value() and grey->width == 50 and grey->height == 100); ++checks;   // small grey: no upscale

    // ── camera argument ──
    assert(look_camera_is_front("") == true and look_camera_is_front("front") == true); ++checks;
    assert(look_camera_is_front("360") == false and look_camera_is_front("panorama") == false); ++checks;
    assert(not look_camera_is_front("ceiling").has_value()); ++checks;

    // ── the look() tool message: role tool, the result text, ONE base64 image that decodes back to the JPEG ──
    const QByteArray jpeg(reinterpret_cast<const char*>(img->jpeg.data()), static_cast<qsizetype>(img->jpeg.size()));
    const QString tool = look_tool_result("front camera", 120, 768, 432);
    assert(tool.startsWith("OK:") and tool.contains("768x432") and tool.contains("120 ms")); ++checks;
    assert(look_tool_result("360-degree camera", -1, 1, 1).contains("unknown age")); ++checks;
    const QJsonObject m = make_look_tool_message(tool, jpeg);
    assert(m.value("role").toString() == "tool" and m.value("tool_name").toString() == "look"); ++checks;
    assert(m.value("content").toString() == tool); ++checks;
    const QJsonArray images = m.value("images").toArray();
    assert(images.size() == 1 and QByteArray::fromBase64(images[0].toString().toLatin1()) == jpeg); ++checks;

    // ── a request body as ollama receives it: the image survives a JSON round trip ──
    const QJsonObject call{{"function", QJsonObject{{"name", "look"}, {"arguments", QJsonObject{{"camera", "front"}}}}}};
    QJsonArray msgs{QJsonObject{{"role", "system"}, {"content", "sys"}},
                    QJsonObject{{"role", "user"}, {"content", "what do you see?"}},
                    QJsonObject{{"role", "assistant"}, {"content", ""}, {"tool_calls", QJsonArray{call}}}, m};
    const QJsonObject body{{"model", "qwen3-vl:8b-instruct"}, {"messages", msgs}, {"stream", true}};
    const QJsonObject parsed = QJsonDocument::fromJson(QJsonDocument(body).toJson(QJsonDocument::Compact)).object();
    assert(parsed.value("messages").toArray().last().toObject().value("images").toArray().size() == 1); ++checks;

    // ── logs never get the payload ──
    const QJsonArray red = redact_images(msgs);
    const QString red_text = QString::fromUtf8(QJsonDocument(red).toJson(QJsonDocument::Compact));
    assert(not red_text.contains(images[0].toString().left(40)) and red_text.contains("<base64,")); ++checks;
    assert(red.size() == msgs.size() and red[2].toObject() == msgs[2].toObject()); ++checks;   // the rest untouched

    // ── one-turn lifetime: strip_images drops every image and leaves a placeholder ──
    std::deque<QJsonObject> hist{msgs[1].toObject(), msgs[2].toObject(), m};
    assert(hist.back().value("role").toString() == "tool"); ++checks;
    assert(strip_images(hist) == 1); ++checks;
    assert(not has_image(hist.back()) and hist.back().value("content").toString().contains("no longer attached")); ++checks;
    assert(strip_images(hist) == 0); ++checks;                                      // idempotent

    std::println("test_look: {} checks passed", checks);
    return 0;
}
