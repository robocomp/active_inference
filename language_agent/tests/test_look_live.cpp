// Live check of the look() path (needs ollama + a vision model): a SAVED camera frame goes through the agent's own
// code — prepare_image (image_prep.h) → make_look_tool_message (look_messages.h) → OllamaClient (streaming) — exactly
// as ChatSession does after a look() call, minus the media plane. Not part of run_tests.sh.
//   tests/run_look_test.sh [model]
// Steps: (1) "what colour is the door ahead?" with the look tool offered ⇒ the model must call look;
//        (2) the tool result carrying the frame, follow-up without tools ⇒ the answer must say "green";
//        (3) the same in Spanish ⇒ the answer must stay in Spanish ("verde");
//        (4) strip_images leaves no image in the history.
#include "../src/image_prep.h"
#include "../src/look_messages.h"
#include "../src/ollama_client.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QTimer>
#include <clocale>
#include <deque>
#include <functional>
#include <memory>
#include <print>

using namespace rc::chat;

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    std::setlocale(LC_ALL, "");
    OllamaConfig cfg;
    cfg.model = argc > 1 ? argv[1] : "qwen3-vl:8b-instruct";
    cfg.first_token_timeout_ms = 120000;
    cfg.num_predict = 400;
    const QString frame_path = argc > 2 ? argv[2]
        : "../../room_concept/estimation_scripts/crossing/try7/snapshots/zed_1791753485762.png";   // a green, shut door
    auto* cli = new OllamaClient(cfg);
    int failures = 0;

    QFile pf("../etc/system_prompt.md");
    const QString robot = pf.open(QIODevice::ReadOnly) ? QString::fromUtf8(pf.readAll()) : QString("You are a robot.");
    const QString sys = robot + "\n- look(camera) shows you the current picture of a camera. When the person asks what you "
        "see or about colours, CALL look() first and answer from the picture. Lines and labels drawn over a picture are "
        "debug overlays.\nSTATE:\ncurrent room: room_1\ndoors of the current room (1):\n  - door_0: 1.8 m away, bearing "
        "+2 deg (ahead), P(open)=0.10\n";
    const QJsonArray tools{QJsonObject{{"type", "function"}, {"function", QJsonObject{{"name", "look"},
        {"description", "Look through a camera now: the current picture is shown to you in the next message. Read-only."},
        {"parameters", QJsonObject{{"type", "object"}, {"properties", QJsonObject{{"camera", QJsonObject{{"type", "string"},
            {"enum", QJsonArray{"front", "360"}}}}}}}}}}}};

    const cv::Mat bgr = cv::imread(frame_path.toStdString(), cv::IMREAD_COLOR);
    const auto img = prepare_image(bgr, 1280, 85);
    if (not img.has_value()) { std::println("cannot read/encode {}", frame_path.toStdString()); return 2; }
    const QByteArray jpeg(reinterpret_cast<const char*>(img->jpeg.data()), static_cast<qsizetype>(img->jpeg.size()));
    std::println("frame {} → {}x{}, {} KiB JPEG", frame_path.toStdString(), img->width, img->height, jpeg.size() / 1024);

    // One question: ask with tools, expect look(), attach the frame to the tool result, follow up, grade the answer.
    auto ask = [&](const QString& question, const QString& must_contain, std::function<void()> next)
    {
        auto hist = std::make_shared<std::deque<QJsonObject>>(std::deque<QJsonObject>{
            QJsonObject{{"role", "user"}, {"content", question}}});
        auto msgs_of = [sys, hist]()
        {
            QJsonArray a{QJsonObject{{"role", "system"}, {"content", sys}}};
            for (const auto& m : *hist) a.append(m);
            return a;
        };
        auto calls = std::make_shared<QJsonArray>();
        cli->chat(msgs_of(), tools, {.on_token = [](const QString&) {},
            .on_tool_calls = [calls](const QJsonArray& c) { *calls = c; },
            .on_done = [&, question, must_contain, next, hist, calls, msgs_of](const ChatStats& s1)
            {
                const QString name = calls->isEmpty() ? QString() : calls->at(0).toObject().value("function").toObject().value("name").toString();
                std::println("[{}] tool call: {} ({:.0f} ms)", question.toStdString(),
                             QJsonDocument(*calls).toJson(QJsonDocument::Compact).toStdString(), s1.wall_ms);
                if (name != "look") { ++failures; next(); return; }
                hist->push_back(QJsonObject{{"role", "assistant"}, {"content", ""}, {"tool_calls", *calls}});
                hist->push_back(make_look_tool_message(look_tool_result("front camera", 120, img->width, img->height), jpeg));
                auto answer = std::make_shared<QString>();
                cli->chat(msgs_of(), {}, {.on_token = [answer](const QString& t) { *answer += t; },
                    .on_tool_calls = {},
                    .on_done = [&, must_contain, next, hist, answer](const ChatStats& s2)
                    {
                        const bool ok = answer->toLower().contains(must_contain);
                        std::println("    answer: '{}' — prompt {} tok, out {} tok, {:.0f} ms  {}", answer->trimmed().toStdString(),
                                     s2.prompt_eval_count, s2.eval_count, s2.wall_ms, ok ? "PASS" : "FAIL");
                        if (not ok) ++failures;
                        const int n = strip_images(*hist);
                        const bool clean = n == 1 and std::ranges::none_of(*hist, [](const QJsonObject& m) { return has_image(m); });
                        if (not clean) { std::println("    strip_images left an image — FAIL"); ++failures; }
                        next();
                    },
                    .on_error = [&, next](const QString& e) { std::println("    ERROR {}", e.toStdString()); ++failures; next(); }});
            },
            .on_error = [&, next](const QString& e) { std::println("ERROR {}", e.toStdString()); ++failures; next(); }});
    };

    std::vector<std::function<void(std::function<void()>)>> steps{
        [&](std::function<void()> n) { ask("what colour is the door ahead of you?", "green", n); },
        [&](std::function<void()> n) { ask("¿de qué color es la puerta que tienes delante?", "verde", n); },
        [&](std::function<void()> n) { ask("what do you see?", "door", n); },
    };
    std::size_t i = 0;
    std::function<void()> run_next = [&]()
    {
        if (i == steps.size()) { std::println("test_look_live: {} failure(s)", failures); app.exit(failures == 0 ? 0 : 1); return; }
        auto& s = steps[i++];
        QTimer::singleShot(0, [&s, &run_next]() { s(run_next); });
    };
    run_next();
    return app.exec();
}
