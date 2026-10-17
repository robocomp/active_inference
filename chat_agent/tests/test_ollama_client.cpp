// Live check of rc::chat::OllamaClient against a running ollama (needs the model pulled). Not part of run_tests.sh
// (it needs the server); run with: tests/run_ollama_test.sh [model]
#include "../src/ollama_client.h"
#include <QCoreApplication>
#include <QJsonDocument>
#include <QTimer>
#include <cstdio>
#include <functional>
#include <print>

using namespace rc::chat;

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    OllamaConfig cfg;
    if (argc > 1) cfg.model = argv[1];
    cfg.first_token_timeout_ms = 90000;
    auto* cli = new OllamaClient(cfg);
    int failures = 0;
    std::vector<std::function<void(std::function<void()>)>> steps;

    const QJsonArray tools{QJsonObject{{"type", "function"}, {"function", QJsonObject{{"name", "cross_door"},
        {"description", "Take the robot through a door of the current room."},
        {"parameters", QJsonObject{{"type", "object"}, {"properties", QJsonObject{{"door", QJsonObject{{"type", "string"},
            {"enum", QJsonArray{"door_19", "door_7", "nearest"}}}}}}, {"required", QJsonArray{"door"}}}}}}}};
    const QString sys = "You are a robot. Answer only from STATE, briefly.\nSTATE:\ncurrent room: room_1\ndoors: door_19 (2.3 m, ahead-left), door_7 (5.1 m, behind-right)\n";

    steps.push_back([&](std::function<void()> next) {
        cli->ping([&, next](const OllamaClient::PingResult& r) {
            std::println("[ping] up={} model={} msg='{}'", r.server_up, r.model_present, r.message.toStdString());
            if (not r.server_up or not r.model_present) ++failures;
            next(); }); });
    steps.push_back([&](std::function<void()> next) {   // streaming text
        auto ntok = std::make_shared<int>(0);
        QJsonArray msgs{QJsonObject{{"role", "system"}, {"content", sys}}, QJsonObject{{"role", "user"}, {"content", "How many doors are there? Answer in one sentence."}}};
        cli->chat(msgs, {}, {.on_token = [ntok](const QString& t) { ++*ntok; std::printf("%s", t.toUtf8().constData()); std::fflush(stdout); },
                             .on_tool_calls = {},
                             .on_done = [&, next, ntok](const ChatStats& s) { std::println("\n[text] tokens_cb={} eval={} first={:.0f} ms wall={:.0f} ms", *ntok, s.eval_count, s.first_token_ms, s.wall_ms); if (*ntok < 2) ++failures; next(); },
                             .on_error = [&, next](const QString& e) { std::println("[text] ERROR {}", e.toStdString()); ++failures; next(); }}); });
    steps.push_back([&](std::function<void()> next) {   // tool call
        QJsonArray msgs{QJsonObject{{"role", "system"}, {"content", sys}}, QJsonObject{{"role", "user"}, {"content", "Cruza la puerta de detrás a la derecha"}}};
        auto calls = std::make_shared<QJsonArray>();
        cli->chat(msgs, tools, {.on_token = [](const QString&) {},
                                .on_tool_calls = [calls](const QJsonArray& c) { *calls = c; },
                                .on_done = [&, next, calls](const ChatStats& s) {
                                    std::println("[tool] calls={} wall={:.0f} ms", QJsonDocument(*calls).toJson(QJsonDocument::Compact).toStdString(), s.wall_ms);
                                    if (calls->isEmpty()) ++failures; next(); },
                                .on_error = [&, next](const QString& e) { std::println("[tool] ERROR {}", e.toStdString()); ++failures; next(); }}); });
    steps.push_back([&](std::function<void()> next) {   // cancel mid-stream; no on_done may follow
        QJsonArray msgs{QJsonObject{{"role", "user"}, {"content", "Write a 300-word story about a robot."}}};
        auto got_err = std::make_shared<bool>(false);
        cli->chat(msgs, {}, {.on_token = [](const QString&) {},
                             .on_tool_calls = {},
                             .on_done = [&](const ChatStats&) { std::println("[cancel] on_done fired after cancel — BAD"); ++failures; },
                             .on_error = [got_err](const QString& e) { *got_err = true; std::println("[cancel] on_error('{}')", e.toStdString()); }});
        if (cli->chat(QJsonArray{}, {}, {})) { std::println("[cancel] second request accepted while busy — BAD"); ++failures; }
        QTimer::singleShot(400, [&, next, got_err]() { cli->cancel(); QTimer::singleShot(1500, [&, next, got_err]() {
            if (not *got_err or cli->busy()) ++failures; next(); }); }); });
    steps.push_back([&](std::function<void()> next) {   // transport error
        OllamaConfig bad = cfg; bad.url = "http://localhost:1";
        auto* b = new OllamaClient(bad);
        b->chat(QJsonArray{}, {}, {.on_token = {}, .on_tool_calls = {}, .on_done = [&, next](const ChatStats&) { ++failures; next(); },
                                   .on_error = [&, next](const QString& e) { std::println("[badurl] on_error('{}')", e.toStdString()); next(); }}); });
    steps.push_back([&](std::function<void()> next) {   // unknown model → {"error":...}
        OllamaConfig bad = cfg; bad.model = "no_such_model:1b";
        auto* b = new OllamaClient(bad);
        b->chat(QJsonArray{QJsonObject{{"role", "user"}, {"content", "hi"}}}, {}, {.on_token = {}, .on_tool_calls = {}, .on_done = [&, next](const ChatStats&) { ++failures; next(); },
                                   .on_error = [&, next](const QString& e) { std::println("[badmodel] on_error('{}')", e.toStdString()); next(); }}); });

    std::size_t i = 0;
    std::function<void()> run_next = [&]() {
        if (i == steps.size()) { std::println("test_ollama_client: {} failure(s)", failures); app.exit(failures == 0 ? 0 : 1); return; }
        auto& s = steps[i++];
        QTimer::singleShot(0, [&s, &run_next]() { s(run_next); });
    };
    run_next();
    return app.exec();
}
