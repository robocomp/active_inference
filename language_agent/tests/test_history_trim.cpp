// History trimming in blocks (history_trim.h): the prefix stays put between trims so ollama's prompt cache holds, and
// the token guard drops blocks until the request fits. Build+run: tests/run_tests.sh
#include "../src/history_trim.h"

#include <cassert>
#include <print>

using namespace rc::chat;

namespace
{
QJsonObject msg(const char* role, int i, int pad = 0)
{
    return QJsonObject{{"role", role}, {"content", QString("m%1").arg(i) + QString(pad, 'x')}};
}
QString first_content(const std::deque<QJsonObject>& h) { return h.front().value("content").toString(); }
}   // namespace

int main()
{
    int checks = 0;
    auto ok = [&](bool c, const char* what) { ++checks; if (not c) { std::println("FAIL: {}", what); std::exit(1); } };

    {   // 401 pushes with max 400 / trim_to 200 leave 200; the prefix is stable between trims
        std::deque<QJsonObject> h;
        QString prefix_at_400;
        int changes = 0;
        QString prev;
        for (int i = 0; i < 401; ++i)
        {
            h.push_back(msg(i % 2 ? "assistant" : "user", i));
            trim_history(h, 400, 200);
            if (h.size() > 0 and first_content(h) != prev) { ++changes; prev = first_content(h); }
        }
        ok(h.size() == 200, "401 pushes leave 200");
        ok(changes == 2, "front changed only at the start and at the one trim");
        ok(first_content(h) == "m201", "block kept the newest 200");
    }
    {   // no leading tool message
        std::deque<QJsonObject> h;
        for (int i = 0; i < 10; ++i) h.push_back(msg(i == 5 ? "tool" : "user", i));
        trim_history(h, 9, 5);   // would leave m5 (tool) in front
        ok(not h.empty() and h.front().value("role").toString() != "tool", "no leading tool");
        ok(first_content(h) == "m6", "tool dropped with its cut call");
        ok(not trim_history(h, 100, 50), "no trim below max");
        std::deque<QJsonObject> g;
        for (int i = 0; i < 11; ++i) g.push_back(msg("user", i));
        trim_history(g, 10, 0);
        ok(g.size() == 5, "trim_to 0 defaults to max/2");
    }
    {   // over budget: trimmed until it fits
        std::deque<QJsonObject> h;
        for (int i = 0; i < 64; ++i) h.push_back(msg("user", i, 350));   // ~100 tokens each
        std::string logged;
        const int d = trim_to_budget(h, 500, 3000, [&](const std::string& s) { logged = s; });
        std::size_t t = 500;
        for (const auto& m : h) t += estimate_tokens(m);
        ok(d > 0 and t <= 3000, "fits after trim");
        ok(h.back().value("content").toString().startsWith("m63"), "newest kept");
        ok(not logged.empty(), "logged");
        ok(trim_to_budget(h, 500, 100000, nullptr) == 0, "under budget untouched");
    }
    {   // one huge message: no endless loop, newest kept, logged
        std::deque<QJsonObject> h;
        h.push_back(msg("user", 0, 10));
        h.push_back(msg("user", 1, 100000));
        std::string logged;
        trim_to_budget(h, 100, 1000, [&](const std::string& s) { logged = s; });
        ok(h.size() == 1 and h.front().value("content").toString().startsWith("m1"), "huge newest kept alone");
        ok(logged.find("alone exceeds") != std::string::npos, "logged the unfixable case");
    }
    {   // images count a flat amount, not their base64
        QJsonObject m = msg("tool", 0);
        m["images"] = QJsonArray{QString(500000, 'A')};
        ok(estimate_tokens(m) < 600, "image ~400 tokens");
    }
    std::println("test_history_trim: {} checks OK", checks);
}
