// The plan window driven by a real Interpreter on a scripted host — no robot, no graph, no controller.
//
//   plan_view_demo                 a window: the mission plays out in ~25 s (a door that fails once, a loop, a wait)
//   plan_view_demo --snapshot DIR  offscreen PNGs of the same run at a few instants (QT_QPA_PLATFORM=offscreen)
//
// Each tick encodes the interpreter's snapshot with encode_progress() and decodes it with decode_progress(), the
// same round trip the controller_mission node puts between the controller and the chat.
#include "../src/plan_view.h"
#include "../../common/mission/mission_json.h"
#include "../../common/mission/mission_progress.h"
#include "../../common/mission/mission_text.h"

#include <QApplication>
#include <QTimer>

#include <map>
#include <print>

using namespace rc::mission;

namespace
{
/// Leaves finish on a schedule: [id][attempt] → (ms after the leaf started, how it ends).
struct ScriptHost final : Host
{
    std::uint64_t* now;
    std::map<std::string, std::uint64_t> began;
    std::map<std::string, int> attempts;
    explicit ScriptHost(std::uint64_t* n) : now(n) {}

    std::optional<std::string> start(const Step& s) override
    {
        began[s.id] = *now;
        ++attempts[s.id];
        return std::nullopt;
    }
    LeafStatus poll(const Step& s) override
    {
        const auto dt = *now - began[s.id];
        if (s.op == "cross" and s.id == "2" and attempts[s.id] == 1 and dt > 1500)
            return {.state = LeafState::Failed, .detail = "unreachable"};
        if ((s.op == "cross" or s.op == "go_to" or s.op == "open") and dt > 2000) return {.state = LeafState::Succeeded};
        return {};
    }
    void cancel(const Step&) override {}
    void say(const std::string&) override {}
    void halt() override {}
    std::optional<Pose> pose() override { return Pose{.room = "room_1", .x = 0.4, .y = -1.1, .yaw = 1.57}; }
    std::optional<double> eval(const Pred& p) override   // the door was crossed into room_2
    {
        if (p.name == "in_room") return p.refs.at("room").name == "room_2" ? 1.0 : 0.0;
        return std::nullopt;
    }
};

constexpr std::string_view kMission = R"(mission "kitchen round"
  anchor start
  cross door_1 expect=room_2 retry=2
  repeat 2
    open door_7
    wait s=4s
  explore room_2 budget_s=6s
  go_to anchor:start
)";
}   // namespace

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    const bool snapshot = argc >= 3 and std::string_view(argv[1]) == "--snapshot";

    std::uint64_t now = 1'760'000'000'000;   // a fixed epoch: the snapshots do not depend on the day they ran
    ScriptHost host(&now);
    Interpreter in(host);
    const Mission m = parse_text(kMission).value();

    rc::chat::PlanView view;
    view.set_clock([&now] { return now; });
    view.set_mission(m, QString::fromStdString(encode_json(m)));
    in.start(m, now);

    const auto step = [&](std::uint64_t dt)
    {
        now += dt;
        in.tick(now);
        view.set_progress(decode_progress(encode_progress(in, "remote")).value());
    };

    if (snapshot)
    {
        view.resize(640, 360);
        view.show();
        const QString dir = QString::fromUtf8(argv[2]);
        const std::map<std::uint64_t, QString> at{{1200, "1_first_try"}, {2300, "2_retry"}, {9500, "3_loop_wait"},
                                                  {19500, "4_explore"}, {40000, "5_done"}};
        std::uint64_t t = 0;
        for (const auto& [when, name] : at)
        {
            while (t < when) { step(100); t += 100; }
            now += 250;   // a quarter period on: the halo is mid-pulse, the flash mid-fade
            QApplication::processEvents();   // let the layout catch up with the labels' new text
            view.grab().save(dir + "/" + name + ".png");
            now -= 250;
            std::println("{}: {}", name.toStdString(), view.model().headline(now));
        }
        return 0;
    }

    view.show();
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&] { step(100); });
    timer.start(100);
    return app.exec();
}
