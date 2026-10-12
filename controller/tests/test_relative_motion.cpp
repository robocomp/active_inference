// RelativeMotion (controller_relative_motion.h): grammar, pre-checks and a closed-loop run on a kinematic robot.
// Build+run (pure Eigen + std):
//   g++ -std=c++23 -O0 -g -Wall -Wextra -I/usr/include/eigen3 -o /tmp/trm test_relative_motion.cpp && /tmp/trm
#include "../src/controller_relative_motion.h"
#include <cassert>
#include <clocale>
#include <numbers>
#include <print>

using rc::RelativeMotion;

namespace
{
// A 4 x 6 m room [-2,2] x [-3,3] with a 0.47 m square body: free iff the body's bounding circle is inside the walls,
// minus an optional box obstacle.
struct World
{
    float r = 0.34f;   // ~ circumscribed radius of the body
    std::optional<Eigen::AlignedBox2f> box;
    bool free(const Eigen::Vector2f& p, float) const
    {
        if (p.x() - r < -2 or p.x() + r > 2 or p.y() - r < -3 or p.y() + r > 3) return false;
        if (box and box->exteriorDistance(p) < r) return false;
        return true;
    }
    bool inside(const Eigen::Vector2f& p) const { return std::abs(p.x()) < 2 and std::abs(p.y()) < 3; }
};

struct Sim
{
    Eigen::Vector2f pos{0, 0};
    float theta = 0;   // faces +y
    void apply(const RelativeMotion::Command& c, float dt)
    {
        pos += c.adv * dt * RelativeMotion::forward(theta);
        theta += c.rot * dt;
    }
};

// Run to the end; returns the final phase.
RelativeMotion::Phase run(RelativeMotion& rm, Sim& sim, const World& w, std::uint64_t& t)
{
    const auto free = [&](const Eigen::Vector2f& p, float h) { return w.free(p, h); };
    for (int i = 0; i < 4000 and rm.active(); ++i)
    {
        t += 50;
        sim.apply(rm.step(sim.pos, sim.theta, t, free), 0.05f);
    }
    return rm.view().phase;
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // agents run under es_ES: the parser must not care (CLAUDE.md)
    int checks = 0;
    // ── grammar ──
    auto p = RelativeMotion::parse("move 0.5");
    assert(p and p->size() == 1 and (*p)[0].kind == RelativeMotion::Step::Kind::Move and (*p)[0].value == 0.5f); ++checks;
    p = RelativeMotion::parse("move -0.30m");
    assert(p and (*p)[0].value == -0.3f); ++checks;
    p = RelativeMotion::parse("turn 90 then move 1 m");
    assert(p and p->size() == 2 and (*p)[0].kind == RelativeMotion::Step::Kind::Turn); ++checks;
    assert(std::abs((*p)[0].value - std::numbers::pi_v<float> / 2) < 1e-5f); ++checks;
    p = RelativeMotion::parse("turn -180 deg");
    assert(p and std::abs((*p)[0].value + std::numbers::pi_v<float>) < 1e-5f); ++checks;
    assert(not RelativeMotion::parse("move")); ++checks;
    assert(not RelativeMotion::parse("move abc")); ++checks;
    assert(not RelativeMotion::parse("move 5")); ++checks;              // > kMaxMoveM
    assert(not RelativeMotion::parse("turn 400")); ++checks;
    assert(not RelativeMotion::parse("move 0")); ++checks;              // nothing to do
    assert(not RelativeMotion::parse("move 1 deg")); ++checks;
    assert(not RelativeMotion::parse("jump 1")); ++checks;
    assert(RelativeMotion::describe(*RelativeMotion::parse("turn 90 move 0.3")) == "turn 90 move 0.30"); ++checks;

    World w;
    const auto free = [&](const Eigen::Vector2f& q, float h) { return w.free(q, h); };
    const auto inside = [&](const Eigen::Vector2f& q) { return w.inside(q); };
    std::uint64_t t = 1000;

    // ── forward 1 m from the centre ──
    {
        RelativeMotion rm; Sim sim;
        assert(not rm.start(*RelativeMotion::parse("move 1"), sim.pos, sim.theta, t, free, inside)); ++checks;
        assert(run(rm, sim, w, t) == RelativeMotion::Phase::Done); ++checks;
        assert(std::abs(sim.pos.y() - 1.f) < 0.04f and std::abs(sim.pos.x()) < 0.02f); ++checks;
    }
    // ── backward 0.5 m ──
    {
        RelativeMotion rm; Sim sim;
        assert(not rm.start(*RelativeMotion::parse("move -0.5"), sim.pos, sim.theta, t, free, inside)); ++checks;
        assert(run(rm, sim, w, t) == RelativeMotion::Phase::Done); ++checks;
        assert(std::abs(sim.pos.y() + 0.5f) < 0.04f); ++checks;
    }
    // ── turn left 90 then move 1: ends at (-1, 0) facing -x ──
    {
        RelativeMotion rm; Sim sim;
        assert(not rm.start(*RelativeMotion::parse("turn 90 move 1"), sim.pos, sim.theta, t, free, inside)); ++checks;
        assert(run(rm, sim, w, t) == RelativeMotion::Phase::Done); ++checks;
        assert(std::abs(sim.pos.x() + 1.f) < 0.05f and std::abs(sim.pos.y()) < 0.05f); ++checks;
    }
    // ── turn 270 left is 270 left, not 90 right; turn 360 is a full turn ──
    {
        RelativeMotion rm; Sim sim;
        assert(not rm.start(*RelativeMotion::parse("turn 270"), sim.pos, sim.theta, t, free, inside)); ++checks;
        assert(run(rm, sim, w, t) == RelativeMotion::Phase::Done); ++checks;
        assert(std::abs(sim.theta - 1.5f * std::numbers::pi_v<float>) < 0.05f); ++checks;
        Sim s2; RelativeMotion r2;
        assert(not r2.start(*RelativeMotion::parse("turn 360"), s2.pos, s2.theta, t, free, inside)); ++checks;
        assert(run(r2, s2, w, t) == RelativeMotion::Phase::Done and std::abs(s2.theta - 2 * std::numbers::pi_v<float>) < 0.05f); ++checks;
    }
    // ── refusals: out of the room, wall in the way, obstacle in the way ──
    {
        RelativeMotion rm; Sim sim;
        auto why = rm.start(*RelativeMotion::parse("move 2.9"), sim.pos, sim.theta, t, free, inside);
        assert(why and why->find("clear") != std::string::npos); ++checks;     // 2.66 m are clear, the wall stops it
        sim.pos = {0, 2.5f};
        why = rm.start(*RelativeMotion::parse("move 1"), sim.pos, sim.theta, t, free, inside);
        assert(why and why->find("leave the room") != std::string::npos); ++checks;
        assert(rm.view().phase == RelativeMotion::Phase::Failed); ++checks;
        World wb; wb.box = Eigen::AlignedBox2f(Eigen::Vector2f(-0.3f, 0.8f), Eigen::Vector2f(0.3f, 1.2f));
        const auto freeb = [&](const Eigen::Vector2f& q, float h) { return wb.free(q, h); };
        Sim s3;
        why = rm.start(*RelativeMotion::parse("move 1.5"), s3.pos, s3.theta, t, freeb, inside);
        assert(why and why->find("only 0.4") != std::string::npos); ++checks;
        assert(not rm.start(*RelativeMotion::parse("move 0.3"), s3.pos, s3.theta, t, freeb, inside)); ++checks;
    }
    // ── an obstacle appearing mid-move stops it ──
    {
        RelativeMotion rm; Sim sim; World wd;
        const auto freed = [&](const Eigen::Vector2f& q, float h) { return wd.free(q, h); };
        assert(not rm.start(*RelativeMotion::parse("move 2"), sim.pos, sim.theta, t, freed, inside)); ++checks;
        for (int i = 0; i < 20; ++i) { t += 50; sim.apply(rm.step(sim.pos, sim.theta, t, freed), 0.05f); }
        wd.box = Eigen::AlignedBox2f(Eigen::Vector2f(-0.3f, 1.0f), Eigen::Vector2f(0.3f, 1.3f));
        assert(run(rm, sim, wd, t) == RelativeMotion::Phase::Failed); ++checks;
        assert(rm.view().last_event.find("stopped") != std::string::npos); ++checks;
        assert(sim.pos.y() < 1.0f - wd.r + 0.05f); ++checks;
        std::println("  mid-move obstacle: stopped at y={:.2f} — {}", sim.pos.y(), rm.view().last_event);
    }
    // ── cancel ──
    {
        RelativeMotion rm; Sim sim;
        assert(not rm.start(*RelativeMotion::parse("turn 90"), sim.pos, sim.theta, t, free, inside)); ++checks;
        rm.cancel("stopped by the user");
        assert(not rm.active() and rm.view().phase == RelativeMotion::Phase::Cancelled); ++checks;
        assert(rm.step(sim.pos, sim.theta, t, free).rot == 0.f); ++checks;
    }
    std::println("test_relative_motion: {} checks passed", checks);
    return 0;
}
