// door_approach_law.h — the door approach as a state-feedback ("thermostat") law, on a kinematic unicycle with wall points.
//   (a) 2.7 m from the door, door 90 deg off, beside a wall corner, the turn refused by the wall: backs off along its trail,
//       turns, drives, ends at V facing the door, never touching.
//   (b) pushed 0.5 m sideways at V: returns to V and faces the door again.
//   (c) starts facing away: rotates BEFORE it translates.
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 test_door_approach_law.cpp -o /tmp/test_door_approach_law && /tmp/test_door_approach_law
#include "../src/door_approach_law.h"

#include <print>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

using V2 = Eigen::Vector2f;
namespace da = rc::door_approach;

// Room y > 0 above the door wall y = 0 (door at the origin, normal +y), right wall x = 3.
std::vector<V2> walls()
{
    std::vector<V2> w;
    for (float x = -3.f; x <= 3.f; x += 0.02f) w.push_back({x, 0.f});
    for (float y = 0.f; y <= 4.f; y += 0.02f) { w.push_back({3.f, y}); w.push_back({-3.f, y}); }
    return w;
}
constexpr float kBody = 0.33f;     // the body's circumscribed radius: contact when a wall point is nearer than this
constexpr float kSweep = 0.45f;    // a turn is refused while a wall point is nearer than this (the grown body's reach)

struct Sim
{
    V2 p; float theta; std::vector<V2> trail; std::vector<V2> wall = walls();
    bool backed = false;
    float min_clear = 1e9f, travelled_before_aligned = 0.f; bool aligned = false;
    float clearance() const { float m = 1e9f; for (const auto& q : wall) m = std::min(m, (q - p).norm()); return m; }
    // route: [start..lead, V] polyline with a forward-only projection
    std::vector<V2> route; float prog = 0.f;
    float len() const { float L = 0; for (size_t i = 1; i < route.size(); ++i) L += (route[i] - route[i - 1]).norm(); return L; }
    V2 at(float s) const
    {
        for (size_t i = 1; i < route.size(); ++i)
        {
            const float l = (route[i] - route[i - 1]).norm();
            if (s <= l or i + 1 == route.size()) return route[i - 1] + (route[i] - route[i - 1]) * std::clamp(s / std::max(l, 1e-6f), 0.f, 1.f);
            s -= l;
        }
        return route.back();
    }
    void project()
    {
        float best = 1e9f, bs = prog;
        for (float s = prog; s <= len(); s += 0.01f) if (const float d = (at(s) - p).norm(); d < best) { best = d; bs = s; }
        prog = bs;
    }
    da::Command step(const V2& V, const V2& door, float dt)
    {
        if (trail.empty() or (trail.back() - p).norm() >= 0.05f) trail.push_back(p);
        project();
        da::State st;
        st.p = p; st.theta = theta; st.vantage = V; st.door = door; st.rho = 0.25f; st.v_max = 0.5f; st.w_max = 0.7f; st.tau = 1.f;
        const float L = len();
        st.aim = at(std::min(prog + 0.4f, L));
        st.d_route = std::max(0.f, L - prog) + (p - at(prog)).norm();
        st.trail = trail;
        auto c = da::law(st);
        if (c.rot != 0.f and clearance() < kSweep) { st.w_cap = 0.f; c = da::law(st); }
        backed = backed or c.mode == da::Mode::BackOff;
        const float f = theta + float(M_PI_2);
        const V2 np = p + c.adv * dt * V2(std::cos(f), std::sin(f));
        const V2 old = p;
        p = np;
        if (clearance() < kBody) p = old;   // the base's own stop: no motion into contact
        if (std::abs(da::heading_error(old, theta, door)) < 0.8f and not aligned) aligned = true;
        if (not aligned) travelled_before_aligned += (p - old).norm();
        theta += c.rot * dt;
        min_clear = std::min(min_clear, clearance());
        return c;
    }
};
}  // namespace

int main()
{
    const V2 door(0.f, 0.f), V(0.f, 1.0f), lead(0.f, 1.4f);

    {   // (a)
        Sim s; s.p = {2.5f, 0.40f}; s.theta = -float(M_PI);   // facing the door wall, the door 90 deg to the left
        for (float y = 1.6f; y >= 0.40f; y -= 0.05f) s.trail.push_back({2.5f, y});
        s.route = {s.p, {2.5f, 1.4f}, lead, V};
        da::Command c;
        for (int i = 0; i < 1200; ++i) c = s.step(V, door, 0.1f);
        CHECK((s.p - V).norm() < 0.25f);
        CHECK(std::abs(da::heading_error(s.p, s.theta, door)) < 0.087f);
        CHECK(s.min_clear >= kBody);
        CHECK(s.backed);   // the refused turn was given room by retracing the trail
        std::println("(a) end ({:.2f},{:.2f}) psi_door {:.1f} deg min clearance {:.2f}", s.p.x(), s.p.y(),
                     da::heading_error(s.p, s.theta, door) * 57.3f, s.min_clear);
    }
    {   // (b) pushed 0.5 m sideways at V
        Sim s; s.p = {0.5f, 1.0f}; s.theta = -float(M_PI) + 0.1f;
        s.route = {{0.f, 1.4f}, V}; s.prog = 0.4f;
        for (int i = 0; i < 600; ++i) s.step(V, door, 0.1f);
        CHECK((s.p - V).norm() < 0.25f);
        CHECK(std::abs(da::heading_error(s.p, s.theta, door)) < 0.087f);
        CHECK(s.min_clear >= kBody);
    }
    {   // (c) starts facing away from the door, 3 m out on the normal
        Sim s; s.p = {0.f, 3.0f}; s.theta = 0.f + float(M_PI_2) * 0.f - float(M_PI_2) * 0.f;   // forward = +y: away from the door
        s.route = {s.p, lead, V};
        for (int i = 0; i < 1000; ++i) s.step(V, door, 0.1f);
        CHECK(s.travelled_before_aligned < 0.3f);   // it turned before it translated
        CHECK((s.p - V).norm() < 0.25f);
        CHECK(std::abs(da::heading_error(s.p, s.theta, door)) < 0.087f);
        std::println("(c) travelled {:.2f} m before facing within 46 deg", s.travelled_before_aligned);
    }
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
