// carve_replay — scores recorded simulated LiDAR sweeps (carve_replay_rays.py) with the AGENT'S OWN carve code:
// rc::exist::carve_box (the solid fitted cuboid) and rc::exist::carve_parts (the class's composed parts from its
// manifest), both through solid_delta with the agent's sensor model. Not a re-implementation: the same headers the
// agent compiles.
// Usage: carve_replay RAYS.bin MANIFEST_DIR
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <vector>
#include "existence_belief/existence_belief.h"
#include "concept_manifest/shape_parts.h"
#include "concept_manifest/back_panel.h"
#include "countable_shape_parts.h"

namespace
{
struct ClassShape { std::vector<rc::manifest::ShapePart> parts; std::optional<rc::manifest::BackPanel> panel; };
struct Raw { double occ = 0, in = 0, fr = 0, reached = 0; void add(const rc::exist::Evidence& e) { occ += e.e_occ; in += e.e_interior; fr += e.e_free; reached += e.n_reached; } };
struct Acc { double box = 0.0, parts = 0.0; int n = 0; Raw rb, rp; std::vector<double> occ_by_part; };

// DIAGNOSTIC ONLY (does not touch what carve_parts returns): the part holding each return = max inside mass
// (Phi((1-tn)/st) - Phi((1-tf)/st)) of the same slab test; that beam's e_occ is the agent's carve_parts on the lone beam.
std::size_t holding_part(const Eigen::Vector3f& origin, const Eigen::Vector3f& q, float cx, float cy, float yaw,
                         std::span<const rc::exist::BoxPart> parts, float surf, const rc::exist::SensorModel& sm)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    const auto phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };
    const float sg = std::sqrt(sm.sensor_sigma_m * sm.sensor_sigma_m + surf * surf);
    const Eigen::Vector3f O(c * (origin.x() - cx) + s * (origin.y() - cy), -s * (origin.x() - cx) + c * (origin.y() - cy), origin.z());
    const Eigen::Vector3f P(c * (q.x() - cx) + s * (q.y() - cy), -s * (q.x() - cx) + c * (q.y() - cy), q.z());
    const Eigen::Vector3f dir = P - O;
    const float st = sg / std::max(dir.norm(), 1e-4f);
    std::size_t best = parts.size(); float bm = 0.0f;
    for (std::size_t k = 0; k < parts.size(); ++k)
    {
        const auto& pt = parts[k];
        const float lo[3] = {pt.cx - 0.5f * pt.sx, pt.cy - 0.5f * pt.sy, pt.z0}, hi[3] = {pt.cx + 0.5f * pt.sx, pt.cy + 0.5f * pt.sy, pt.z1};
        float tn = -1e30f, tf = 1e30f; bool miss = false;
        for (int a = 0; a < 3 and not miss; ++a)
        {
            if (std::abs(dir(a)) < 1e-6f) { if (O(a) < lo[a] or O(a) > hi[a]) miss = true; continue; }
            float t1 = (lo[a] - O(a)) / dir(a), t2 = (hi[a] - O(a)) / dir(a);
            if (t1 > t2) std::swap(t1, t2);
            tn = std::max(tn, t1); tf = std::min(tf, t2);
        }
        if (miss or tn > tf or tf < 0.0f or tn < 0.0f) continue;
        const float m = std::max(0.0f, phi((1.0f - tn) / st) - phi((1.0f - tf) / st));
        if (m > bm) { bm = m; best = k; }
    }
    return best;
}

template <class T> bool rd(std::ifstream& f, T* p, std::size_t n = 1)
{
    return static_cast<bool>(f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(sizeof(T) * n)));
}

std::string class_of(std::string name)
{
    for (const char* pre : {"phantom_floor_", "phantom_wall_"})
        if (name.starts_with(pre)) name = name.substr(std::string(pre).size());
    return name.substr(0, name.find('_'));
}

// --explain: for the part holding the return, which face the beam entered through, the depth below it, ws.
struct Explained { std::size_t part = 0; int face = -1; float depth = 0, ws = 0, len = 0; bool ok = false; };
Explained explain_beam(const Eigen::Vector3f& origin, const Eigen::Vector3f& q, float cx, float cy, float yaw,
                       std::span<const rc::exist::BoxPart> parts, float surf, const rc::exist::SensorModel& sm)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    const auto phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };
    const float sg = std::sqrt(sm.sensor_sigma_m * sm.sensor_sigma_m + surf * surf);
    const Eigen::Vector3f O(c * (origin.x() - cx) + s * (origin.y() - cy), -s * (origin.x() - cx) + c * (origin.y() - cy), origin.z());
    const Eigen::Vector3f P(c * (q.x() - cx) + s * (q.y() - cy), -s * (q.x() - cx) + c * (q.y() - cy), q.z());
    const Eigen::Vector3f dir = P - O;
    const float len = dir.norm(), st = sg / std::max(len, 1e-4f);
    Explained best; float bm = 0.0f;
    for (std::size_t k = 0; k < parts.size(); ++k)
    {
        const auto& pt = parts[k];
        const float lo[3] = {pt.cx - 0.5f * pt.sx, pt.cy - 0.5f * pt.sy, pt.z0}, hi[3] = {pt.cx + 0.5f * pt.sx, pt.cy + 0.5f * pt.sy, pt.z1};
        float tn = -1e30f, tf = 1e30f; bool miss = false; int face = -1;
        for (int a = 0; a < 3 and not miss; ++a)
        {
            if (std::abs(dir(a)) < 1e-6f) { if (O(a) < lo[a] or O(a) > hi[a]) miss = true; continue; }
            float t1 = (lo[a] - O(a)) / dir(a), t2 = (hi[a] - O(a)) / dir(a);
            const bool lo_first = t1 <= t2;
            if (not lo_first) std::swap(t1, t2);
            if (t1 > tn) { tn = t1; face = 2 * a + (lo_first ? 0 : 1); }   // 0:-x 1:+x 2:-y 3:+y 4:-z(bottom) 5:+z(top)
            tf = std::min(tf, t2);
        }
        if (miss or tn > tf or tf < 0.0f or tn < 0.0f) continue;
        const float m = std::max(0.0f, phi((1.0f - tn) / st) - phi((1.0f - tf) / st));
        if (m > bm) { bm = m; best = {k, face, (1.0f - tn) * len, std::exp(-0.5f * (1.0f - tn) * (1.0f - tn) * len * len / (sg * sg)), len, true}; }
    }
    return best;
}
const char* const kFace[6] = {"-x", "+x", "-y", "+y", "bottom", "top"};
}  // namespace

int main(int argc, char** argv)
{
    bool breakdown = false; std::string explain;
    struct Top { float e; Eigen::Vector3f o, q; Explained x; std::string part; };
    std::vector<Top> top; std::array<double, 6> face_hist{}; std::map<int, double> depth_hist, z_hist; double ex_n = 0; bool ex_header = false;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) { if (std::strcmp(argv[i], "--breakdown") == 0) breakdown = true;
        else if (std::strcmp(argv[i], "--explain") == 0 and i + 1 < argc) explain = argv[++i];
        else pos.push_back(argv[i]); }
    if (pos.size() < 2) { std::print(stderr, "usage: carve_replay RAYS.bin MANIFEST_DIR [--breakdown]\n"); return 2; }
    std::ifstream f(pos[0], std::ios::binary);
    if (not f) { std::print(stderr, "cannot open {}\n", pos[0]); return 2; }
    const std::string mdir = pos[1];
    const rc::exist::SensorModel sm{0.03f, 0.85f, 0.05f};   // the agent's: sigma 3 cm, Pd .85, Pc .05

    std::map<std::string, ClassShape> shapes;
    std::map<std::string, Acc> acc;
    std::vector<std::string> order;
    std::map<std::string, std::vector<std::string>> partnames;
    std::uint32_t nn;
    while (rd(f, &nn))
    {
        std::string name(nn, '\0');
        float v[8], o[3]; std::uint32_t n;
        if (not rd(f, name.data(), nn) or not rd(f, v, 8) or not rd(f, o, 3) or not rd(f, &n)) break;
        std::vector<Eigen::Vector3f> sweep(n);
        if (n and not rd(f, reinterpret_cast<float*>(sweep.data()), 3 * static_cast<std::size_t>(n))) break;

        const std::string cls = class_of(name);
        if (not shapes.contains(cls))
        {
            const std::string path = mdir + "/" + cls + ".concept.toml";
            shapes[cls] = {rc::manifest::load_shape_parts(path), rc::manifest::load_back_panel(path)};
            if (shapes[cls].parts.empty()) { std::print(stderr, "{}: no [[shape.part]] for class '{}' (name {})\n", path, cls, name); return 2; }
        }
        const ClassShape& cs = shapes[cls];
        const float cx = v[0], cy = v[1], yaw = v[2], w = v[3], h = v[4], H = v[5];
        const float surf = std::sqrt(0.5f * (v[6] * v[6] + v[7] * v[7]));   // belief footprint sigma, as the agent
        const Eigen::Vector3f origin(o[0], o[1], o[2]);

        const auto eb = rc::exist::carve_box(origin, sweep, cx, cy, yaw, w, h, 0.0f, H, surf, sm);
        const auto parts = rc::countable::carve_parts_for(w, h, H, cs.parts, cs.panel);
        const auto ep = rc::exist::carve_parts(origin, sweep, cx, cy, yaw, parts, surf, sm);

        if (not acc.contains(name)) order.push_back(name);
        Acc& a = acc[name];
        a.box += rc::exist::solid_delta(eb, sm);
        a.parts += rc::exist::solid_delta(ep, sm);
        ++a.n;
        a.rb.add(eb); a.rp.add(ep);
        if (not explain.empty() and name == explain)
        {
            if (not ex_header)
            {
                std::print("[explain {}] first scored box: world cx {:.3f} cy {:.3f} yaw {:.3f} w {:.3f} h {:.3f} H {:.3f}; surface_sigma {:.4f} -> sigma_surf {:.4f}\n",
                           name, cx, cy, yaw, w, h, H, surf, std::sqrt(0.03f * 0.03f + surf * surf));
                ex_header = true;
            }
            for (const auto& q : sweep)
            {
                const auto e1 = rc::exist::carve_parts(origin, std::vector<Eigen::Vector3f>{q}, cx, cy, yaw, parts, surf, sm);
                if (e1.e_occ <= 0.05f) continue;
                const Explained x = explain_beam(origin, q, cx, cy, yaw, parts, surf, sm);
                if (not x.ok) continue;
                face_hist[x.face] += 1; ex_n += 1;
                depth_hist[static_cast<int>(std::floor(x.depth / 0.02f))] += 1;
                z_hist[static_cast<int>(std::floor(q.z() / 0.05f))] += 1;
                top.push_back({e1.e_occ, origin, q, x, x.part < cs.parts.size() ? cs.parts[x.part].name : "back_panel"});
            }
        }
        if (breakdown)
        {
            a.occ_by_part.resize(parts.size() + 1, 0.0);   // last slot: no part holds the return
            for (const auto& q : sweep)
            {
                const auto e1 = rc::exist::carve_parts(origin, std::vector<Eigen::Vector3f>{q}, cx, cy, yaw, parts, surf, sm);
                if (e1.e_occ > 0.0f) a.occ_by_part[holding_part(origin, q, cx, cy, yaw, parts, surf, sm)] += e1.e_occ;
            }
            partnames[name].clear();
            for (std::size_t k = 0; k < parts.size(); ++k)
                partnames[name].push_back(k < cs.parts.size() ? cs.parts[k].name : "back_panel");
        }
    }
    std::print("{:<28} {:>6} {:>12} {:>12} {:>12} {:>12}\n", "node", "cycles", "sum_box", "sum_parts", "mean_box", "mean_parts");
    for (const auto& name : order)
    {
        const Acc& a = acc[name];
        if (a.rb.reached == 0.0 and a.rp.reached == 0.0) { std::print("{:<28} {:>6} UNPROBED (no beam reached it)\n", name, a.n); continue; }
        std::print("{:<28} {:>6} {:>12.1f} {:>12.1f} {:>+12.2f} {:>+12.2f}\n", name, a.n, a.box, a.parts, a.box / a.n, a.parts / a.n);
        if (not breakdown) continue;
        std::print("   [breakdown] {} cycles | box:   occ {:.1f} interior {:.1f} free {:.1f} n_reached {:.0f} | parts: occ {:.1f} interior {:.1f} free {:.1f} n_reached {:.0f}\n",
                   a.n, a.rb.occ, a.rb.in, a.rb.fr, a.rb.reached, a.rp.occ, a.rp.in, a.rp.fr, a.rp.reached);
        std::print("   [breakdown] parts occ by holding part:");
        for (std::size_t k = 0; k < a.occ_by_part.size(); ++k)
            std::print(" {}={:.1f}", k + 1 < a.occ_by_part.size() ? partnames[name][k] : std::string("none"), a.occ_by_part[k]);
        std::print("\n");
    }
    if (not explain.empty() and ex_n > 0)
    {
        std::print("[explain {}] beams with e_occ>0.05: {:.0f}\nentry face:", explain, ex_n);
        for (int k = 0; k < 6; ++k) std::print(" {}={:.0f}", kFace[k], face_hist[k]);
        std::print("\ndepth (2 cm bins, m):");
        for (const auto& [b, n] : depth_hist) std::print(" [{:.2f}]={:.0f}", b * 0.02f, n);
        std::print("\nreturn z (5 cm bins, m):");
        for (const auto& [b, n] : z_hist) std::print(" [{:.2f}]={:.0f}", b * 0.05f, n);
        std::sort(top.begin(), top.end(), [](const Top& a, const Top& b) { return a.e > b.e; });
        std::print("\ntop 5 by e_occ:\n");
        for (std::size_t i = 0; i < std::min<std::size_t>(5, top.size()); ++i)
        {
            const auto& t = top[i];
            std::print("  e_occ {:.3f} origin ({:.2f},{:.2f},{:.2f}) return ({:.2f},{:.2f},{:.2f}) part {} face {} depth {:.3f} ws {:.3f} len {:.2f}\n",
                       t.e, t.o.x(), t.o.y(), t.o.z(), t.q.x(), t.q.y(), t.q.z(), t.part, kFace[t.x.face], t.x.depth, t.x.ws, t.x.len);
        }
    }
}
