// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/wall_support_test.cpp -o /tmp/wst && /tmp/wst [run_dir frame_index ...]
//
// rc::walls::wall_real_prob (wall_support.h): P(an outline edge is a real wall | the free-space raster), the obj_exist_prob
// room_concept publishes on its walls and ltsm's per-edge extent reads. Synthetic cases always; with arguments, the live
// refiner's exact input (<run_dir>/refine_input.bin, frame index) with that frame's fitted sigma/eps/ell
// (<run_dir>/layout_refine.csv), printing the support of every outline edge.
#include "wall_support.h"

#include <charconv>
#include <cstring>
#include <clocale>
#include <fstream>
#include <print>
#include <random>
#include <sstream>
#include <string>

namespace
{
int failures = 0;
void check(bool ok, int line, const char* what)
{
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
    else std::println("  ok   {}", what);
}
#define CHECK(x) check((x), __LINE__, #x)
using V = Eigen::Vector2f;

struct Input { std::int64_t ts = 0; std::vector<V> poly, surf, fr; std::vector<float> cs, cf; };
std::vector<Input> read_inputs(const std::string& path)
{
    std::vector<Input> out;
    std::ifstream f(path, std::ios::binary);
    auto get = [&](auto& v) { f.read(reinterpret_cast<char*>(&v), sizeof(v)); return bool(f); };
    auto pts = [&](std::vector<V>& P, std::vector<float>* c)
    {
        std::uint32_t n = 0; if (not get(n)) return false;
        for (std::uint32_t i = 0; i < n; ++i)
        {
            float x, y, w = 1.f; get(x); get(y); if (c) { get(w); c->push_back(w); }
            P.emplace_back(x, y);
        }
        return bool(f);
    };
    while (true)
    {
        Input in;
        if (not get(in.ts) or not pts(in.poly, nullptr) or not pts(in.surf, &in.cs) or not pts(in.fr, &in.cf)) break;
        out.push_back(std::move(in));
    }
    return out;
}

/// sigma, eps, ell of the layout_refine.csv row stamped `ts` (columns ts,surf,free,sigma,eps,ell,...), from_chars only.
bool fit_of(const std::string& csv, std::int64_t ts, rc::walls::SupportParams& P)
{
    std::ifstream f(csv);
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line))
    {
        std::vector<std::string> c;
        std::stringstream ss(line);
        for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
        if (c.size() < 6) continue;
        std::int64_t t = 0;
        std::from_chars(c[0].data(), c[0].data() + c[0].size(), t);
        if (t != ts) continue;
        std::from_chars(c[3].data(), c[3].data() + c[3].size(), P.sigma);
        std::from_chars(c[4].data(), c[4].data() + c[4].size(), P.eps);
        std::from_chars(c[5].data(), c[5].data() + c[5].size(), P.ell);
        return true;
    }
    return false;
}

std::vector<float> supports(const std::vector<V>& poly, const rc::walls::EvidenceRaster& R, rc::walls::SupportParams P)
{
    float a2 = 0.f, per = 0.f;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const auto& p = poly[i]; const auto& q = poly[(i + 1) % poly.size()];
        a2 += p.x() * q.y() - q.x() * p.y(); per += (q - p).norm();
    }
    P.r_eff = 0.5f * std::abs(a2) / per;
    const float o = a2 >= 0.f ? 1.f : -1.f;
    std::vector<float> s;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const V a = poly[i], b = poly[(i + 1) % poly.size()];
        const float L = (b - a).norm();
        const V u = L > 0.f ? V((b - a) / L) : V(1.f, 0.f);
        s.push_back(rc::walls::wall_real_prob(a, b, o * V(u.y(), -u.x()), R, P));
    }
    return s;
}
}   // namespace

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    {   // synthetic 6 x 4 room. Bottom wall seen (hits); right wall: a phantom, free space continues 2 m beyond it; top wall:
        // a real wall with a cupboard in front of its left half (free runs stop 0.6 m short there) and hits on its right
        // half; left wall: never observed (no run reaches it).
        const float c = 0.05f;
        std::vector<V> surf, fr; std::vector<float> cs, cf;
        std::mt19937 rng(3); std::normal_distribution<float> nz(0.f, 0.01f);
        for (float x = 0.5f * c; x < 8.f; x += c)
            for (float y = 0.5f * c; y < 4.f; y += c)
            {
                if (x < 1.0f) continue;                                   // left strip never observed
                if (y > 3.4f - c and x < 3.f) continue;                   // behind the cupboard
                fr.emplace_back(x, y); cf.push_back(1.f);
            }
        for (float x = 1.f + 0.5f * c; x < 6.f; x += c) { surf.emplace_back(x, nz(rng)); cs.push_back(1.f); }
        for (float x = 3.f + 0.5f * c; x < 6.f; x += c) { surf.emplace_back(x, 4.f + nz(rng)); cs.push_back(1.f); }
        for (float x = 1.f + 0.5f * c; x < 3.f; x += c) { surf.emplace_back(x, 3.4f + nz(rng)); cs.push_back(1.f); }   // cupboard face
        const std::vector<V> room{{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        const rc::walls::EvidenceRaster R(surf, cs, fr, cf, c);
        rc::walls::SupportParams P; P.cell = c; P.sigma = 0.03f; P.eps = 0.1f; P.ell = 0.5f;
        const auto s = supports(room, R, P);
        std::println("synthetic: bottom {:.3f} right(phantom) {:.3f} top(half occluded) {:.3f} left(unobserved) {:.3f}", s[0], s[1], s[2], s[3]);
        CHECK(s[0] > 0.99f);
        CHECK(s[1] < 0.01f);
        CHECK(s[2] > 0.99f);    // occlusion is not absence: the cupboard half adds nothing against
        CHECK(std::abs(s[3] - 0.5f) < 1e-3f);   // never observed: the prior
    }
    // --walls <run_dir> <frame> <file>: the support of given wall SEGMENTS (lines "name ax ay bx by", the outline's frame) on
    // that frame's raster, oriented by that frame's outline -- exactly what publish_wall_seen_prob does per wall node.
    if (argc >= 5 and std::string(argv[1]) == "--walls")
    {
        const std::string dir = argv[2];
        int idx = 0;
        std::from_chars(argv[3], argv[3] + std::strlen(argv[3]), idx);
        const auto in = read_inputs(dir + "/refine_input.bin");
        if (idx < 0) idx += int(in.size());
        const auto& f = in.at(std::size_t(idx));
        rc::walls::SupportParams P;
        if (not fit_of(dir + "/layout_refine.csv", f.ts, P)) { std::println("no fit"); return 1; }
        float a2 = 0.f, per = 0.f;
        for (std::size_t i = 0; i < f.poly.size(); ++i)
        {
            const auto& p = f.poly[i]; const auto& q = f.poly[(i + 1) % f.poly.size()];
            a2 += p.x() * q.y() - q.x() * p.y(); per += (q - p).norm();
        }
        P.r_eff = 0.5f * std::abs(a2) / per;
        const float o = a2 >= 0.f ? 1.f : -1.f;
        const rc::walls::EvidenceRaster R(f.surf, f.cs, f.fr, f.cf, P.cell);
        std::ifstream w(argv[4]);
        std::string line;
        while (std::getline(w, line))
        {
            std::stringstream ss(line);
            std::string name; std::string t[4]; float v[4];
            if (not (ss >> name >> t[0] >> t[1] >> t[2] >> t[3])) continue;
            for (int k = 0; k < 4; ++k) std::from_chars(t[k].data(), t[k].data() + t[k].size(), v[k]);
            const V a(v[0], v[1]), b(v[2], v[3]);
            const V u = (b - a).normalized();
            std::println("{} {:.4f}", name, rc::walls::wall_real_prob(a, b, o * V(u.y(), -u.x()), R, P));
        }
        return 0;
    }
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string dir = argv[i];
        int idx = 0;
        std::from_chars(argv[i + 1], argv[i + 1] + std::strlen(argv[i + 1]), idx);
        const auto in = read_inputs(dir + "/refine_input.bin");
        if (idx < 0) idx += int(in.size());
        if (idx < 0 or idx >= int(in.size())) { std::println("{}: no frame {}", dir, idx); ++failures; continue; }
        const auto& f = in[std::size_t(idx)];
        rc::walls::SupportParams P;
        if (not fit_of(dir + "/layout_refine.csv", f.ts, P)) { std::println("{}: no fit for ts {}", dir, f.ts); ++failures; continue; }
        const rc::walls::EvidenceRaster R(f.surf, f.cs, f.fr, f.cf, P.cell);
        const auto s = supports(f.poly, R, P);
        std::print("{} frame {} (sigma {:.4f} eps {:.4f} ell {:.3f}):", dir, idx, P.sigma, P.eps, P.ell);
        for (std::size_t e = 0; e < s.size(); ++e) std::print(" {}:{:.2f}m={:.3f}", e, (f.poly[(e + 1) % f.poly.size()] - f.poly[e]).norm(), s[e]);
        std::println("");
    }
    std::println("{} ({} failures)", failures ? "FAILED" : "all ok", failures);
    return failures ? 1 : 0;
}
