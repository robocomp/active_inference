// carve_replay — scores recorded simulated LiDAR sweeps (carve_replay_rays.py) with the AGENT'S OWN carve code:
// rc::exist::carve_box (the solid fitted cuboid) and rc::exist::carve_parts (the class's composed parts from its
// manifest), both through solid_delta with the agent's sensor model. Not a re-implementation: the same headers the
// agent compiles.
// Usage: carve_replay RAYS.bin MANIFEST_DIR
#include <cstdint>
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
struct Acc { double box = 0.0, parts = 0.0; int n = 0; };

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
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { std::print(stderr, "usage: carve_replay RAYS.bin MANIFEST_DIR\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    if (not f) { std::print(stderr, "cannot open {}\n", argv[1]); return 2; }
    const std::string mdir = argv[2];
    const rc::exist::SensorModel sm{0.03f, 0.85f, 0.05f};   // the agent's: sigma 3 cm, Pd .85, Pc .05

    std::map<std::string, ClassShape> shapes;
    std::map<std::string, Acc> acc;
    std::vector<std::string> order;
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
    }
    std::print("{:<28} {:>6} {:>12} {:>12} {:>12} {:>12}\n", "node", "cycles", "sum_box", "sum_parts", "mean_box", "mean_parts");
    for (const auto& name : order)
    {
        const Acc& a = acc[name];
        std::print("{:<28} {:>6} {:>12.1f} {:>12.1f} {:>+12.2f} {:>+12.2f}\n", name, a.n, a.box, a.parts, a.box / a.n, a.parts / a.n);
    }
}
