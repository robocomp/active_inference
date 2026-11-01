#include "canvas_model.h"

#include <algorithm>
#include <format>

#include <QCoreApplication>

#include "../../common/obj/obj_loader.h"
#include "approach.h"

namespace rc::chat
{

std::filesystem::path assets_root()
{
    return std::filesystem::path(QCoreApplication::applicationDirPath().toStdString()).parent_path().parent_path();
}

namespace
{
std::optional<std::filesystem::path> resolve(const std::string& rel, const std::vector<std::filesystem::path>& bases)
{
    const std::filesystem::path p(rel);
    if (p.is_absolute()) return std::filesystem::exists(p) ? std::optional(p) : std::nullopt;
    for (const auto& b : bases)
        if (std::filesystem::exists(b / p)) return b / p;
    return std::nullopt;
}

std::string lower(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not s.empty() and s.back() == ' ') s.pop_back();
    if (s.starts_with("the ")) s = s.substr(4);
    return s;
}
}   // namespace

std::expected<ModelToShow, std::string> find_model(DSR::DSRGraph& G, const std::vector<ObjectFact>& objects,
                                                   const std::string& what_raw)
{
    const std::string what = lower(what_raw);
    const auto root = assets_root();
    std::optional<DSR::Node> node;
    std::string rel;
    bool unit_model = false;

    const auto robots = G.get_nodes_by_type("robot");
    if (what.empty() or what == "robot" or what == "you" or what == "yourself" or
        (not robots.empty() and lower(robots.front().name()) == what))
    {
        if (robots.empty()) return std::unexpected(std::string("there is no robot node in the graph"));
        node = robots.front();
        const auto p = G.get_attrib_by_name<path_att>(node.value());
        if (not p.has_value() or p->get().empty())
            return std::unexpected("the robot node '" + node->name() + "' carries no mesh (no `path` attribute)");
        rel = p->get();
    }
    else
    {
        node = G.get_node(what);
        if (not node.has_value())
        {
            const auto o = pick_object(objects, what);
            if (not o.has_value()) return std::unexpected(o.error());
            node = G.get_node(o->id);
            if (not node.has_value()) return std::unexpected("'" + o->name + "' has just left the graph");
        }
        const auto p = G.get_attrib_by_name<mesh_path_att>(node.value());
        if (not p.has_value() or p->empty())
            return std::unexpected("'" + node->name() + "' carries no 3-D model (no mesh_path attribute)");
        rel = p.value();
        unit_model = true;
    }

    const auto file = resolve(rel, {root, root / "robot_concept", std::filesystem::current_path()});
    if (not file.has_value()) return std::unexpected("the model file '" + rel + "' of " + node->name() + " was not found");
    auto mesh = rc::obj::load_obj_mesh_data(*file);
    if (not mesh.has_value() or mesh->triangles.empty())
        return std::unexpected("the model file '" + rel + "' could not be read");

    ModelToShow m{.node = node->name(), .file = *file, .triangles = std::move(mesh->triangles)};
    m.description = node->name() + " — " + rel;
    if (unit_model)
    {
        const auto w = G.get_attrib_by_name<width_m_att>(node.value());
        const auto d = G.get_attrib_by_name<depth_m_att>(node.value());
        const auto h = G.get_attrib_by_name<height_m_att>(node.value());
        if (w.has_value() and d.has_value() and h.has_value())
        {
            m.size = QVector3D(*w, *d, *h);
            for (auto& v : m.triangles) v *= *m.size;
            m.description += std::format(", {:.2f} x {:.2f} x {:.2f} m", *w, *d, *h);
        }
        else m.description += " (size unknown: drawn as the unit model)";
    }
    return m;
}

}   // namespace rc::chat
