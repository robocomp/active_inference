#pragma once
/*
 * canvas_commands.h — the side canvas' fast path: "show me the zed camera", "show me the room", "show me the plan",
 * "show me the graph [of room_1]", "show me the robot", "draw object chair_1", "show me the lidars",
 * "close the canvas" are drawn at once with a stock script, without the model (2026-10-18).
 *
 * Only these fixed phrasings (English and the obvious Spanish) — anything looser ("draw the doors in red") is a
 * sentence for the model, which has the draw tool. Pure, header-only; tests/test_canvas.cpp.
 */

#include <algorithm>
#include <optional>
#include <regex>
#include <string>

namespace rc::chat
{

struct ShowRequest
{
    enum class Kind { camera, layout, plan, graph, model, lidar, hide } kind;
    std::string arg;   ///< camera: "zed" | "ricoh"; graph: the node to zoom in on ("" = all); model: "robot" | a node | a class;
                       ///< lidar: "helios" | "bpearl" | "" (both)
};

[[nodiscard]] inline std::optional<ShowRequest> parse_show(const std::string& lowered)
{
    std::string t = lowered;
    while (not t.empty() and (t.back() == '.' or t.back() == '!' or t.back() == '?' or t.back() == ' ')) t.pop_back();
    static const std::regex show(R"re(^(?:please )?(?:show(?: me)?|display|open|draw|dibuja|muestrame|muéstrame|ensename|enséñame))re"
                                 R"re( (?:the |your |my |la |el |las |los |tu )?(.+?)(?: please)?$)re");
    static const std::regex hide(R"re(^(?:hide|close|cierra|oculta) (?:the |el )?(?:canvas|lienzo|drawing|dibujo)$)re");
    if (std::regex_match(t, hide)) return ShowRequest{ShowRequest::Kind::hide, ""};
    // 3-D models: "draw object chair_1", "show me the robot", "show me the 3d model of the fridge", "the table model"
    static const std::regex draw_object(R"re(^(?:draw|show(?: me)?) (?:the )?(?:object|3d object) ([a-z0-9_ ]+)$)re");
    static const std::regex model_of(R"re(^(?:draw|show(?: me)?|display) (?:the |a )?(?:3d |3-d )?model of (?:the )?([a-z0-9_ ]+)$)re");
    static const std::regex x_model(R"re(^(?:draw|show(?: me)?|display) (?:the |a )?([a-z0-9_ ]+?) (?:3d |3-d )?model$)re");
    static const std::regex robot(R"re(^(?:draw|show(?: me)?|display) (?:the )?(?:robot|yourself)(?: in 3d)?$|^(?:muestrame|muéstrame) el robot$)re");
    // 3-D LiDARs: "show me the lidars", "draw the 3d lidar", "show me the helios", "the point cloud", "el lidar"
    static const std::regex lidar(R"re(^(?:draw|show(?: me)?|display|muestrame|muéstrame|dibuja) (?:the |los |el |la )?)re"
                                  R"re((?:(3d |3-d )?lidars?|(3d |3-d )?lidar (?:points|point cloud|clouds?)|point clouds?|laser|lasers|)re"
                                  R"re(helios(?: lidar)?|bpearl(?: lidar)?|nube de puntos)$)re");
    if (std::regex_match(t, lidar))
        return ShowRequest{ShowRequest::Kind::lidar,
                           t.find("helios") != std::string::npos ? "helios" : t.find("bpearl") != std::string::npos ? "bpearl" : ""};
    if (std::regex_match(t, robot)) return ShowRequest{ShowRequest::Kind::model, "robot"};
    for (const auto* re : {&draw_object, &model_of, &x_model})
        if (std::smatch mm; std::regex_match(t, mm, *re))
        {
            std::string what = mm[1].str();
            while (not what.empty() and what.back() == ' ') what.pop_back();
            if (what.size() > 1 and what.back() >= '0' and what.back() <= '9' and what.find(' ') != std::string::npos)
                std::ranges::replace(what, ' ', '_');   // "chair 1" → chair_1; "dining table" stays words
            return ShowRequest{ShowRequest::Kind::model, what};
        }
    std::smatch m;
    if (not std::regex_match(t, m, show)) return std::nullopt;
    const std::string what = m[1].str();
    static const std::regex zed(R"re(^(?:zed|front|frontal)(?: camera)?$|^(?:camara|cámara) (?:zed|frontal|delantera)$|^camera$|^(?:camara|cámara)$)re");
    static const std::regex ricoh(R"re(^(?:ricoh|360|panorama|panoramic)(?: camera)?$|^(?:camara|cámara) (?:ricoh|360|panoramica|panorámica)$)re");
    static const std::regex layout(R"re(^(?:room|map|layout|floor plan|plan of the room|layout of the room|room layout|)re"
                                   R"re(habitacion|habitación|mapa|plano(?: de la habitaci[oó]n)?)$)re");
    static const std::regex plan(R"re(^(?:plan|mission|mission plan|misi[oó]n)$)re");
    if (std::regex_match(what, zed)) return ShowRequest{ShowRequest::Kind::camera, "zed"};
    if (std::regex_match(what, ricoh)) return ShowRequest{ShowRequest::Kind::camera, "ricoh"};
    if (std::regex_match(what, layout)) return ShowRequest{ShowRequest::Kind::layout, ""};
    if (std::regex_match(what, plan)) return ShowRequest{ShowRequest::Kind::plan, ""};
    // "graph", "dsr graph", "working memory", "graph of room_1" (the node name as written: names are lower-case)
    static const std::regex graph(R"re(^(?:(?:dsr |working memory )?graph|working memory|grafo|memoria de trabajo))re"
                                  R"re((?: (?:of|around|for|de|del) (?:the |el |la )?([a-z0-9_ ]+))?$)re");
    if (std::smatch gm; std::regex_match(what, gm, graph))
    {
        std::string focus = gm[1].str();
        std::ranges::replace(focus, ' ', '_');   // "room 1" → room_1
        return ShowRequest{ShowRequest::Kind::graph, focus};
    }
    return std::nullopt;
}

/// A library name a person or the model may give a saved script: [a-z0-9_], 1..40 chars; "" when nothing is left.
[[nodiscard]] inline std::string canvas_script_name(const std::string& raw)
{
    std::string s;
    for (char c : raw)
    {
        if (c >= 'A' and c <= 'Z') c = char(c - 'A' + 'a');
        if ((c >= 'a' and c <= 'z') or (c >= '0' and c <= '9') or c == '_') s += c;
        else if ((c == ' ' or c == '-') and not s.empty() and s.back() != '_') s += '_';
        if (s.size() == 40) break;
    }
    while (not s.empty() and s.back() == '_') s.pop_back();
    return s;
}

}   // namespace rc::chat
