// Card <-> JSON round trip, closed term list, subtype fallback. Pure. Runs under the AGENT's locale
// (setlocale(LC_ALL,"")) so the es_ES decimal-comma truncation cannot hide.   LANG=es_ES.UTF-8 ./card_test
#include "perception_card/card.h"
#include <clocale>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;

static Card make()
{
    Card c;
    c.concept_ = "door"; c.graph_type = "door"; c.level = Level::Articulated;
    c.channels = {{"zed", Role::Identity, true}, {"helios", Role::Both, false}, {"bpearl", Role::Occupancy, true}};
    c.geometry = {0.7f, 0.05f, 2.0f, 0.0f, 2.05f};
    SubtypeParams d;  d.env = {0.12f, 0.55f, 0.07f}; d.q_return = 0.85f;
    d.terms = {{"envelope", {}}, {"range_conf", {{"r0", 4.5f}, {"k", 0.123456f}}}};
    SubtypeParams r;  r.env = {0.2f, 0.7f, 0.05f}; r.q_return = 0.5f; r.terms = {{"kappa", {{"x", -0.0005f}}}};
    c.subtypes[""] = d; c.subtypes["round"] = r;
    c.instance_terms = {"p_clear", "blind_spot_map"}; c.instance_prior = {{"p_clear", 0.8f}};
    c.blind_spots = {"glass", "mirror"};
    c.calibration = "fitted"; c.calibration_date = "2026-10-31"; c.blind_spot_map_path = "etc/bsm \"x\".json";
    return c;
}
static bool eq(const SubtypeParams& a, const SubtypeParams& b)
{
    if (a.env.min_fill != b.env.min_fill or a.env.max_fill != b.env.max_fill or a.env.soft != b.env.soft
        or a.q_return != b.q_return or a.terms.size() != b.terms.size()) return false;
    for (std::size_t i = 0; i < a.terms.size(); ++i)
        if (a.terms[i].name != b.terms[i].name or a.terms[i].params != b.terms[i].params) return false;
    return true;
}

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    std::printf("LC_NUMERIC = %s\n", std::setlocale(LC_NUMERIC, nullptr));

    const Card a = make();
    const std::string js = to_json(a);
    CHECK(js.find(',') != std::string::npos and js.find("0,05") == std::string::npos);
    const auto r = card_from_json(js);
    CHECK(r.has_value());
    if (r)
    {
        const Card& b = *r;
        CHECK(b.version == a.version and b.concept_ == a.concept_ and b.graph_type == a.graph_type);
        CHECK(b.level == a.level);
        CHECK(b.channels.size() == 3);
        for (std::size_t i = 0; i < b.channels.size() and i < a.channels.size(); ++i)
            CHECK(b.channels[i].sensor == a.channels[i].sensor and b.channels[i].role == a.channels[i].role
                  and b.channels[i].modelled == a.channels[i].modelled);
        CHECK(b.geometry.w == a.geometry.w and b.geometry.d == a.geometry.d and b.geometry.h == a.geometry.h
              and b.geometry.z0 == a.geometry.z0 and b.geometry.z1 == a.geometry.z1);
        CHECK(b.subtypes.size() == 2);
        for (const auto& [k, v] : a.subtypes) CHECK(b.subtypes.contains(k) and eq(v, b.subtypes.at(k)));
        CHECK(b.instance_terms == a.instance_terms and b.blind_spots == a.blind_spots);
        CHECK(b.instance_prior == a.instance_prior and b.instance_prior.at("p_clear") == 0.8f);
        CHECK(b.calibration == a.calibration and b.calibration_date == a.calibration_date);
        CHECK(b.blind_spot_map_path == a.blind_spot_map_path);
        CHECK(to_json(b) == js);
    }

    { // unknown term, in a subtype and in instance_terms
        Card m = make(); m.subtypes[""].terms.push_back({"magic", {}});
        const auto e = card_from_json(to_json(m));
        CHECK(not e and e.error().find("magic") != std::string::npos);
        Card n = make(); n.instance_terms.push_back("magic");
        const auto e2 = card_from_json(to_json(n));
        CHECK(not e2 and e2.error().find("magic") != std::string::npos);
        // a hypothesis term is not an instance term, and vice versa
        Card o = make(); o.instance_terms.push_back("motion");
        CHECK(not card_from_json(to_json(o)));
        Card q2 = make(); q2.subtypes[""].terms.push_back({"p_clear", {}});
        CHECK(not card_from_json(to_json(q2)));
    }
    { // missing level
        std::string j = js;
        const auto p = j.find("\"level\":\"articulated\",");
        CHECK(p != std::string::npos);
        if (p != std::string::npos) j.erase(p, std::string("\"level\":\"articulated\",").size());
        CHECK(not card_from_json(j));
    }
    { // unknown role
        std::string j = js;
        const auto p = j.find("\"identity\"");
        CHECK(p != std::string::npos);
        if (p != std::string::npos) j.replace(p, 10, "\"psychic\"");
        const auto e = card_from_json(j);
        CHECK(not e and e.error().find("psychic") != std::string::npos);
    }
    { // params_for
        const Card c = make();
        CHECK(params_for(c, "round").q_return == 0.5f);
        CHECK(params_for(c, "oval").q_return == 0.85f);
        CHECK(params_for(c, "").q_return == 0.85f);
    }
    { // level 1 (wall occlusion): round trip, negative plane sentinel survives; a card without it loads as "not declared"
        Card w = make(); w.walls = WallOcclusion{0.70f, -1.0f, 0.0f};
        const auto e = card_from_json(to_json(w));
        CHECK(e and e->walls and e->walls->own_wall_skip_m == 0.70f and e->walls->own_wall_plane_m == -1.0f
              and e->walls->wall_thickness_m == 0.0f and to_json(*e) == to_json(w));
        const auto old = card_from_json(to_json(make()));
        CHECK(old and not old->walls);
        std::string j = to_json(w);
        const auto k = j.find("\"wall.thickness_m\"");
        j.erase(k, j.find(',', k) - k + 1);
        CHECK(not card_from_json(j));   // a half-declared block is a schema error
    }
    { // generative block: round trip (es_ES locale), absent block loads as nullopt, closed vocabularies
        Card g = make();
        Generative gb;
        gb.shape = {{Prim::Slab, {{"w", {"sw", 1.0f, 0.0f}}, {"d", {"", 0.0f, 0.4f}}, {"h", {"", 0.0f, 0.04f}},
                                  {"x", {"", 0.0f, 0.0f}}, {"y", {"", 0.0f, 0.0f}}, {"z", {"sh", 1.0f, -0.04f}}}},
                    {Prim::LegSet, {{"n", {"", 0.0f, 4.0f}}, {"half", {"", 0.0f, 0.02f}}, {"fw", {"sw", 1.0f, 0.0f}},
                                    {"fd", {"", 0.0f, 0.4f}}, {"z0", {"", 0.0f, 0.0f}}, {"z1", {"sh", 1.0f, -0.04f}}}}};
        gb.vars = {{"sw", 0.6f, 0.15f, 0.3f, 0.65f, false}, {"sh", 0.595f, 0.123456f, 0.38f, 0.55f, true}};
        gb.placement = {"floor", {{"free", 0.5f}, {"against_wall", 0.25f}}, 0.05f, 0.05f};
        gb.bounds_source = "unit test";
        g.generative = gb;
        const std::string gj = to_json(g);
        const auto e = card_from_json(gj);
        CHECK(e and e->generative and to_json(*e) == gj);
        if (e and e->generative)
        {
            const Generative& r = *e->generative;
            CHECK(r.shape.size() == 2 and r.shape[0].type == Prim::Slab and r.shape[1].type == Prim::LegSet);
            CHECK(r.shape[0].dims.at("z").var == "sh" and r.shape[0].dims.at("z").b == -0.04f);
            CHECK(r.vars.size() == 2 and r.vars[1].sigma == 0.123456f and r.vars[1].log_normal and not r.vars[0].log_normal);
            CHECK(r.placement.contexts.at("against_wall") == 0.25f and r.placement.wall_gap_mean == 0.05f);
            CHECK(r.bounds_source == "unit test");
        }
        CHECK(not card_from_json(to_json(make()))->generative);
        const auto bad_edit = [&](const std::string& from, const std::string& to) -> std::expected<Card, std::string>
        { std::string j = gj; const auto p = j.find(from); CHECK(p != std::string::npos); if (p != std::string::npos) j.replace(p, from.size(), to); return card_from_json(j); };
        const auto e1 = bad_edit("\"slab\"", "\"torus\"");
        CHECK(not e1 and e1.error().find("torus") != std::string::npos);
        const auto e2 = bad_edit("gen.shape.0.dim.w.", "gen.shape.0.dim.q.");     // a dim outside the closed list (and w now missing)
        CHECK(not e2 and e2.error().find("'q'") != std::string::npos);
        const auto e3 = bad_edit("gen.placement.ctx.free", "gen.placement.ctx.afloat");
        CHECK(not e3 and e3.error().find("afloat") != std::string::npos);
        const auto e4 = bad_edit("\"floor\"", "\"ceiling\"");
        CHECK(not e4 and e4.error().find("ceiling") != std::string::npos);
        Card u = g; u.generative->shape[0].dims["z"].var = "undeclared";
        const auto e6 = card_from_json(to_json(u));
        CHECK(not e6 and e6.error().find("undeclared") != std::string::npos);
        Card m = g; m.generative->shape[0].dims.erase("h");
        CHECK(not card_from_json(to_json(m)));   // a primitive must declare every dim of its type
    }
    std::printf(bad ? "FAILED (%d)\n" : "ALL OK\n", bad);
    return bad ? 1 : 0;
}
