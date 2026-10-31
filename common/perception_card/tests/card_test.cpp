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
    c.instance_terms = {"motion", "coverage"};
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
    std::printf(bad ? "FAILED (%d)\n" : "ALL OK\n", bad);
    return bad ? 1 : 0;
}
