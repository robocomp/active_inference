// write_memory_attrs: a door's existence memory read back from a real (standalone) DSRGraph; absent optionals are not written.
// Build/run: see tests/CMakeLists.txt (target test_door_memory_graph).
#include "door_memory_graph.h"

#include <cstdio>

namespace
{
int failures = 0;
void check(bool ok, int line, const char* what) { if (not ok) { ++failures; std::printf("FAIL line %d: %s\n", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)
}

int main()
{
    using namespace rc::door::memnote;
    DSR::DSRGraph G("door_memory_graph_test", 995, std::string(SHADOW_JSON), true, static_cast<int8_t>(85), DSR::Queue);

    {   // everything present
        DSR::Node n = DSR::Node::create<object_node_type>("door_t1");
        NoteAttrs a;
        a.exist_evidence = 2.5f; a.exist_physical = -0.5f; a.crossed = true; a.seen_ms = 1700000000123ull; a.sighted_ms = 1700000000456ull;
        a.place_sd = 0.07f; a.height_sd = 0.03f; a.phi_sd = 0.2f; a.ledger = std::string("v1;v2"); a.visit = 4;
        write_memory_attrs(G, n, a);
        G.insert_node(n);
        auto r = G.get_node("door_t1");
        CHECK(r.has_value());
        CHECK(G.get_attrib_by_name<door_exist_evidence_att>(*r).value_or(-9.f) == 2.5f);
        CHECK(G.get_attrib_by_name<door_exist_physical_att>(*r).value_or(9.f) == -0.5f);
        CHECK(G.get_attrib_by_name<door_crossed_att>(*r).value_or(false) == true);
        CHECK(G.get_attrib_by_name<door_seen_ms_att>(*r).value_or(0) == 1700000000123ull);
        CHECK(G.get_attrib_by_name<door_sighted_ms_att>(*r).value_or(0) == 1700000000456ull);
        CHECK(G.get_attrib_by_name<door_place_sd_att>(*r).value_or(-9.f) == 0.07f);
        CHECK(G.get_attrib_by_name<door_height_sd_att>(*r).value_or(-9.f) == 0.03f);
        CHECK(G.get_attrib_by_name<door_phi_sd_att>(*r).value_or(-9.f) == 0.2f);
        CHECK(G.get_attrib_by_name<door_view_ledger_att>(*r).has_value() and
              std::string(G.get_attrib_by_name<door_view_ledger_att>(*r).value().get()) == "v1;v2");
        CHECK(G.get_attrib_by_name<door_visit_att>(*r).value_or(-1) == 4);
        CHECK(not G.get_attrib_by_name<door_adopted_att>(*r).has_value());   // not this writer's
    }
    {   // only evidence present: the rest is absent, never a 0
        DSR::Node n = DSR::Node::create<object_node_type>("door_t2");
        NoteAttrs a;
        a.exist_evidence = -1.0f;
        write_memory_attrs(G, n, a);
        G.insert_node(n);
        auto r = G.get_node("door_t2");
        CHECK(r.has_value());
        CHECK(G.get_attrib_by_name<door_exist_evidence_att>(*r).value_or(9.f) == -1.0f);
        CHECK(not G.get_attrib_by_name<door_exist_physical_att>(*r).has_value());
        CHECK(not G.get_attrib_by_name<door_crossed_att>(*r).has_value());
        CHECK(not G.get_attrib_by_name<door_seen_ms_att>(*r).has_value());
        CHECK(not G.get_attrib_by_name<door_place_sd_att>(*r).has_value());
        CHECK(not G.get_attrib_by_name<door_height_sd_att>(*r).has_value());
        CHECK(not G.get_attrib_by_name<door_view_ledger_att>(*r).has_value());
        CHECK(not G.get_attrib_by_name<door_visit_att>(*r).has_value());
    }
    {   // write on change: a second identical note carries nothing; one changed field carries only that field
        NoteAttrs a;
        a.exist_evidence = 1.f; a.place_sd = 0.1f; a.ledger = std::string("L1"); a.visit = 2;
        NoteAttrs last;
        const auto d1 = changed_since(a, nullptr);
        CHECK(any_change(d1) and d1.ledger.has_value());
        apply_changes(last, d1);
        CHECK(not any_change(changed_since(a, &last)));
        NoteAttrs b = a; b.exist_evidence = 1.5f; b.ledger.reset();   // ledger not re-serialised: absent, not "cleared"
        const auto d2 = changed_since(b, &last);
        CHECK(any_change(d2) and d2.exist_evidence.value_or(0) == 1.5f and not d2.place_sd and not d2.ledger and not d2.visit);
        apply_changes(last, d2);
        CHECK(last.ledger.value_or("") == "L1" and last.exist_evidence.value_or(0) == 1.5f);
        CHECK(not any_change(changed_since(b, &last)));
    }
    std::printf(failures == 0 ? "OK\n" : "%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
