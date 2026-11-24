// neighbour_size_test — the interim size std of a neighbour's published box (plan rev 2 F1.7): its class from the node name,
// and Δ_w/√12, Δ_h/√12 from that class's manifest. Run: ./neighbour_size_test ../../common/concept_manifest
#include <cmath>
#include <cstdio>
#include <string>
#include "countable_neighbour_size.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "../../../common/concept_manifest";
    CHECK(rc::node_class("nightstand_2") == "nightstand");
    CHECK(rc::node_class("bed_1") == "bed");
    CHECK(rc::node_class("refrigerator_12") == "refrigerator");
    CHECK(rc::node_class("cabinet_run_3") == "cabinet_run");
    CHECK(rc::node_class("toilet") == "toilet");          // no numeric suffix: the name is the class
    CHECK(rc::node_class("door_") == "door_");            // an empty suffix is not a serial
    const float k = 1.0f / std::sqrt(12.0f);
    // the manifests' declared Δ (w, h): bed 0.20 / 0.03, nightstand 0.04 / 0.05, toilet 0.25 / 0.12
    struct Want { const char* cls; float dw, dh; };
    for (const Want w : {Want{"bed", 0.20f, 0.03f}, Want{"nightstand", 0.04f, 0.05f}, Want{"toilet", 0.25f, 0.12f}})
    {
        const auto sd = rc::neighbour_size_std(dir + "/" + w.cls + ".concept.toml");
        CHECK(sd.has_value());
        if (sd) std::printf("  %-10s sigma_w %.4f sigma_h %.4f\n", w.cls, (*sd)[0], (*sd)[1]);
        CHECK(sd and std::abs((*sd)[0] - w.dw * k) < 1e-7f and std::abs((*sd)[1] - w.dh * k) < 1e-7f);
    }
    CHECK(not rc::neighbour_size_std(dir + "/no_such_class.concept.toml").has_value());   // no manifest ⇒ none (caller: 0)
    std::printf(bad ? "neighbour_size_test: %d FAILED\n" : "neighbour_size_test: OK\n", bad);
    return bad != 0;
}
