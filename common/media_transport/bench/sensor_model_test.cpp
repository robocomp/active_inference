// sensor_model_from_json + derived helpers. Pure: no DSR, no DDS traffic.
// Runs under the AGENT's locale (setlocale(LC_ALL,"")), as Qt does: a "C"-locale harness would hide the es_ES
// decimal-comma truncation that the descriptor parser must be immune to.
#include "media_transport.h"
#include <clocale>
#include <cmath>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol; }

int main()
{
    std::setlocale(LC_ALL, "");
    using namespace rc::media;

    MediaDescriptor d;
    d.domain_id = 7; d.type_name = "LidarFrame"; d.type_tag = LIDAR_FRAME_TYPE_TAG;
    d.streams["lidar"] = "helios_points"; d.stream_types["lidar"] = "LidarFrame";

    // 1. descriptor without a model
    CHECK(not sensor_model_from_json(d.to_json()).has_value());

    // 2. descriptor with a model (table of 4 rings, values with a decimal point)
    d.model.version = 1; d.model.source = "simulated"; d.model.frame = "helios";
    d.model.rings = 4; d.model.ring_elev_deg = {-15.5f, -7.25f, 0.f, 14.6f};
    const auto m = sensor_model_from_json(d.to_json());
    CHECK(m.has_value());
    if (m)
    {
        CHECK(m->version == 1 and m->source == "simulated" and m->frame == "helios");
        CHECK(m->ring_elev_deg.size() == 4);
        CHECK(near(lowest_elev_deg(*m).value_or(0), -15.5f));
        CHECK(near(highest_elev_deg(*m).value_or(0), 14.6f));
        CHECK(near(mean_ring_spacing_rad(*m).value_or(0), (14.6f + 15.5f) * 3.14159265f / 180.f / 3.f, 1e-4f));
    }

    // 3. malformed JSON, empty string, wrong type
    CHECK(not sensor_model_from_json("{not json").has_value());
    CHECK(not sensor_model_from_json("").has_value());
    CHECK(not sensor_model_from_json("[1,2,3]").has_value());

    // 4. helpers with no table: fov span / (rings-1)
    SensorModel f; f.version = 1; f.rings = 32; f.fov_start_deg = -15.f; f.fov_end_deg = 15.f;
    CHECK(near(lowest_elev_deg(f).value_or(99), -15.f));
    CHECK(near(highest_elev_deg(f).value_or(99), 15.f));
    CHECK(near(mean_ring_spacing_rad(f).value_or(0), 30.f * 3.14159265f / 180.f / 31.f, 1e-6f));

    // 5. nothing to derive from
    SensorModel e; e.version = 1;
    CHECK(not lowest_elev_deg(e) and not highest_elev_deg(e) and not mean_ring_spacing_rad(e));
    SensorModel one; one.version = 1; one.rings = 1; one.fov_start_deg = 0.f; one.fov_end_deg = 10.f;
    CHECK(not mean_ring_spacing_rad(one).has_value());

    std::printf("%s\n", bad ? "FAILURES" : "ALL OK");
    return bad != 0;
}
