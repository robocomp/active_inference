// imagine_view (src/imagine_view.h): the answer formatter on a fixed McResult, the place words, and the whole path on a
// stub scene (chair card with a generative block, a 6x4 room, a ZED and the two LiDARs).
#include "../src/imagine_view.h"

#include <clocale>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <sstream>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL  %s  (line %d)\n", #c, __LINE__); ++fails; } } while (0)

using namespace rc::chat;
using namespace rc::card;

static bool has(const std::string& s, const char* w) { return s.find(w) != std::string::npos; }

static Generative chair_gen()
{
    const auto K = [](float b) { return Bind{"", 0.0f, b}; };
    const auto V = [](const char* v, float a = 1.0f, float b = 0.0f) { return Bind{v, a, b}; };
    const float T = 0.075f;
    Generative g;
    g.shape.push_back({Prim::Slab, {{"w", V("seat_w")}, {"d", V("seat_d")}, {"h", K(T)}, {"x", K(0)}, {"y", K(0)}, {"z", V("seat_h", 1, -T)}}});
    g.shape.push_back({Prim::Panel, {{"w", V("seat_w")}, {"t", K(T)}, {"h", V("back_h")}, {"x", K(0)}, {"y", V("seat_d", -0.5f, 0.5f * T)}, {"z", V("seat_h")}}});
    g.shape.push_back({Prim::LegSet, {{"n", K(4)}, {"half", K(0.02f)}, {"fw", V("seat_w")}, {"fd", V("seat_d")}, {"z0", K(0)}, {"z1", V("seat_h", 1, -T)}}});
    g.vars = {{"seat_w", 0.60f, 0.15f, 0.30f, 0.65f, false}, {"seat_d", 0.52f, 0.15f, 0.30f, 0.65f, false},
              {"seat_h", 0.595f, 0.15f, 0.38f, 0.70f, false}, {"back_h", 0.655f, 0.15f, 0.30f, 0.80f, false}};
    g.placement.support = "floor";
    g.placement.contexts = {{"free", 0.5f}, {"against_wall", 0.5f}};
    g.placement.wall_gap_mean = 0.05f; g.placement.wall_gap_sigma = 0.05f;
    return g;
}

static Card chair_card(const char* calibration = "fitted")
{
    Card c; c.concept_ = "chair"; c.graph_type = "object"; c.level = Level::Countable;
    c.channels = {{"zed", Role::Identity, true}};
    c.geometry = {0.45f, 0.45f, 0.9f, 0.0f, 0.9f};
    c.walls = WallOcclusion{0.15f, -1.0f, 0.0f};
    SubtypeParams d; d.env = {0.0f, 1.32f, 0.26f}; d.q_return = 0.8f;
    c.subtypes[""] = d;
    c.calibration = calibration;
    c.generative = chair_gen();
    return c;
}

static LidarRig helios()
{
    LidarRig r;
    r.model.version = 1; r.model.rings = 16; r.model.fov_start_deg = -55.0f; r.model.fov_end_deg = 15.0f;
    r.model.azimuth_step_deg = 0.8f; r.model.range_min_m = 0.1f; r.model.range_max_m = 80.0f;
    r.mount = Eigen::Isometry3f::Identity();
    r.mount.linear() = Eigen::AngleAxisf(std::numbers::pi_v<float>, Eigen::Vector3f::UnitX()).toRotationMatrix();
    r.mount.translation() = Eigen::Vector3f(0, 0, 1.075f);
    return r;
}

static SeeScene scene(bool with_chair, float zed_z = 0.945f, bool height_known = true)
{
    SeeScene s;
    s.room = "room_1";
    s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    s.robot_xy = Eigen::Vector2f(1, 2);
    s.robot_yaw = 0.0f - std::numbers::pi_v<float> / 2.0f;   // faces +x
    SeeDoor d; d.name = "door_1"; d.xy = {3, 0}; d.width_m = 1.0f; d.phi_rad = 0.0f;
    s.doors.push_back(d);
    SeeObject t; t.name = "table_1"; t.subtype = "table"; t.x = 4.5f; t.y = 2.0f; t.w = 1.2f; t.d = 0.7f; t.h = 0.75f;
    t.standoff = Eigen::Vector2f(3.5f, 2.0f);
    s.objects.push_back(t);
    if (with_chair)
    {
        SeeObject c; c.name = "chair_1"; c.subtype = "chair"; c.x = 2.0f; c.y = 2.0f; c.w = 0.45f; c.d = 0.45f; c.h = 0.9f;
        s.objects.push_back(c);
    }
    rc::nbv::Sensor z;
    constexpr float f = 448.0f;
    z.hfov_rad = 2.0f * std::atan(640.0f / f); z.vfov_rad = 2.0f * std::atan(360.0f / f); z.height_m = zed_z;
    s.zed = z; s.zed_mount_z = zed_z;
    s.helios = helios();
    s.room_height_known = height_known;
    return s;
}

static std::string slurp(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main()
{
    std::setlocale(LC_ALL, "");
    // ---- the answer formatter on a fixed McResult and fixed facts ----
    {
        McResult mc; mc.p_notice_mean = 0.8123f; mc.p_notice_sd = 0.0449f; mc.p_recognise_mean = 0.31f; mc.p_recognise_sd = 0.0801f;
        mc.modal_limit = Limit::TooSmall; mc.n = 32;
        ImagineFacts f; f.cls = "chair"; f.place_words = "against a wall"; f.shown = 8; f.accepted = 30; f.drawn = 60;
        f.rejected = 25; f.outside_polygon = 5; f.inside_count = 2; f.excluded = {"chair_1"}; f.room_height_known = false;
        f.seed = 18446744073709551615ull; f.calibration = "prior";
        const auto t = answer_imagine_view(f, mc);
        CHECK(has(t, "8 chairs against a wall, seen from here"));
        CHECK(has(t, "P 0.81 ± 0.04") and has(t, "P 0.31 ± 0.08"));
        CHECK(has(t, "too small"));                                            // the modal limit, in words
        CHECK(has(t, "the viewpoint is inside the imagined chair"));
        CHECK(has(t, "Only 50%"));                                             // acceptance below 0.8
        CHECK(has(t, "left chair_1 out"));
        CHECK(has(t, "Room height 2.5 m assumed"));
        CHECK(has(t, "seed 18446744073709551615"));
        CHECK(has(t, "isn't calibrated yet"));
        f.accepted = 60; f.inside_count = 0; f.excluded.clear(); f.room_height_known = true; f.calibration = "fitted";
        const auto u = answer_imagine_view(f, mc);
        CHECK(not has(u, "Only ") and not has(u, "inside the imagined") and not has(u, "left ") and not has(u, "assumed")
              and not has(u, "calibrated"));
        f.shown = 0;
        CHECK(has(answer_imagine_view(f, mc), "could not place any imagined chair"));
    }
    // ---- places ----
    {
        const auto s = scene(true);
        CHECK(not resolve_place(s, "chair", "").value().ctx.at and not resolve_place(s, "chair", "").value().ctx.force_context);
        CHECK(resolve_place(s, "chair", "against the wall").value().ctx.force_context == std::optional<std::string>("against_wall"));
        CHECK(resolve_place(s, "chair", "junto a la pared").value().ctx.force_context.has_value());
        const auto dp = resolve_place(s, "chair", "door_1").value();
        CHECK(dp.ctx.at and std::abs(dp.ctx.at->y() - 0.3f) < 1e-4f and dp.words == "at door_1");
        const auto tp = resolve_place(s, "chair", "table_1").value();
        CHECK(tp.ctx.at and std::abs(tp.ctx.at->x() - 3.5f) < 1e-6f and tp.exclude.empty());
        const auto cp = resolve_place(s, "chair", "chair_1").value();
        CHECK(cp.exclude.size() == 1 and cp.exclude[0] == "chair_1");
        CHECK(not resolve_place(s, "chair", "the moon").has_value());
        CHECK(is_hypothetical(scene(false), "chair", "door_1") and not is_hypothetical(s, "chair", "door_1")
              and not is_hypothetical(scene(false), "chair", "") and not is_hypothetical(scene(false), "chair", "wall"));
    }
    // ---- the whole path on the stub scene ----
    const auto dir = std::filesystem::temp_directory_path() / "imagine_view_test";
    std::filesystem::remove_all(dir);
    {
        const auto s = scene(true, 0.945f, false);
        ImagineRequest rq; rq.concept_ = "chair"; rq.where = "against the wall"; rq.from = ""; rq.n = 8;
        const auto o = imagine_view(chair_card(), s, rq, 4242, dir);
        CHECK(o.drawn);
        CHECK(has(o.text, "seed 4242") and has(o.text, "against a wall") and has(o.text, "Room height 2.5 m assumed"));
        CHECK(has(o.text, "8 chairs") and has(o.text, "±"));
        const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(slurp(dir / "imagined.json"))).object();
        CHECK(doc["samples"].toArray().size() == 8 and doc["seed"].toString() == "4242");
        CHECK(doc["lidar"].toArray().size() == 2 and doc["lidar"].toArray()[0].toArray().size() == 8);
        CHECK(doc["lidar"].toArray()[0].toArray()[0].toObject()["modelled"].toBool());          // helios modelled
        CHECK(not doc["lidar"].toArray()[1].toArray()[0].toObject()["modelled"].toBool());      // no bpearl in the scene
        CHECK(std::filesystem::exists(dir / "imagined_zed.png"));
        CHECK(doc["mc"].toObject()["n"].toInt() == 32);
        // against_wall is forced: every sample's backrest is within ~0.5 m of the polygon edge
        bool all_wall = true;
        for (const auto& v : doc["samples"].toArray())
        {
            const auto sm = v.toObject();
            const double x = sm["x"].toDouble(), y = sm["y"].toDouble();
            const double dmin = std::min({x, 6 - x, y, 4 - y});
            all_wall = all_wall and dmin < 0.8;
        }
        CHECK(all_wall);
        // deterministic: the same seed gives byte-identical data
        const auto dir2 = std::filesystem::temp_directory_path() / "imagine_view_test_b";
        const auto o2 = imagine_view(chair_card(), s, rq, 4242, dir2);
        CHECK(o2.text == o.text and slurp(dir / "imagined.json") == slurp(dir2 / "imagined.json"));
        std::filesystem::remove_all(dir2);
        // sensor = zed: no LiDAR scans; sensor = lidar: no PNG
        rq.sensor = "zed";
        (void)imagine_view(chair_card(), s, rq, 1, dir);
        const auto dz = QJsonDocument::fromJson(QByteArray::fromStdString(slurp(dir / "imagined.json"))).object();
        CHECK(dz["lidar"].toArray()[0].toArray().isEmpty() and std::filesystem::exists(dir / "imagined_zed.png"));
        rq.sensor = "lidar";
        (void)imagine_view(chair_card(), s, rq, 1, dir);
        CHECK(not std::filesystem::exists(dir / "imagined_zed.png"));
    }
    // ---- the target is not its own occluder; the viewpoint inside the imagined chair is reported ----
    {
        const auto s = scene(true, 0.55f);
        ImagineRequest rq; rq.concept_ = "chair"; rq.where = "chair_1"; rq.from = "chair_1"; rq.n = 8;
        const auto o = imagine_view(chair_card(), s, rq, 7, dir);
        // the robot's body is an obstacle of the sampler: where it stands nothing can be imagined, so no P for a chair it stands in
        CHECK(not o.drawn and has(o.text, "could not place any imagined chair") and has(o.text, "my own body"));
        CHECK(not has(o.text, "I would notice"));
        // at door_1, from door_1: the same, never a P
        rq.where = "door_1"; rq.from = "door_1";
        const auto od = imagine_view(chair_card(), s, rq, 7, dir);
        CHECK(not od.drawn and has(od.text, "could not place any imagined chair at door_1") and not has(od.text, "I would notice"));
        CHECK(has(can_see_hypothetical_reply(chair_card(), scene(false), "chair", "door_1", "door_1"), "no free place"));
        CHECK(not has(can_see_hypothetical_reply(chair_card(), scene(false), "chair", "door_1", "door_1"), "recognise it"));
        // the footprint test: a seat below the camera counts, the camera's own height does not matter
        {
            Placed pl = instantiate(chair_gen(), {{"seat_w", 0.6f}, {"seat_d", 0.52f}, {"seat_h", 0.595f}, {"back_h", 0.655f}}, 2.0f, 2.0f, 0.3f);
            CHECK(viewpoint_in_footprint(Viewpoint{{2.0f, 2.0f}, std::nullopt}, pl));
            CHECK(not viewpoint_in_footprint(Viewpoint{{3.0f, 2.0f}, std::nullopt}, pl));
        }
        rq.where = "chair_1"; rq.from = "chair_1";
        // another place: nothing excluded, nothing inside
        rq.where = "table_1"; rq.from = "door_1";
        const auto o2 = imagine_view(chair_card(), s, rq, 7, dir);
        CHECK(not has(o2.text, "left ") and not has(o2.text, "inside the imagined") and has(o2.text, "by table_1"));
        // fixed place: the P(detect) is over the pictures' own draw (heading drawn, rejection), n = max(8, 32) accepted
        CHECK(o2.drawn);
        const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(slurp(dir / "imagined.json"))).object();
        CHECK(doc["mc"].toObject()["n"].toInt() == 32 and doc["samples"].toArray().size() == 8);
        // R12: an uncertain door (no phi, no P(open)) is taken as shut, and the answer says so; a measured door says nothing
        CHECK(not has(o2.text, "Assumed"));
        auto su = s; su.doors[0].phi_rad.reset();
        const auto o3 = imagine_view(chair_card(), su, rq, 7, dir);
        CHECK(has(o3.text, "Assumed: door_1") and has(o3.text, "taken as shut"));
        auto hu = scene(false); hu.doors[0].phi_rad.reset();
        CHECK(has(can_see_hypothetical_reply(chair_card(), hu, "chair", "table_1", "door_1"), "Assumed: door_1"));
    }
    // ---- exclusion by the instance's own footprint; is_hypothetical never steals a named instance ----
    {
        const auto card = chair_card();
        const auto count = [&](const SeeScene& sc, const Eigen::Vector2f& at, const char* cls, std::vector<std::string>& ex)
        {
            ImaginePlace pl; pl.ctx.at = at;
            return imagine_world(sc, card, cls, pl, ex).obstacles.size();
        };
        std::vector<std::string> ex;
        auto s = scene(true);                                   // table_1 (4.5,2) 1.2x0.7, chair_1 (2,2) 0.45x0.45
        const auto n0 = s.objects.size();
        CHECK(count(s, {2.1f, 2.1f}, "chair", ex) == n0 - 1 and ex == std::vector<std::string>{"chair_1"});   // inside it
        CHECK(count(s, {2.4f, 2.0f}, "chair", ex) == n0 and ex.empty());            // 0.4 m away, outside the footprint
        CHECK(count(s, {4.1f, 2.0f}, "chair", ex) == n0 and ex.empty());            // inside the TABLE: another class stays
        SeeObject fr; fr.name = "refrigerator_1"; fr.subtype = "refrigerator"; fr.x = 1.0f; fr.y = 1.0f; fr.w = 1.2f; fr.d = 1.0f; fr.h = 1.8f;
        s.objects.push_back(fr);
        CHECK(count(s, {1.5f, 1.0f}, "refrigerator", ex) == n0 and ex == std::vector<std::string>{"refrigerator_1"});   // 0.5 m from its centre
        s.objects.back().w.reset(); s.objects.back().d.reset();   // unpublished: the card's geometry half-extents (0.45)
        CHECK(count(s, {1.5f, 1.0f}, "refrigerator", ex) == n0 and ex.empty());   // not excluded (a sizeless instance is no obstacle either)
        CHECK(count(s, {1.1f, 1.0f}, "refrigerator", ex) == n0 and ex == std::vector<std::string>{"refrigerator_1"});   // inside the card-sized footprint
        // is_hypothetical
        auto h = scene(false);
        SeeObject blank; blank.name = "thing_1"; blank.subtype = ""; blank.x = 1; blank.y = 1;
        h.objects.push_back(blank);
        CHECK(not is_hypothetical(h, "chair", "thing_1"));          // an instance with an empty class: the old path
        CHECK(is_hypothetical(h, "chair", "table_1"));              // no chair, a known different class
        CHECK(not is_hypothetical(scene(true), "chair", "table_1")); // a chair exists: the old path evaluates the named instance
        CHECK(not is_hypothetical(scene(true), "chair", "chair_1"));
    }
    // ---- the evaluator and the renderer agree on where the camera looks (live: chair in the ZED render, P 0.00 everywhere) ----
    {
        const auto s = scene(false);                      // robot at (1,2), faces +x
        const auto vp = resolve_from(s, "");
        CHECK(vp.has_value());
        Hypothesis h; h.concept_ = "chair"; h.x = 2.5f; h.y = 2.0f; h.yaw = 0.0f;   // 1.5 m straight ahead
        const auto r = evaluate(chair_card(), world_for(s, ""), h, *vp);
        CHECK(r.p_notice > 0.5f and r.p_recognise > 0.5f);
        CHECK(r.limit != Limit::OutOfView);
        Hypothesis hb = h; hb.x = 1.0f; hb.y = 3.5f;      // 1.5 m to the LEFT of the robot (+y): outside the field of view
        const auto rb = evaluate(chair_card(), world_for(s, ""), hb, *vp);
        CHECK(rb.p_notice < 0.1f and rb.limit == Limit::OutOfView);
        // the SAME out-of-view judgement for every heading: the label must not depend on the world bearing of the target
        // (v_no_fov looked along world +x, so a target beyond +-86 deg of +x was never "out of view": limit None)
        int n_heading = 0;
        for (float rad = -3.0f; rad <= 3.0f; rad += 0.5f)
            for (const float off : {-2.5f, -1.5f, -1.0f, 1.0f, 1.5f, 2.5f, 3.1f})   // beyond the +-30 deg half field of view
            {
                auto sr = s; sr.robot_yaw = rad; sr.robot_xy = Eigen::Vector2f(3.0f, 2.0f); sr.objects.clear(); sr.doors.clear();
                const auto vr = resolve_from(sr, "");
                Hypothesis ho; ho.concept_ = "chair"; ho.yaw = 0.0f;
                ho.x = 3.0f + 1.5f * std::cos(*vr->yaw + off); ho.y = 2.0f + 1.5f * std::sin(*vr->yaw + off);
                if (ho.x < 0.2f or ho.x > 5.8f or ho.y < 0.2f or ho.y > 3.8f) continue;   // keep it inside the 6x4 room
                ++n_heading;
                CHECK(evaluate(chair_card(), world_for(sr, ""), ho, *vr).limit == Limit::OutOfView);
            }
        CHECK(n_heading >= 30);
    }
    // ---- no room: imagine in free space in front of the robot; a named place still needs the room ----
    {
        auto s = scene(false);
        s.room.clear(); s.polygon.clear(); s.doors.clear(); s.objects.clear(); s.robot_xy.reset();
        ImagineRequest rq; rq.concept_ = "chair"; rq.n = 8;
        const auto o = imagine_view(chair_card(), s, rq, 7, dir);
        CHECK(o.drawn and not o.refusal);
        CHECK(has(o.text, "typical chairs standing on the floor") and has(o.text, "m in front of me") and has(o.text, "±"));
        CHECK(not has(o.text, "from here") and not has(o.text, "room") and not has(o.text, "Room height") and not has(o.text, "against a wall"));
        const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(slurp(dir / "imagined.json"))).object();
        CHECK(doc["samples"].toArray().size() == 8);
        // the thing stands straight ahead, in view of the camera: it is noticed, not 0.00 from a wrong heading
        CHECK(doc["mc"].toObject()["p_notice_mean"].toDouble() > 0.2);
        ImagineRequest rw = rq; rw.where = "against the wall";
        const auto r = imagine_view(chair_card(), s, rw, 7, dir);
        CHECK(not r.drawn and r.refusal and r.text == kNoRoomRefusal);
        // a refusal flag on the other refusal sites
        Card nogen = chair_card(); nogen.generative.reset();
        CHECK(imagine_view(nogen, scene(true), rq, 1, dir).refusal);
        ImagineRequest rbad = rq; rbad.where = "the moon";
        CHECK(imagine_view(chair_card(), scene(true), rbad, 1, dir).refusal);
        CHECK(not imagine_view(chair_card(), scene(true), rq, 1, dir).refusal);
    }
    // ---- the canonical view: the robot's idea of the thing, independent of the room ----
    {
        auto none = scene(false);
        none.room.clear(); none.polygon.clear(); none.doors.clear(); none.objects.clear(); none.robot_xy.reset();
        auto walled = scene(true);                       // a room, doors, a chair and a table; wall 1 m in front of the robot
        walled.polygon = {{0, 0}, {2, 0}, {2, 4}, {0, 4}};
        ImagineRequest rq; rq.concept_ = "chair"; rq.n = 8;
        const auto a = imagine_view(chair_card(), none, rq, 11, dir);
        const auto b = imagine_view(chair_card(), walled, rq, 11, dir);
        CHECK(a.drawn and b.drawn and a.text == b.text);
        // "here" is a named place: the room mixture, and it needs the room
        ImagineRequest rh = rq; rh.where = "here";
        const auto h = imagine_view(chair_card(), scene(false), rh, 11, dir);
        CHECK(has(h.text, "wherever such things usually stand"));
        CHECK(imagine_view(chair_card(), none, rh, 11, dir).refusal);
        // the distance comes from the envelope: fills [0.1, 0.4] -> the mean-size instance fills 0.2 there
        Card c = chair_card();
        c.subtypes[""].env = {0.1f, 0.4f, 0.05f};
        c.geometry = {0.45f, 0.45f, 0.9f, 0.0f, 0.9f};
        const auto sc = scene(false);
        const auto cd = canonical_distance(c, sc.zed);
        CHECK(cd.from_envelope);
        rc::nbv::Target t; t.cx = cd.m; t.cy = 0; t.yaw = 0; t.w = 0.45f; t.h = 0.45f; t.z0 = 0; t.z1 = 0.9f;
        const auto [fmax, fmin] = rc::nbv::predicted_fill_axes(t, {0, 0}, *sc.zed, 0.0f);
        CHECK(std::abs(fmax - 0.2f) < 0.03f);
        CHECK(has(imagine_view(c, none, rq, 11, dir).text, std::format("{:.1f} m in front of me", cd.m).c_str()));
        // no envelope: 2 m, said so
        Card ne = chair_card(); ne.subtypes.clear();
        CHECK(not canonical_distance(ne, sc.zed).from_envelope and canonical_distance(ne, sc.zed).m == 2.0f);
        CHECK(has(imagine_view(ne, none, rq, 11, dir).text, "no detector envelope"));
    }
    // the real chair card (fleet envelope, 71 deg camera): the canonical P(notice) is clearly above 0, not the live 0.00
    {
        auto none = scene(false);
        none.room.clear(); none.polygon.clear(); none.doors.clear(); none.objects.clear(); none.robot_xy.reset();
        none.zed->hfov_rad = 2.0f * std::atan(640.0f / 2.0f / 448.0f); none.zed->vfov_rad = 2.0f * std::atan(360.0f / 2.0f / 448.0f);
        Card c = chair_card();
        c.subtypes[""].env = {0.1f, 0.6f, 0.06f};
        ImagineRequest rq; rq.concept_ = "chair"; rq.n = 8;
        (void)imagine_view(c, none, rq, 5, dir);
        const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(slurp(dir / "imagined.json"))).object();
        CHECK(doc["mc"].toObject()["p_notice_mean"].toDouble() > 0.3);
    }
    // ---- hypothetical can_see ----
    {
        const auto s = scene(false);
        const auto t = can_see_hypothetical_reply(chair_card(), s, "chair", "table_1", "door_1");
        CHECK(has(t, "imagined one by table_1") and has(t, "from door_1") and has(t, "±") and has(t, "n=32 imagined chairs") and has(t, "seed 1"));
        CHECK(has(can_see_hypothetical_reply(chair_card("prior"), s, "chair", "table_1", ""), "isn't calibrated yet"));
        CHECK(has(can_see_hypothetical_reply(chair_card(), s, "chair", "the moon", ""), "ASK"));
    }
    std::filesystem::remove_all(dir);
    std::puts(fails == 0 ? "all imagine_view checks passed" : "imagine_view checks FAILED");
    return fails ? 1 : 0;
}
