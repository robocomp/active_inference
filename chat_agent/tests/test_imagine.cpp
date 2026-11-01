// imagined_world.h: opening words, interventions on a copy, the "there" resolver.
#include "../src/imagine_fit.h"
#include "../src/imagine_see.h"
#include "../src/imagine_succeed.h"

#include <cassert>
#include <charconv>
#include <print>

using namespace rc::chat;
using namespace rc::chat::imagine;

namespace
{
constexpr float kPi = std::numbers::pi_v<float>;
bool near(float a, float b, float e = 1e-4f) { return std::abs(a - b) < e; }

ImaginedWorld world()
{
    ImaginedWorld w;
    w.scene.room = "room_1";
    SeeDoor d1; d1.name = "door_1"; d1.xy = {2, 0}; d1.width_m = 0.9f; d1.p_open = 0.3f;
    SeeDoor d2; d2.name = "door_2"; d2.xy = {0, 3}; d2.width_m = 0.9f;
    w.scene.doors = {d1, d2};
    SeeObject t; t.name = "table_1"; t.subtype = "table"; t.x = 1; t.y = 1; t.w = 0.6f; t.d = 0.8f;
    SeeObject n; n.name = "chair_1"; n.subtype = "chair"; n.x = -1; n.y = 1;
    w.scene.objects = {t, n};
    w.scene.robot_xy = Eigen::Vector2f(4, 1);
    w.scene.robot_yaw = 0.2f;
    return w;
}
}

int main()
{
    // parse_opening
    assert(near(*parse_opening("shut")->phi_rad, 0));
    assert(near(*parse_opening(" Closed ")->phi_rad, 0));
    assert(near(*parse_opening("ajar")->phi_rad, kPi / 12));
    const auto half = parse_opening("half");
    assert(near(*half->phi_rad, kPi / 4) and half->words.contains("45"));
    assert(near(*parse_opening("half open")->phi_rad, kPi / 4));
    assert(near(*parse_opening("wide")->phi_rad, kPi / 2));
    assert(near(*parse_opening("fully open")->phi_rad, kPi / 2));
    assert(near(*parse_opening("30")->phi_rad, kPi / 6));
    assert(near(*parse_opening("30 deg")->phi_rad, kPi / 6));
    assert(near(*parse_opening("30°")->phi_rad, kPi / 6));
    assert(near(*parse_opening("120")->phi_rad, kPi / 2));
    assert(near(*parse_opening("-5")->phi_rad, 0));
    assert(parse_opening("-5")->words.contains("taken as 0°") and parse_opening("120")->words.contains("taken as 90°"));
    assert(not parse_opening("as_is")->phi_rad.has_value());
    assert(not parse_opening("")->phi_rad.has_value());
    const auto bad = parse_opening("banana");
    assert(not bad.has_value() and bad.error().starts_with("I don't know how open 'banana' is"));

    // interventions: value semantics
    const ImaginedWorld w0 = world();
    const auto w1 = with_door_phi(w0, "door_1", kPi / 4);
    assert(w1.has_value());
    assert(near(*w1->scene.doors[0].phi_rad, kPi / 4) and not w1->scene.doors[0].p_open.has_value());
    assert(not w1->scene.doors[1].phi_rad.has_value());
    assert(not w0.scene.doors[0].phi_rad.has_value() and w0.scene.doors[0].p_open.has_value() and w0.notes.empty());
    assert(w1->notes.size() == 1);
    const auto nodoor = with_door_phi(w0, "door_9", 0);
    assert(not nodoor.has_value() and nodoor.error() == "I don't know a door called door_9");

    const auto wm = with_object_moved(w0, "table_1", 0.5f, -0.25f);
    assert(wm.has_value() and near(wm->scene.objects[0].x, 1.5f) and near(wm->scene.objects[0].y, 0.75f));
    assert(near(w0.scene.objects[0].x, 1.0f));
    assert(not with_object_moved(w0, "sofa_1", 1, 1).has_value());

    const auto wr = with_robot_at(w0, {1, 2}, 0.5f);
    assert(wr.has_value() and near(wr->scene.robot_xy->x(), 1) and near(wr->scene.robot_yaw, 0.5f));

    // resolve_standpoint
    const auto here = resolve_standpoint(w0, "");
    assert(here.has_value() and near(here->xy.x(), 4) and here->yaw.has_value());
    assert(resolve_standpoint(w0, "room_1").has_value());
    assert(resolve_standpoint(w0, "aquí").has_value() and resolve_standpoint(w0, "aqui").has_value() and resolve_standpoint(w0, "Aquí").has_value());
    assert(canonical_concept("silla") == "chair" and canonical_concept("nevera") == "refrigerator");
    assert(canonical_concept("frigorífico") == "refrigerator" and canonical_concept("Mesa") == "table" and canonical_concept("puerta") == "door");
    assert(norm_name("la nevera") == "nevera" and norm_name("The Door-2") == "door_2");
    ImaginedWorld lost = w0; lost.scene.robot_xy.reset();
    assert(resolve_standpoint(lost, "here").error() == "I don't know where I am");
    const auto dp = resolve_standpoint(w0, "door_2");
    assert(dp.has_value() and near(dp->xy.y(), 3) and not dp->yaw.has_value());
    const auto op = resolve_standpoint(w0, "table_1");
    assert(op.has_value());
    const float r = 0.5f * std::hypot(0.6f, 0.8f) + kRobotRadiusM;
    assert(near((op->xy - Eigen::Vector2f(1, 1)).norm(), r));
    assert(op->xy.x() > 1);                       // the robot's side (robot at x = 4)
    assert(near(std::cos(*op->yaw), -1.0f));      // facing the object, back toward -x
    const auto byclass = resolve_standpoint(w0, "the table");
    assert(byclass.has_value() and near(byclass->xy.x(), op->xy.x()));   // article stripped
    assert(resolve_standpoint(w0, "table").has_value());
    const auto nosize = resolve_standpoint(w0, "chair_1");
    assert(nosize.has_value() and nosize->words.contains("size is unknown"));
    assert(near((nosize->xy - Eigen::Vector2f(-1, 1)).norm(), kRobotRadiusM));
    assert(resolve_standpoint(w0, "door 2").has_value() and resolve_standpoint(w0, "the door_2").has_value());
    assert(with_door_phi(w0, "the Door-2", 0.1f).has_value() and with_object_moved(w0, "the table_1", 1, 0).has_value());
    assert(near(*parse_opening("half-open")->phi_rad, kPi / 4) and near(*parse_opening("half_open")->phi_rad, kPi / 4));
    assert(near(*parse_opening("halfway")->phi_rad, kPi / 4) and parse_opening("half-open")->words.contains("45"));
    assert(near(*parse_opening("medio abierta")->phi_rad, kPi / 4) and near(*parse_opening("cracked")->phi_rad, kPi / 12));
    assert(near(*parse_opening("de par en par")->phi_rad, kPi / 2) and near(*parse_opening("cerrada")->phi_rad, 0));
    assert(near(*parse_opening("fully_open")->phi_rad, kPi / 2) and near(*parse_opening("all the way")->phi_rad, kPi / 2));
    assert(not parse_opening("as_is")->phi_rad.has_value());
    assert(resolve_standpoint(w0, "the moon").error() == "ASK: which place do you mean by 'there'?");

    // ── Q1: through a door ──
    {
        ImaginedWorld f = world();
        f.body_half_width = 0.31f;
        f.robot_sigma_m = 0.02f; f.door_sigma_m["door_1"] = 0.02f;
        const auto leaves = rc::chat::imagine::fit_detail::leaves_at(0.9f, kPi / 4);
        const auto g = rc::anchor_clear_gap(0.9f, &leaves[0]);
        assert(g.width > 0.2f and g.width < 0.3f);
        const auto a45 = answer_fit_door(f, "door_1", *parse_opening("45"));
        assert(a45.p and *a45.p < 0.05f and near(a45.slack_m, g.width - 0.62f, 1e-3f));
        assert(a45.text.contains("P(fit) = ") and a45.text.contains("cm") and a45.text.contains("45") and a45.text.contains("about 7")
               and a45.text.contains("depth across the doorway") and not a45.text.contains("passable"));
        const auto a90 = answer_fit_door(f, "door_1", *parse_opening("wide"));
        assert(a90.p and *a90.p > 0.9f);
        // phi* round trip
        assert(a45.phi_needed_rad.has_value());
        const auto rt = answer_fit_door(f, "door_1", Opening{*a45.phi_needed_rad, "phi*"});
        assert(rt.p and std::abs(*rt.p - 0.5f) < 0.02f);
        // sigma pulls P toward 0.5, from below and from above
        ImaginedWorld big = f; big.door_sigma_m["door_1"] = 0.3f;
        assert(*answer_fit_door(big, "door_1", *parse_opening("45")).p > *a45.p);
        assert(*answer_fit_door(big, "door_1", *parse_opening("wide")).p < *a90.p);
        // no body model, unknown door, unknown opening, exact geometry, never fits
        ImaginedWorld nb = f; nb.body_half_width.reset();
        assert(not answer_fit_door(nb, "door_1", *parse_opening("45")).p and answer_fit_door(nb, "door_1", *parse_opening("45")).text == "I have no model of my body width");
        assert(answer_fit_door(f, "door_7", *parse_opening("45")).text.contains("door_7 is not in this room")
               and answer_fit_door(f, "door_7", *parse_opening("45")).text.contains("only judge doors of the room I am in"));
        assert(answer_fit_door(f, "Door 1", *parse_opening("45")).p.has_value() and answer_fit_door(f, "the door-1", *parse_opening("45")).p.has_value());
        ImaginedWorld noroom = f; noroom.scene.room.clear();
        assert(not answer_fit_door(noroom, "door_1", *parse_opening("45")).p
               and answer_fit_door(noroom, "door_1", *parse_opening("45")).text.starts_with("I do not know which room I am in"));
        assert(a45.text.contains("line up centred on the gap"));
        // as_is with no measured phi: weighted over the door's own states (door_1: P(open) = 0.3)
        const auto unk = answer_fit_door(f, "door_1", *parse_opening("as_is"));
        const auto pshut = *answer_fit_door(f, "door_1", *parse_opening("shut")).p, pwide = *answer_fit_door(f, "door_1", *parse_opening("wide")).p;
        assert(unk.p and near(*unk.p, 0.7f * pshut + 0.3f * pwide, 1e-3f));
        assert(unk.text.contains("door state not measured: weighted over shut/open by P(open)=0.30") and unk.text.contains("shut:") and unk.text.contains("wide open:"));
        ImaginedWorld uni = f; uni.scene.doors[0].p_open.reset();
        const auto un = answer_fit_door(uni, "door_1", *parse_opening("as_is"));
        assert(un.p and *un.p > pshut and *un.p < pwide and un.text.contains("uniform opening"));
        // a measured phi is used as it is, with no mixture
        ImaginedWorld ph = f; ph.scene.doors[0].phi_rad = kPi / 4;
        assert(near(*answer_fit_door(ph, "door_1", *parse_opening("as_is")).p, *a45.p, 1e-3f) and not answer_fit_door(ph, "door_1", *parse_opening("as_is")).text.contains("not measured"));
        // the body note and the interventions' notes are said
        ImaginedWorld bn = f; bn.body_note = "without the controller's safety margin";
        assert(answer_fit_door(bn, "door_1", *parse_opening("45")).text.contains("(without the controller's safety margin)"));
        ImaginedWorld ob = *with_door_phi(f, "door_2", 0.1f);
        assert(answer_fit_door(ob, "door_1", *parse_opening("45")).text.contains("imagined: door_2 opened"));
        ImaginedWorld ex = f; ex.robot_sigma_m.reset(); ex.door_sigma_m.clear();
        assert(answer_fit_door(ex, "door_1", *parse_opening("wide")).text.contains("treating the geometry as exact"));
        ImaginedWorld fat = f; fat.body_half_width = 0.6f;
        assert(answer_fit_door(fat, "door_1", *parse_opening("wide")).text.contains("even fully open I would not fit"));
        ImaginedWorld thin = f; thin.body_half_width = 0.0f;
        assert(answer_fit_door(thin, "door_1", *parse_opening("wide")).text.contains("I fit at any opening"));
    }

    // ── geometry ──
    {
        using namespace rc::chat::imagine::fit_detail;
        const auto A = box_corners(0, 0, 0, 1, 1), B = box_corners(2, 0, 0, 1, 1);
        assert(near(footprint_distance(A, B), 1.0f));
        const auto C = box_corners(2, 2, 0, 1, 1);
        assert(near(footprint_distance(A, C), std::sqrt(2.0f)));                     // corner to corner
        const auto R = box_corners(2, 0, std::numbers::pi_v<float> / 4, 1, 1);                              // diamond, near vertex at x = 2 - 0.7071
        assert(near(footprint_distance(A, R), 2.0f - std::sqrt(0.5f) - 0.5f));
        assert(near(footprint_distance(A, box_corners(0.5f, 0, 0, 1, 1)), 0.0f));     // overlap
        assert(near(footprint_distance(A, box_corners(0, 0, 0, 0.2f, 0.2f)), 0.0f));  // containment
        assert(near(footprint_distance(A, {Eigen::Vector2f(3, -5), Eigen::Vector2f(3, 5)}), 2.5f));   // box-segment
        assert(near(footprint_distance(A, {Eigen::Vector2f(-5, 2), Eigen::Vector2f(5, 2)}), 1.5f));
        assert(near(footprint_distance(A, {Eigen::Vector2f(-5, 0), Eigen::Vector2f(5, 0)}), 0.0f));   // crosses it
    }

    // ── Q1b: between two referents ──
    {
        ImaginedWorld f = world();
        f.body_half_width = 0.31f;
        f.scene.polygon = {{-5, -5}, {5, -5}, {5, 5}, {-5, 5}};
        f.scene.objects.clear();
        SeeObject c; c.name = "chair_1"; c.subtype = "chair"; c.x = 4.25f; c.y = 0; c.w = 0.4f; c.d = 0.4f; c.h = 0.9f; c.z_origin = 0.f;
        // wall at x = 5; the chair's far face at 4.45 ⇒ gap 0.55
        f.scene.objects = {c};
        const auto lo = answer_fit_between(f, "chair_1", "wall");
        assert(lo.p and *lo.p < 0.1f and near(lo.slack_m, 0.55f - 0.62f, 1e-3f) and lo.text.contains("chair_1")
               and lo.text.contains("would have to move") and lo.text.contains("7 cm") and lo.text.contains("live LiDAR"));
        assert(answer_fit_between(f, "the chair", "wall").p.has_value());              // class with one instance
        const auto mv = with_object_moved(f, "chair_1", -0.2f, 0);
        assert(mv.has_value());
        const auto hi = answer_fit_between(*mv, "chair_1", "wall");
        assert(hi.p and *hi.p > *lo.p and near(hi.slack_m, 0.75f - 0.62f, 1e-3f));
        ImaginedWorld wide = f; wide.scene.objects[0].x = 4.0f;                         // far face 4.2 ⇒ gap 0.80
        assert(*answer_fit_between(wide, "chair_1", "wall").p > 0.9f);
        // the displacement needed round-trips to P = 0.5
        const float need = -lo.slack_m;
        ImaginedWorld fs = f; fs.robot_sigma_m = 0.05f;
        const auto mv2 = with_object_moved(fs, "chair_1", -need, 0);
        assert(std::abs(*answer_fit_between(*mv2, "chair_1", "wall").p - 0.5f) < 0.02f);
        // a wall shelf above the band does not block; at base 0 it does
        ImaginedWorld sh = f;
        sh.scene.objects[0].z_origin = 1.4f; sh.scene.objects[0].h = 0.3f;
        const auto up = answer_fit_between(sh, "chair_1", "wall");
        assert(up.p and near(*up.p, 1.0f) and up.text.contains("does not block") and up.text.contains("my body's height is not known"));
        // the band is the body's own height when known: a shelf at 1.4 m blocks a 1.5 m body
        ImaginedWorld tall = sh; tall.robot_height_m = 1.5f;
        assert(*answer_fit_between(tall, "chair_1", "wall").p < 0.1f and not answer_fit_between(tall, "chair_1", "wall").text.contains("height is not known"));
        sh.scene.objects[0].z_origin = 0.f;
        assert(*answer_fit_between(sh, "chair_1", "wall").p < 0.1f);
        // below the floor: ignored too
        sh.scene.objects[0].z_origin = -1.f; sh.scene.objects[0].h = 0.5f;
        assert(near(*answer_fit_between(sh, "chair_1", "wall").p, 1.0f));
        // an ignored referent still reports the footprint clearance
        assert(up.text.contains("55 cm") and near(up.slack_m, 0.55f - 0.62f, 1e-3f));
        // unknown size (no w, no d): no judgement
        ImaginedWorld us = f; us.scene.objects[0].w.reset(); us.scene.objects[0].d.reset(); us.scene.objects[0].h.reset();
        us.scene.objects[0].x = 4.2f; us.robot_sigma_m = 0.05f;
        const auto unk = answer_fit_between(us, "chair_1", "wall");
        assert(not unk.p and unk.text == "I can't judge: chair_1's size is unknown");
        // one size known: judged on what is known, and said
        us.scene.objects[0].w = 0.4f;
        const auto part = answer_fit_between(us, "chair_1", "wall");
        assert(part.p and part.text.contains("only partly known"));
        // an unknown room is said first; the moved chair's note is printed
        ImaginedWorld nr = f; nr.scene.room.clear();
        assert(answer_fit_between(nr, "chair_1", "wall").text.starts_with("I do not know which room I am in"));
        assert(answer_fit_between(*mv, "chair_1", "wall").text.contains("imagined: chair_1 moved by"));
        // one sigma missing is named; unmeasured width is stated
        ImaginedWorld os = f; os.door_sigma_m["door_1"] = 0.02f;
        assert(answer_fit_door(os, "door_1", *parse_opening("wide")).text.contains("my own position uncertainty is unknown"));
        os.scene.doors[0].width_m = 0;
        assert(answer_fit_door(os, "door_1", *parse_opening("wide")).text.contains("its width is not measured; taken as 0.90 m"));
        // errors
        assert(answer_fit_between(f, "wall", "the wall").text == "between two walls is the room, ask me about a door");
        assert(answer_fit_between(f, "sofa", "wall").text == "I don't know an object called sofa");
        // two objects: the larger sigma moves
        SeeObject t; t.name = "table_1"; t.subtype = "table"; t.x = 4.25f - 1.0f; t.y = 0; t.w = 0.4f; t.d = 0.4f; t.h = 0.7f;
        ImaginedWorld two = f; two.scene.objects.push_back(t);
        two.object_sigma_m["chair_1"] = 0.01f; two.object_sigma_m["table_1"] = 0.1f;
        const auto tw = answer_fit_between(two, "chair_1", "table_1");
        assert(tw.p and near(tw.slack_m, 0.6f - 0.62f, 1e-3f) and tw.text.contains("table_1 would have to move"));
        ImaginedWorld nb = f; nb.body_half_width.reset();
        assert(not answer_fit_between(nb, "chair_1", "wall").p);
    }

    // answer_see: can_see's evaluator on the intervened scene
    {
        rc::card::Card c;
        c.concept_ = "refrigerator"; c.level = rc::card::Level::Countable; c.calibration = "fitted";
        c.channels = {{"zed", rc::card::Role::Identity, true}};
        c.geometry = {0.7f, 0.7f, 1.8f, 0.0f, 1.8f};
        rc::card::SubtypeParams sp; sp.env = {0.0f, 1.32f, 0.26f};
        c.subtypes[""] = sp;
        const auto cd = std::optional<rc::card::Card>(c);
        ImaginedWorld sw;
        SeeScene& s = sw.scene;
        s.room = "r"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        rc::nbv::Sensor z; constexpr float f = 448.0f;
        z.hfov_rad = 2.0f * std::atan(640.0f / f); z.vfov_rad = 2.0f * std::atan(360.0f / f); z.height_m = 0.945f;
        s.zed = z; s.zed_mount_z = 0.945f;
        s.robot_xy = Eigen::Vector2f(0.8f, 2.0f);
        SeeObject fr; fr.name = "refrigerator_1"; fr.subtype = "refrigerator"; fr.x = 5.2f; fr.y = 2.0f; fr.w = 0.7f; fr.d = 0.7f; fr.h = 1.8f;
        SeeObject st; st.name = "stool_1"; st.subtype = "stool"; st.x = 1.0f; st.y = 3.0f; st.w = 0.3f; st.d = 0.3f; st.h = 0.5f;
        s.objects = {fr, st};
        const auto txt = [&](const ImaginedWorld& iw, std::string_view from, std::string_view door = {}, const std::optional<Opening>& o = {}) {
            return answer_see(iw, cd, "fridge", from, door, o);
        };
        const auto base = txt(sw, "here");
        std::println("{}", base);
        assert(base.contains("refrigerator_1") and base.contains("P "));
        // one name rule: "the fridge", "la nevera" and "Refrigerator_1" are the same target
        for (const char* nm : {"the fridge", "la nevera", "Refrigerator_1", "the refrigerator-1"})
            assert(answer_see(sw, cd, nm, "here").contains("refrigerator_1"));
        // a box between the standpoint and the fridge is named as the occluder
        ImaginedWorld bw = sw;
        SeeObject bx; bx.name = "box_1"; bx.subtype = "box"; bx.x = 3.0f; bx.y = 2.0f; bx.w = 1.2f; bx.d = 1.2f; bx.h = 2.2f;
        bw.scene.objects.push_back(bx);
        const auto occ = txt(bw, "here");
        std::println("{}", occ);
        assert(occ.contains("hidden behind box_1"));
        // standing against the fridge: too close
        ImaginedWorld cw = sw;
        cw.scene.robot_xy = Eigen::Vector2f(3.0f, 3.5f);
        SeeObject nx; nx.name = "stool_2"; nx.subtype = "stool"; nx.x = 4.3f; nx.y = 2.0f; nx.w = 0.3f; nx.d = 0.3f; nx.h = 0.5f;
        cw.scene.objects.push_back(nx);
        const auto close = txt(cw, "stool_2");
        std::println("{}", close);
        assert(close.contains("too close for the sensor") or close.contains("viewpoint inside"));
        // opening words + the unknown place
        assert(txt(sw, "here", "door_9", *parse_opening("half")) == "I don't know a door called door_9");
        assert(txt(sw, "the sofa") == "ASK: which place do you mean by 'there'?");
        assert(txt(sw, "here", "door_1", *parse_opening("as_is")) == base);
        // a doorway between the standpoint and the fridge: shut < open (the can_see construction: wall x = 3 with a door)
        ImaginedWorld dw;
        dw.scene.room = "r"; dw.scene.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        dw.scene.zed = z; dw.scene.zed_mount_z = 0.945f;
        dw.scene.robot_xy = Eigen::Vector2f(2.0f, 2.0f);
        SeeDoor dr; dr.name = "door_2"; dr.xy = {6, 2}; dr.width_m = 0.9f; dr.phi_rad = 0.0f;
        dw.scene.doors = {dr};
        fr.x = 7.5f; fr.y = 2.0f;
        dw.scene.objects = {fr};
        const auto shut = answer_see(dw, cd, "fridge", "here", "door_2", *parse_opening("shut"));
        const auto wide = answer_see(dw, cd, "fridge", "here", "door_2", *parse_opening("wide"));
        std::println("{}\n{}", shut, wide);
        assert(wide.contains("if door_2 were wide (taken as 90°)") and shut.contains("if door_2 were shut"));
        const auto p_of = [](const std::string& a)
        {
            const auto i = a.find("(P ");
            float v = 0;
            std::from_chars(a.data() + i + 3, a.data() + a.size(), v);
            return v;
        };
        assert(p_of(wide) > p_of(shut));
        assert(answer_see(dw, std::nullopt, "fridge", "here") == "I have no detector for that.");
    }

    // would-succeed (Task 6)
    {
        const auto rec = [](const char* door, const char* outcome, const char* room, const char* after, std::uint64_t dur_ms = 10000)
        {
            rc::em::Record r; r.level = rc::em::Level::Step; r.kind = "cross"; r.target = door; r.t0 = 1000; r.t1 = 1000 + dur_ms;
            r.room = room; r.room_after = after; r.outcome = outcome;
            return r;
        };
        const auto pas = [](const char* id, const char* a, const char* b, const char* door, double al, double be)
        {
            TopoPassageInfo p; p.id = id; p.room_a = a; p.room_b = b; p.live_door = door; p.pass_alpha = al; p.pass_beta = be;
            return p;
        };
        ImaginedWorld sw;
        // room_1 -door_1-> room_2 directly; or room_1 -door_2-> room_3 -door_3-> room_2
        sw.topology.passages = {pas("passage_0", "room_1", "room_2", "door_1", 1, 1), pas("passage_1", "room_1", "room_3", "door_2", 1, 1),
                                pas("passage_2", "room_3", "room_2", "door_3", 9, 1)};
        sw.em_available = true;
        for (int i = 0; i < 15; ++i) sw.records.push_back(rec("door_1", "satisfied", i % 2 ? "room_1" : "room_2", ""));
        for (int i = 0; i < 3; ++i) sw.records.push_back(rec("door_1", "door_wont_open", "room_1", ""));
        for (int i = 0; i < 7; ++i) sw.records.push_back(rec("door_2", "satisfied", "room_1", "", 20000));
        sw.records.push_back(rec("door_2", "cross_unreachable", "room_1", ""));
        const auto m1 = route_odds(sw, {"passage_0"});
        assert(near(float(m1.legs[0].mean), 16.0f / 20.0f) and m1.legs[0].n_success == 15 and m1.legs[0].n_fail == 3);
        assert(not m1.legs[0].used_ltsm_prior and near(float(*m1.legs[0].mean_duration_s), 10.0f) and m1.legs[0].n_duration == 15);
        const auto two = route_odds(sw, {"passage_1", "passage_2"});
        const double m2 = 8.0 / 10.0, e2 = m2 * 9.0 / 11.0 * 0 + (8.0 * 9.0) / (10.0 * 11.0);
        assert(two.legs[1].used_ltsm_prior and near(float(two.legs[1].mean), 0.9f));
        const double p = m2 * 0.9, e2b = 9.0 * 10.0 / (10.0 * 11.0);
        assert(near(float(two.p), float(p)) and near(float(two.sd), float(std::sqrt(e2 * e2b - p * p))));
        assert(near(float(*two.legs[0].mean_duration_s), 20.0f));
        // exclusions change nothing; preempted with a room change is a success
        ImaginedWorld ex = sw;
        for (const char* o : {"stopped_by_user", "superseded", "lost", "banana"}) ex.records.push_back(rec("door_1", o, "room_1", ""));
        ex.records.push_back(rec("door_1", "preempted", "room_1", "room_1"));
        ex.records.push_back(rec("door_1", "preempted", "room_1", ""));
        assert(near(float(route_odds(ex, {"passage_0"}).legs[0].mean), 0.8f) and route_odds(ex, {"passage_0"}).legs[0].n_success == 15);
        ex.records.push_back(rec("door_1", "preempted", "room_1", "room_2"));
        assert(route_odds(ex, {"passage_0"}).legs[0].n_success == 16);
        // no data: LTSM Beta(9,1) vs flat
        ImaginedWorld nd;
        nd.topology.passages = {pas("passage_0", "room_1", "room_2", "door_1", 9, 1), pas("passage_1", "room_1", "room_2", "door_2", 1, 1)};
        assert(near(float(route_odds(nd, {"passage_0"}).legs[0].mean), 0.9f) and route_odds(nd, {"passage_0"}).legs[0].used_ltsm_prior);
        assert(answer_succeed(nd, "room_1", "room_2", "door_2").contains("no data on door_2"));
        // answers
        const auto a = answer_succeed(sw, "room_1", "room_2", "door_2");
        std::println("{}", a);
        assert(a.contains(" vs ") and a.contains("15 of 18") and a.contains("7 of 8"));
        assert(answer_succeed(sw, "room_1", "room_2", "door_9") == "I don't know a door called door_9");
        ImaginedWorld iso = sw; iso.topology.passages.push_back(pas("passage_9", "room_8", "room_9", "door_9", 1, 1));
        assert(answer_succeed(iso, "room_1", "room_2", "door_9") == "no remembered route through door_9 reaches room_2");
        // one name rule across succeed: spellings of rooms and doors
        assert(answer_succeed(sw, "Room 1", "room-2", "the door 2").contains(" vs "));
        assert(route_via(sw.topology, "Room_1", "ROOM 2", "passage_1").has_value() and not route(sw.topology, "room 1", "Room_2").empty());
        assert(resolve_passage(sw.topology, "the door 2") == "passage_1" and resolve_passage(sw.topology, "Door-3") == "passage_2");
        assert(answer_succeed(sw, "", "room_2", "").starts_with("I do not know which room I am in"));
        assert(answer_succeed(sw, "Room_2", "room 2", "").contains("already in"));
        // excluded crossings are counted aloud
        assert(a.contains("not counted") == false);
        const auto exa = answer_succeed(ex, "room_1", "room_2", "");
        assert(exa.contains("stopped by you") and exa.contains("superseded") and exa.contains("not counted"));
        // episodic memory not joined
        ImaginedWorld withem = nd; withem.em_available = true;
        assert(answer_succeed(nd, "room_1", "room_2", "").contains("not connected to my episodic memory"));
        assert(not answer_succeed(withem, "room_1", "room_2", "").contains("not connected"));
        ImaginedWorld notopo; notopo.em_available = true;
        assert(answer_succeed(notopo, "room_1", "room_2", "").contains("no remembered topology"));
        assert(answer_succeed(sw, "room_1", "", "") == "ASK: where should the mission end?");
        assert(answer_succeed(sw, "room_1", "room_1", "").contains("already in room_1"));
        assert(answer_succeed(sw, "room_1", "room_2", "").contains("door_1"));
    }

    std::println("test_imagine: ok");
    return 0;
}
