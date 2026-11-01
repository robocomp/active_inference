// Task 11 (2026-10-29, recog_proto1): the leaf's SWING side flipped on EVERY frame while the robot stood in DOOR_1's doorway
// (door_28, phi 90-95 deg, 5 viewpoints, door_concept.log l.1033-1211: "SWING FLIPPED … along (+2.7)" / "… against (+0.1)"
// alternating for 20 s; ai2_log_phi.csv swing_llr 2.70 / 0.077 on consecutive rows). Two terms of the frame's swing evidence
// depended on WHICH side was believed, so the same rays argued for the other side whichever side that was:
//
//   (a) THE ANCHOR's swing log-odds was tempered by the BELIEVED leaf's spread (anchor_phi_loglik's n_inf over hs = 0 only),
//       so the same rays scored with the labels swapped did not give minus the same number. Replayed from the run's own
//       etc/anchor_frames.bin (tests/data/recog_proto1_anchor_frame.csv, one record): −11.4 with the leaf believed against
//       the normal, +4.1 for the same rays believed along it. A log-likelihood ratio must be odd under relabelling.
//   (b) THE BODY TERM SCORED HYPOTHESES THE FRAME HAD NOT. With the passage anchor active, DoorFit::estimate_phi clears the
//       field's other-swing curves (the anchor's log-odds carries the swing that frame), but then the robot body in the
//       aperture (Task C) — a term COMMON to all four hypotheses — re-created the missing curves as zero + the body term.
//       The believed swing was then scored by LiDAR + anchor + body (a curve normalised to the field's joint maximum, so
//       ≤ 0, and −F when the field preferred the other side) and the other swing by the body alone (≈ 0): a frame's swing
//       evidence was "believed curve vs nothing", negative by construction — anti-MAP on every frame. (The hinge's
//       log-odds drifted 25.2 → 21.5 the same way.) With the anchor's +A for "along" and the field's −F for it (F > A) the
//       believed side lost on every frame, whichever it was.
//
//   (c) A genuinely new set of viewpoints showing the other side still flips the belief (once), and it stays there.
//   (d) The swing / hinge evidence is pooled in an ABSOLUTE frame (log-odds of "along the normal" / "the near jamb"): a MAP
//       flip or an external re-sign (resign_leaf_llrs: belief re-created the other way, log-odds negated with it) re-labels the
//       published number and never touches the pooled evidence; a value handed in from outside (a mirror, a restore) becomes
//       the carried prior.
//
// Build + run (from door_concept/):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore \
//     tests/test_door_swing_stability.cpp src/door_belief.cpp -o /tmp/test_door_swing_stability -lfastdds -lfastcdr -lQt5Core \
//     && /tmp/test_door_swing_stability
// The recorded frame is read relative to the working directory (door_concept/). SWING_TRACE=1 prints (b) and (c) per frame.
#include <algorithm>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <limits>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "door_belief.h"
#include "door_body_evidence.h"
#include "door_leaf_channel.h"
#include "door_anchor_open.h"

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f;
constexpr int   NSTEP = 25;
using Curve = std::vector<std::pair<float, float>>;

static float grid(int k) { return PHI_MIN + (PHI_MAX - PHI_MIN) * float(k) / float(NSTEP - 1); }
// A frame's φ curve peaked at phi0, `offset` nats below the frame's joint maximum (FieldResult's normalisation).
static Curve peaked(float phi0, float offset, float sd = 0.10f)
{
    Curve c;
    for (int k = 0; k < NSTEP; ++k) { const float z = (grid(k) - phi0) / sd; c.emplace_back(grid(k), -0.5f * z * z - offset); }
    return c;
}

static rc::DoorBelief make_belief(float swing)
{
    rc::DoorBeliefParams p;
    p.wall_O = {-2, 0}; p.wall_u = {1, 0}; p.thickness = 0.05f; p.floor_z = 0;
    p.leaf = {0, rc::door::HingeSide::Far, swing};
    return rc::DoorBelief({2.0f, 1.0f, 2.1f}, p);   // the opening is x in [0, 1] on the wall y = 0, centre (0.5, 0)
}

struct Door   // what DoorFitState carries for the leaf channel
{
    rc::door::LeafChannel ch;
    float hinge_llr = 10.0f, swing_llr = 0.0f, phi_est = 1.6f, phi_sigma = 0.05f;
    bool first = true;
};

// ─── (a) the recorded anchor frame ─────────────────────────────────────────────────────────────────────────────────────
struct AnchorFrame
{
    rc::interaction::RelPose post;
    float width = 0.f, thickness = 0.f, p_view = 0.f;
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    rc::door::LeafPrior leaf;
    std::vector<Eigen::Vector3f> rays;
    bool ok = false;
};
static std::vector<float> nums(std::string_view s)
{
    std::vector<float> v;
    while (not s.empty())
    {
        const auto c = s.find(',');
        const auto tok = s.substr(0, c);
        float x = 0.f;
        if (std::from_chars(tok.data(), tok.data() + tok.size(), x).ec == std::errc{}) v.push_back(x);
        if (c == std::string_view::npos) break;
        s.remove_prefix(c + 1);
    }
    return v;
}
static AnchorFrame load_anchor_frame(const char* path)
{
    AnchorFrame f;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() or line[0] == '#') continue;
        const auto c = line.find(',');
        if (c == std::string::npos) continue;
        const std::string_view key(line.data(), c);
        const auto v = nums(std::string_view(line).substr(c + 1));
        if (key == "post" and v.size() == 12)
        {
            f.post.x = Eigen::Vector3f(v[0], v[1], v[2]);
            for (int i = 0; i < 9; ++i) f.post.P(i / 3, i % 3) = v[std::size_t(3 + i)];
        }
        else if (key == "door" and v.size() == 2) { f.width = v[0]; f.thickness = v[1]; }
        else if (key == "origin" and v.size() == 3) f.origin = Eigen::Vector3f(v[0], v[1], v[2]);
        else if (key == "leaf" and v.size() == 7)
        {
            f.leaf.known = true; f.leaf.phi_known = true;
            f.leaf.hinge_u = v[0]; f.leaf.dir_u = v[1]; f.leaf.swing_y = v[2]; f.leaf.phi = v[3]; f.leaf.phi_sigma = v[4];
            f.leaf.phi_max = v[5]; f.leaf.len = v[6];
        }
        else if (key == "p_view" and v.size() == 1) f.p_view = v[0];
        else if (key == "ray" and v.size() == 3) f.rays.emplace_back(v[0], v[1], v[2]);
    }
    f.ok = not f.rays.empty() and f.width > 0.f and f.leaf.known;
    return f;
}

// ─── (b) one frame as DoorFit::estimate_phi assembles it while the passage anchor serves the door ─────────────────────────
// The field scores all four hypotheses (hinge_question_live); its curves are normalised to their joint maximum, which the
// field puts on `field_side` (F nats above the other swing). The anchor frame then CLEARS the field's other-swing curves
// and adds its own (swing-marginalised) curves to the two it keeps; its swing log-odds says `anchor_side` by A nats —
// odd under relabelling, as (a) requires. Last, the body term, common to the four hypotheses.
static rc::door::LeafChannelInput anchor_frame_inputs(float believed, float field_side, float F, float anchor_side, float A,
                                                      float phi0, const Eigen::Vector2f& sensor, float dt)
{
    const float H = 12.0f;   // the hinge is not in question: the other jamb is far below
    const float off_believed = believed == field_side ? 0.0f : F;
    Curve bb = peaked(phi0, off_believed), ob = peaked(phi0, off_believed + H);
    Curve bs = peaked(phi0, F - off_believed), os = peaked(phi0, F - off_believed + H);
    // the anchor block (door_fit.cpp, "THE PASSAGE ANCHOR'S BEAMS"): its curves added, the field's other swing cleared
    const Curve a_this = peaked(phi0, 0.0f, 0.15f), a_other = peaked(phi0, H, 0.15f);
    for (int k = 0; k < NSTEP; ++k) { bb[std::size_t(k)].second += a_this[std::size_t(k)].second; ob[std::size_t(k)].second += a_other[std::size_t(k)].second; }
    bs.clear(); os.clear();
    // the robot body in the aperture (door_fit.cpp, Task C), through the agent's own function
    const auto body = rc::door::body_phi_loglik(0.4f, 1.0f, 0.70f, PHI_MIN, PHI_MAX, NSTEP);
    rc::door::add_common_phi_term({&bb, &ob, &bs, &os}, body, PHI_MIN, PHI_MAX);
    rc::door::LeafChannelInput in;
    in.bb = std::move(bb); in.ob = std::move(ob); in.bs = std::move(bs); in.os = std::move(os);
    in.anchor_swing_delta = (believed == anchor_side ? 1.0 : -1.0) * double(A);
    in.have_sensor = true; in.sensor = sensor; in.dt = dt;
    in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
    return in;
}

// A frame with NO anchor: the field scores all four hypotheses, `field_side` F nats above the other swing.
static rc::door::LeafChannelInput field_frame_inputs(float believed, float field_side, float F, float phi0,
                                                     const Eigen::Vector2f& sensor, float dt)
{
    const float H = 12.0f;
    const float off_believed = believed == field_side ? 0.0f : F;
    rc::door::LeafChannelInput in;
    in.bb = peaked(phi0, off_believed); in.ob = peaked(phi0, off_believed + H);
    in.bs = peaked(phi0, F - off_believed); in.os = peaked(phi0, F - off_believed + H);
    in.have_sensor = true; in.sensor = sensor; in.dt = dt;
    in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
    return in;
}

static bool step(Door& d, rc::DoorBelief& b, rc::door::LeafChannelInput in, const char* tag, int f)
{
    const auto r = rc::door::leaf_channel_step(d.ch, b, d.hinge_llr, d.swing_llr, d.phi_est, d.phi_sigma, in);
    if (std::getenv("SWING_TRACE"))
        std::print("    {} f{:02} swing {:+.0f} llr {:+.3f} (along {:+.3f}) frame {:+.2f} vp {} flip {}\n", tag, f,
                   b.params().leaf.swing, d.swing_llr, d.ch.swing.llr(), r.frame_swing, d.ch.views.views.size(), r.swing_flipped);
    return r.swing_flipped;
}

int main()
{
    std::setlocale(LC_ALL, "");   // as the agent (Qt): the recorded frame must parse under the comma locale too

    // ── (a) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    std::print("(a) the anchor's swing log-odds is ODD under relabelling (recorded frame, recog_proto1 t=1793301183151)\n");
    {
        const auto f = load_anchor_frame("tests/data/recog_proto1_anchor_frame.csv");
        check(f.ok and f.rays.size() == 400, std::format("the recorded frame loads ({} rays)", f.rays.size()));
        const auto g = rc::door::phi_grid(PHI_MAX, PHI_STEP);
        auto as_recorded = f.leaf;
        auto swapped = f.leaf;
        swapped.swing_y = -swapped.swing_y;
        const auto r1 = rc::door::anchor_phi_loglik(f.post, f.width, f.thickness, as_recorded, f.origin, f.rays, f.p_view, g.lo, g.hi, g.n);
        const auto r2 = rc::door::anchor_phi_loglik(f.post, f.width, f.thickness, swapped, f.origin, f.rays, f.p_view, g.lo, g.hi, g.n);
        const double along1 = as_recorded.swing_y * r1.swing_llr, along2 = swapped.swing_y * r2.swing_llr;   // log-odds of anchor +y
        std::print("  believed swing_y {:+.0f}: llr {:+.3f}; believed swing_y {:+.0f}: llr {:+.3f}  ⇒ log-odds of +y: {:+.3f} vs {:+.3f}\n",
                   as_recorded.swing_y, r1.swing_llr, swapped.swing_y, r2.swing_llr, along1, along2);
        check(std::abs(along1 - along2) < 1e-6 * std::max(1.0, std::abs(along1)),
              "the same rays give the same evidence for a side whichever side is believed");
        check(along1 > 0.0, "and it is the side the live frames kept voting for (anchor +y, the leaf along the wall's normal)");
    }

    // ── (b) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    // Parked in the doorway (ONE viewpoint), anchor active, body in the aperture, leaf at 93 deg. The anchor says "along" by
    // 4 nats (the replay's magnitude); the field — whose swing curves the anchor frame discards — preferred "against" by 6.
    std::print("\n(b) recog_proto1: parked in the doorway with the passage anchor — the swing must not alternate\n");
    {
        Door d;
        auto b = make_belief(+1.0f);   // inherited from the other room: along (+1)
        int flips = 0;
        for (int f = 0; f < 40; ++f)
        {
            const float believed = b.params().leaf.swing;
            flips += step(d, b, anchor_frame_inputs(believed, -1.0f, 6.0f, +1.0f, 4.0f, 1.62f, {0.40f, 0.30f},
                                                    f == 0 ? std::numeric_limits<float>::infinity() : 0.1f), "b", f)
                     ? 1 : 0;
        }
        std::print("  40 frames from one viewpoint: {} flips, final swing {:+.0f}, log-odds of along {:+.2f}\n", flips,
                   b.params().leaf.swing, d.ch.swing.llr());
        check(flips == 0, std::format("no swing flip while parked ({} flips; live: one per frame)", flips));
        check(b.params().leaf.swing > 0.0f, "the swing stays on the side the frame's evidence (the anchor) supports");
        check(d.hinge_llr > 9.0f, std::format("the hinge holds its jamb ({:+.2f}, from +10)", d.hinge_llr));
    }

    // ── (c) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    std::print("\n(c) a genuinely contrary run of NEW viewpoints flips the swing once, and it stays\n");
    {
        Door d;
        auto b = make_belief(+1.0f);
        // 20 frames from one viewpoint showing "along" (3 nats per frame), the robot parked 2 m out, frontal.
        int f = 0;
        for (; f < 20; ++f)
            step(d, b, field_frame_inputs(b.params().leaf.swing, +1.0f, 3.0f, 1.62f, {0.5f, 2.0f},
                                          f == 0 ? std::numeric_limits<float>::infinity() : 0.1f), "c", f);
        const float settled = d.ch.swing.llr();
        check(b.params().leaf.swing > 0.0f and settled > 2.0f, std::format("settled along ({:+.2f})", settled));
        // Then the leaf is seen on the other side from a sweep of new viewpoints (10 deg apart around the opening).
        int flips = 0, first_flip_view = -1;
        for (int v = 1; v <= 6; ++v)
        {
            const float a = (90.0f - 10.0f * float(v)) * 3.14159265f / 180.0f;
            const Eigen::Vector2f sensor(0.5f + 2.0f * std::cos(a), 2.0f * std::sin(a));
            for (int k = 0; k < 5; ++k, ++f)
                if (step(d, b, field_frame_inputs(b.params().leaf.swing, -1.0f, 3.0f, 1.62f, sensor, 0.1f), "c", f))
                {
                    ++flips;
                    if (first_flip_view < 0) first_flip_view = v;
                }
        }
        std::print("  after 6 new contrary viewpoints: {} flip(s) (first at viewpoint {}), swing {:+.0f}, log-odds of along {:+.2f}\n",
                   flips, first_flip_view, b.params().leaf.swing, d.ch.swing.llr());
        check(flips == 1 and b.params().leaf.swing < 0.0f, "the belief changes its mind exactly once, to the side the new views show");
        check(first_flip_view >= 1 and first_flip_view <= 2,
              "and as soon as the pooled evidence changes sign (one prior viewpoint ⇒ at the second new one)");
    }

    // ── (d) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    std::print("\n(d) the pooled evidence lives in an absolute frame: relabelling never touches it, a handed-in value replaces it\n");
    {
        Door d;
        auto b = make_belief(+1.0f);
        for (int f = 0; f < 10; ++f)
            step(d, b, field_frame_inputs(b.params().leaf.swing, +1.0f, 3.0f, 1.62f, {0.5f, 2.0f},
                                          f == 0 ? std::numeric_limits<float>::infinity() : 0.1f), "d", f);
        const double along = d.ch.swing.llr();
        const double lam = d.ch.views.views.at(0).lam_swing;
        // resign_leaf_llrs: the belief re-created hanging the leaf the other way, the published log-odds negated with it
        b.set_leaf_swing(-b.params().leaf.swing);
        d.swing_llr = -d.swing_llr;
        auto quiet = field_frame_inputs(b.params().leaf.swing, +1.0f, 0.0f, 1.62f, {0.5f, 2.0f}, 0.1f);
        quiet.bb.clear(); quiet.ob.clear(); quiet.bs.clear(); quiet.os.clear();   // a frame that scored nothing
        step(d, b, quiet, "d", 10);
        std::print("  re-signed from outside: along {:+.4f} → {:+.4f}, view λ {:+.4f} → {:+.4f}, swing {:+.0f} llr {:+.3f}\n",
                   along, d.ch.swing.llr(), lam, d.ch.views.views.at(0).lam_swing, b.params().leaf.swing, d.swing_llr);
        check(std::abs(d.ch.swing.llr() - along) < 1e-3, "a re-sign re-labels the published number, the pooled evidence stands");
        check(std::abs(d.ch.views.views.at(0).lam_swing - lam) < 1e-9, "the viewpoint's evidence is kept (not reset as a hand-over)");
        check(b.params().leaf.swing > 0.0f, "and the MAP returns to the side the evidence supports");
        // a value handed in from outside (a mirror, a restore): it becomes the carried prior
        d.swing_llr = 1.25f;
        step(d, b, quiet, "d", 11);
        const double expect = b.params().leaf.swing * 1.25;
        check(std::abs(d.ch.swing.llr() - expect) < 1e-3 and d.ch.views.views.at(0).lam_swing == 0.0,
              std::format("a handed-in log-odds is the new prior, the ledger restarts (along {:+.3f}, expected {:+.3f})",
                          d.ch.swing.llr(), expect));
    }

    // ── (e) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    // warm11 door_23 (2026-11-01): the swing flipped 10 times while the door was SHUT or nearly (phi 4-48 deg), single frames
    // scoring -27..-75 nats, and settled on the wrong side (the leaf hangs in the other room). The two sides' phi filters had
    // drifted apart (one near 0, one far open), so "the + side at ITS phi vs the - side at ITS phi" compared two ANGLES, not
    // two sides. A shut leaf is the same geometry under both sides: the frame must say ~nothing about the swing.
    std::print("\n(e) a shut door says nothing about the swing, however far apart the two sides' phi filters have drifted\n");
    {
        Door d;
        auto b = make_belief(+1.0f);
        const float H = 12.0f;
        auto frame = [&](float phi_plus, float phi_minus, float dt)
        {
            rc::door::LeafChannelInput in;
            const bool plus_believed = b.params().leaf.swing > 0.0f;
            const float pb = plus_believed ? phi_plus : phi_minus, po = plus_believed ? phi_minus : phi_plus;
            in.bb = peaked(pb, 0.0f); in.ob = peaked(pb, H); in.bs = peaked(po, 0.0f); in.os = peaked(po, H);
            in.have_sensor = true; in.sensor = {0.0f, 2.0f}; in.dt = dt;
            in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
            return rc::door::leaf_channel_step(d.ch, b, d.hinge_llr, d.swing_llr, d.phi_est, d.phi_sigma, in);
        };
        // ambiguous frames: under + the leaf reads 4 deg, under - it reads 57 deg, equally well — the filters part
        for (int f = 0; f < 20; ++f) (void)frame(0.07f, 1.0f, f == 0 ? std::numeric_limits<float>::infinity() : 0.1f);
        const float before = d.ch.swing.llr();
        // now the door is shut: both sides predict the leaf at 4 deg, identically
        const auto r = frame(0.07f, 0.07f, 0.1f);
        std::print("  shut frame: swing evidence {:+.3f} nats (pooled along {:+.3f} -> {:+.3f})\n", r.frame_swing, before,
                   d.ch.swing.llr());
        check(std::abs(r.frame_swing) < 0.5, std::format("identical predictions give ~0 swing evidence ({:+.3f})", r.frame_swing));
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
