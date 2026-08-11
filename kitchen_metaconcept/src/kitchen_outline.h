/*
 * kitchen_outline.h — the kitchen as ONE CONTINUOUS SHAPE, and where its joints ought to be.
 *
 * WHY THIS EXISTS. A run of carcasses presents one continuous front surface. Where one cabinet ends
 * and the next begins has no gap, no edge and no depth step — the seams are not faint in the sensor
 * data, they are ABSENT. So the split of that surface into individual runs cannot be recovered from
 * below at any quality of sensing. It has to be imposed from above, and this is the object that
 * imposes it.
 *
 * Measured consequences of not having it (cabinet_concept, 2026-08-10):
 *   · a run absorbed 30 % of its points from a neighbour, all packed into 28 % of its length at ONE
 *     END, and had to fit a depth of 0.89 m to swallow them;
 *   · the peninsula could not reach its wall — a full carcass depth of that wall was already claimed —
 *     so it took a free-PCA axis, landed where the LiDAR sees straight through it, and was killed four
 *     frames after birth, 21 times;
 *   · a 22 cm hole opened between two things that physically touch.
 *
 * THE MODEL. The outline is a chain of SEGMENTS (one per run: its front face, the surface actually
 * seen) meeting at shared VERTICES. A vertex is one point owned by two segments — so a gap and an
 * overlap are not penalised, they are UNREPRESENTABLE, the same move the (wall,tier) cell table
 * already made one level down to make duplicate runs impossible.
 *
 * WHAT IT CORRECTS, AND WHAT IT LEAVES ALONE. Level-1 declares t0/t1 FREE — the ends carry no prior
 * at all — while depth is already over-determined by the mask data (a standing depth prior was
 * measured to be worth 1.3 %). So this layer speaks about JOINTS and says nothing about surfaces.
 * That also fixes the depths without ever pushing on depth: every fit was faithful to its points, so
 * moving a run's end hands the intruder's points back to the intruder and the depth follows.
 *
 * ★NOT modelled: the seams WITHIN one wall's run. They are unobservable and nothing downstream needs
 * them — one run per wall is the right granularity, and the cell table already provides it.
 *
 * Pure geometry — no DSR, no Qt. Validated by kitchen_outline_self_test().
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace rc {

// One run as the outline sees it: the FRONT FACE it presents to the room.
struct OutlineSeg
{
    std::string     name;
    Eigen::Vector2f centre{0, 0};   // footprint centre, room frame
    Eigen::Vector2f u{1, 0};        // along-run unit direction
    Eigen::Vector2f n{0, 1};        // unit normal pointing INTO the room (the front side)
    float           length = 0.0f;
    float           depth  = 0.0f;

    // The front face: the segment [a, b] the room actually sees.
    Eigen::Vector2f face_a() const { return centre + 0.5f * depth * n - 0.5f * length * u; }
    Eigen::Vector2f face_b() const { return centre + 0.5f * depth * n + 0.5f * length * u; }
    // Along-run coordinate of a point, measured from the face centre (+ towards face_b).
    float           along(const Eigen::Vector2f& p) const { return (p - centre).dot(u); }
};

// What KIND of meeting this is. Reading a joint without it is how a perfectly good T-junction gets
// reported as the worst overlap in the kitchen (measured: a peninsula continuing 0.9 m past a corner
// where the wall run ended within 3 mm).
enum class JointKind
{
    Corner,     // both runs terminate at the vertex — a clean L
    Tee,        // one terminates, the other passes through and continues. Normal, not a fault.
    Crossing,   // BOTH pass through ⇒ their bodies interpenetrate. Physically impossible.
    Hole        // at least one stops short and neither passes through ⇒ a gap in the surface
};

// A joint between two segments: the single point their front faces meet at.
struct OutlineJoint
{
    int   i = -1, j = -1;              // indices into the segment list
    Eigen::Vector2f vertex{0, 0};      // where the two front-face lines meet
    // Signed distance from each run's nearest END to that vertex, along its own axis.
    // POSITIVE = the run stops SHORT of the joint (a hole). NEGATIVE = it runs past it (an overlap).
    float gap_i = 0.0f, gap_j = 0.0f;
    bool  end_i_high = false;          // which end of i the joint is at (false = face_a, true = face_b)
    bool  end_j_high = false;
    float cross = 0.0f;                // |sin| between the two axes; ~1 = a clean right angle
    // A run PASSES THROUGH when the vertex lies deeper inside it than its own carcass depth: it is
    // not meeting anything there, it is continuing. Scale taken from the object, not tuned.
    bool  passes_i = false, passes_j = false;
    JointKind kind = JointKind::Corner;

    // The gap that MEANS something: only sides that actually terminate here. A through-run's
    // "gap" is just how far it continues, and counting it buries the real faults.
    float worst_gap() const
    {
        const float gi = passes_i ? 0.0f : gap_i;
        const float gj = passes_j ? 0.0f : gap_j;
        return std::abs(gi) > std::abs(gj) ? gi : gj;
    }
    // How deep two bodies interpenetrate, for a Crossing; 0 otherwise. The shallower penetration is
    // what must be removed to separate them.
    float penetration() const
    { return kind == JointKind::Crossing ? std::min(-gap_i, -gap_j) : 0.0f; }
};

class KitchenOutline
{
public:
    // Two runs are candidate neighbours when their front-face LINES cross close to an end of each.
    // `reach` is how far past its own end a run may still be considered to meet the other — it is a
    // SEARCH radius for pairing, not a tolerance on the answer: the gap it then reports is the
    // measurement. Set it from the scale of the thing being joined (a carcass depth), not by tuning.
    explicit KitchenOutline(float reach_m = 1.20f) : reach_(reach_m) {}

    void set_segments(std::vector<OutlineSeg> segs) { segs_ = std::move(segs); rebuild(); }
    const std::vector<OutlineSeg>&   segments() const { return segs_; }
    const std::vector<OutlineJoint>& joints()   const { return joints_; }

    // Largest |gap| over all joints — one number for "how discontinuous is this kitchen".
    float worst_gap() const
    {
        float w = 0.0f;
        for (const auto& j : joints_) if (std::abs(j.worst_gap()) > std::abs(w)) w = j.worst_gap();
        return w;
    }
    // Deepest interpenetration across all joints — two carcasses cannot share space, so any value
    // above zero is a geometric impossibility the arrangement must resolve.
    float worst_penetration() const
    {
        float w = 0.0f;
        for (const auto& j : joints_) w = std::max(w, j.penetration());
        return w;
    }
    static const char* kind_name(JointKind k)
    { switch (k) { case JointKind::Tee: return "tee"; case JointKind::Crossing: return "crossing";
                   case JointKind::Hole: return "hole"; default: return "corner"; } }

    // Where two front-face lines cross. nullopt when they are parallel (no joint to speak of).
    static std::optional<Eigen::Vector2f> intersect(const OutlineSeg& a, const OutlineSeg& b)
    {
        const float den = a.u.x() * b.u.y() - a.u.y() * b.u.x();   // = sin(angle between axes)
        if (std::abs(den) < 1e-3f)
            return std::nullopt;
        const Eigen::Vector2f pa = a.centre + 0.5f * a.depth * a.n;   // a point on a's face line
        const Eigen::Vector2f pb = b.centre + 0.5f * b.depth * b.n;
        const Eigen::Vector2f d  = pb - pa;
        const float t = (d.x() * b.u.y() - d.y() * b.u.x()) / den;
        return pa + t * a.u;
    }

    static bool self_test();

private:
    void rebuild()
    {
        joints_.clear();
        for (std::size_t i = 0; i < segs_.size(); ++i)
            for (std::size_t j = i + 1; j < segs_.size(); ++j)
            {
                const auto v = intersect(segs_[i], segs_[j]);
                if (not v.has_value())
                    continue;
                // The crossing must be near an END of BOTH, otherwise these two are not neighbours —
                // two parallel counters across a galley kitchen have a crossing point at infinity's
                // cousin, far off the end of both, and joining them would be nonsense.
                const float ai = segs_[i].along(*v), aj = segs_[j].along(*v);
                const float hi = 0.5f * segs_[i].length, hj = 0.5f * segs_[j].length;
                const float di = std::abs(ai) - hi;      // how far past i's own end the crossing lies
                const float dj = std::abs(aj) - hj;
                if (di > reach_ or dj > reach_)
                    continue;

                OutlineJoint jt;
                jt.i = static_cast<int>(i); jt.j = static_cast<int>(j);
                jt.vertex = *v;
                jt.end_i_high = ai > 0.0f;
                jt.end_j_high = aj > 0.0f;
                // Gap measured along each run's own axis: how far its END is from the shared vertex.
                // Positive ⇒ the run stops short (hole); negative ⇒ it runs past (overlap).
                jt.gap_i = di;
                jt.gap_j = dj;
                jt.cross = std::abs(segs_[i].u.x() * segs_[j].u.y() - segs_[i].u.y() * segs_[j].u.x());
                jt.passes_i = (-di) > segs_[i].depth;
                jt.passes_j = (-dj) > segs_[j].depth;
                jt.kind = (jt.passes_i and jt.passes_j) ? JointKind::Crossing
                        : (jt.passes_i or  jt.passes_j) ? JointKind::Tee
                        : (di > 0.01f or dj > 0.01f)    ? JointKind::Hole
                                                        : JointKind::Corner;
                joints_.push_back(jt);
            }
    }

    float                     reach_ = 1.20f;
    std::vector<OutlineSeg>   segs_;
    std::vector<OutlineJoint> joints_;
};

inline bool kitchen_outline_self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* w)
    { if (not c) { std::printf("[kitchen_outline::self_test] FAIL: %s\n", w); ok = false; } };

    // Build an L: one run along +x with its face pointing -y, one along +y with its face pointing +x.
    // Faces meet at the inner corner. `trim` shortens each run so the hole is a known size.
    const auto make_L = [](float trim_a, float trim_b)
    {
        OutlineSeg a; a.name = "along_x"; a.u = {1, 0}; a.n = {0, -1};
        a.depth = 0.60f; a.length = 2.00f - trim_a;
        a.centre = Eigen::Vector2f(0.5f * a.length + trim_a, 0.30f);   // wall at y=0.6, face at y=0
        OutlineSeg b; b.name = "along_y"; b.u = {0, 1}; b.n = {1, 0};
        b.depth = 0.60f; b.length = 2.00f - trim_b;
        b.centre = Eigen::Vector2f(-0.30f, 0.5f * b.length + trim_b); // wall at x=-0.6, face at x=0
        return std::vector<OutlineSeg>{a, b};
    };

    // ── (a) a CLOSED corner: both runs reach the vertex ⇒ no hole ────────────────────────────────
    {
        KitchenOutline o; o.set_segments(make_L(0.0f, 0.0f));
        check(o.joints().size() == 1, "(a) the two runs of an L form exactly one joint");
        if (o.joints().size() == 1)
        {
            const auto& j = o.joints().front();
            check((j.vertex - Eigen::Vector2f(0, 0)).norm() < 1e-3f,
                  "(a) the shared vertex is the inner corner where the two FACES meet");
            check(std::abs(j.gap_i) < 1e-3f and std::abs(j.gap_j) < 1e-3f, "(a) a closed corner has no gap");
            check(j.cross > 0.99f, "(a) the two axes are at a right angle");
        }
    }

    // ── (b) a HOLE: one run stops short ⇒ positive gap, of exactly the amount ────────────────────
    {
        KitchenOutline o; o.set_segments(make_L(0.22f, 0.0f));
        check(o.joints().size() == 1, "(b) still one joint");
        if (o.joints().size() == 1)
        {
            const auto& j = o.joints().front();
            check(std::abs(j.gap_i - 0.22f) < 1e-3f,
                  "(b) ★a run stopping 22 cm short reports a +0.22 m hole — the live case");
            check(std::abs(j.gap_j) < 1e-3f, "(b) its neighbour, which does reach, reports none");
            check(std::abs(o.worst_gap() - 0.22f) < 1e-3f, "(b) worst_gap surfaces it");
        }
    }

    // ── (c) an OVERLAP: one run runs PAST the vertex ⇒ negative gap ──────────────────────────────
    {
        KitchenOutline o; o.set_segments(make_L(-0.15f, 0.0f));
        check(o.joints().size() == 1, "(c) still one joint");
        if (o.joints().size() == 1)
            check(o.joints().front().gap_i < -0.14f,
                  "(c) a run past the vertex reports a NEGATIVE gap (it is inside its neighbour)");
    }

    // ── (d) two PARALLEL runs across a galley are NOT joined ─────────────────────────────────────
    {
        OutlineSeg a; a.name = "north"; a.u = {1, 0}; a.n = {0, -1};
        a.depth = 0.60f; a.length = 3.0f; a.centre = Eigen::Vector2f(1.5f, 2.70f);
        OutlineSeg b = a; b.name = "south"; b.n = {0, 1}; b.centre = Eigen::Vector2f(1.5f, 0.30f);
        KitchenOutline o; o.set_segments({a, b});
        check(o.joints().empty(), "(d) facing counters across a galley form NO joint (parallel)");
    }

    // ── (e) a PENINSULA meeting a wall run's SIDE, which is the apartment's actual layout ────────
    // The peninsula's near end abuts the wall run's FACE — never the wall, which is a full carcass
    // depth behind it. That is why a 25 cm wall-proximity test can never fire for a real peninsula.
    {
        OutlineSeg w; w.name = "wall_run"; w.u = {1, 0}; w.n = {0, -1};
        w.depth = 0.60f; w.length = 3.00f; w.centre = Eigen::Vector2f(0.0f, 2.20f);   // face at y=1.90
        OutlineSeg p; p.name = "peninsula"; p.u = {0, 1}; p.n = {1, 0};
        p.depth = 0.60f; p.length = 1.50f;
        p.centre = Eigen::Vector2f(1.0f, 1.90f - 0.22f - 0.75f);   // stops 22 cm short of that face
        KitchenOutline o; o.set_segments({w, p});
        check(o.joints().size() == 1, "(e) the peninsula joins the run it abuts");
        if (o.joints().size() == 1)
        {
            const auto& j = o.joints().front();
            check(std::abs(j.gap_j - 0.22f) < 1e-2f,
                  "(e) ★the peninsula reports the 22 cm hole to the run's FACE, not to the wall");
            const float to_wall = 0.22f + 0.60f;
            check(std::abs(j.gap_j - to_wall) > 0.5f,
                  "(e) ...and that is a full carcass depth nearer than the wall itself");
        }
    }

    // ── (f) The three situations the LIVE kitchen actually produced ─────────────────────────────
    // Classifying them is the difference between one meaningless "worst gap" and three usable facts.
    {
        // A TEE: the wall run ends at the vertex, the peninsula continues 0.9 m past it. NOT a fault.
        OutlineSeg w; w.name = "wall_run"; w.u = {1, 0}; w.n = {0, -1};
        w.depth = 0.60f; w.length = 2.00f;
        w.centre = Eigen::Vector2f(-0.40f, 2.20f);          // face y=1.90, and it ENDS at x=0.60
        OutlineSeg p; p.name = "peninsula"; p.u = {0, 1}; p.n = {1, 0};
        p.depth = 0.60f; p.length = 2.00f;
        p.centre = Eigen::Vector2f(0.30f, 1.90f);           // face x=0.60; the vertex is 1.0 INSIDE it
        KitchenOutline o; o.set_segments({w, p});
        check(o.joints().size() == 1, "(f) tee: one joint");
        if (o.joints().size() == 1)
        {
            const auto& j = o.joints().front();
            check(j.kind == JointKind::Tee, "(f) ★a run continuing past a corner is a TEE, not an overlap");
            check(j.passes_j and not j.passes_i, "(f) the peninsula passes through; the wall run terminates");
            check(std::abs(o.worst_gap()) < 0.05f,
                  "(f) ★worst_gap IGNORES the through-run — the live line called this a 0.9 m overlap");
            check(o.worst_penetration() == 0.0f, "(f) a tee is not an interpenetration");
        }
    }
    {
        // A CROSSING: two perpendicular runs whose bodies genuinely intersect — impossible geometry.
        OutlineSeg a; a.name = "run_a"; a.u = {1, 0}; a.n = {0, -1};
        a.depth = 0.60f; a.length = 3.00f; a.centre = Eigen::Vector2f(0.0f, 2.20f);
        OutlineSeg b; b.name = "run_b"; b.u = {0, 1}; b.n = {1, 0};
        b.depth = 0.60f; b.length = 3.00f; b.centre = Eigen::Vector2f(0.0f, 1.90f);
        KitchenOutline o; o.set_segments({a, b});
        check(o.joints().size() == 1, "(g) crossing: one joint");
        if (o.joints().size() == 1)
        {
            check(o.joints().front().kind == JointKind::Crossing,
                  "(g) ★two runs each passing through the vertex CROSS — bodies interpenetrate");
            check(o.worst_penetration() > 0.5f, "(g) and the penetration depth is reported");
        }
    }

    if (ok) std::printf("[kitchen_outline::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
