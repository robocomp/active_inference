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
    // ★Are this member's ENDS free to move? A cabinet run's are — level-1 declares t0/t1 free, with
    // no prior at all. An appliance's are NOT: a fridge's width is the object, not an interval to be
    // fitted. So when a run and a fridge share wall, the RUN yields and the fridge does not, and that
    // follows from what each thing is rather than from an arbitration rule.
    bool            ends_free = true;

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
    Hole,       // at least one stops short and neither passes through ⇒ a gap in the surface
    // ★COLLINEAR: two near-parallel members sharing a wall line. Detecting joints by intersecting
    // face lines CANNOT see these — parallel lines do not meet — yet on any single wall EVERYTHING
    // is parallel: cabinets, fridge, dishwasher, oven. So collinear overlap is probably the most
    // common violation in a kitchen and the first version of this file missed all of it.
    // Live case: cabinet_w13_base (3.58 m, yaw 89.01°) running straight through refrigerator_1
    // (yaw 88.36°) — 0.65° apart, so no intersection anywhere near either of them.
    Overlap,    // they share the same stretch of wall — one body inside the other
    Abutting    // collinear and touching end-to-end: the normal way a run continues past an appliance
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
    // Each side is one of three things, and the middle one is what the first version got wrong:
    //   TERMINATES — its end is AT the vertex (|gap| within construction tolerance)
    //   PASSES     — the vertex is meaningfully inside it; it continues past the joint
    //   SHORT      — it stops before reaching the vertex
    // ★"not passing" is NOT the same as "terminating". The first rule asked whether the vertex was
    // deeper inside than a carcass depth, and live a run 0.507 m inside its neighbour fell just under
    // that (depth ~0.53) and was reported as a clean TEE. Two runs half a metre into each other are
    // a crossing however the comparison lands.
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
    // How deep two bodies interpenetrate. For a Crossing the shallower side is what must be removed
    // to separate them; for a collinear Overlap it is the shared stretch of wall (carried in gap_i,
    // negative by the same sign convention as everywhere else).
    float penetration() const
    {
        if (kind == JointKind::Crossing) return std::min(-gap_i, -gap_j);
        if (kind == JointKind::Overlap)  return -gap_i;
        return 0.0f;
    }
};

class KitchenOutline
{
public:
    // Two runs are candidate neighbours when their front-face LINES cross close to an end of each.
    // `reach` is how far past its own end a run may still be considered to meet the other — it is a
    // SEARCH radius for pairing, not a tolerance on the answer: the gap it then reports is the
    // measurement. Set it from the scale of the thing being joined (a carcass depth), not by tuning.
    explicit KitchenOutline(float reach_m = 1.20f) : reach_(reach_m) {}

    // |sin| below which two members count as sharing a wall line rather than meeting at an angle.
    // 2° — a kitchen's runs are installed straight, and the live fridge/cabinet pair sits 0.65°
    // apart. Well clear of a real corner (90°), so nothing in between is being decided by it.
    static constexpr float kParallelSin = 0.035f;

    // How close an end must be to the vertex to count as MEETING it. Construction tolerance for
    // fitted furniture — carcasses are installed flush, so a few centimetres is the whole question.
    // ⚠A genuine number, flagged: "do these two ends meet?" needs some notion of meeting. The raw
    // gaps are always reported beside the verdict, so a reader can second-guess the classification.
    static constexpr float kJoinTolM = 0.05f;

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
                   case JointKind::Hole: return "hole"; case JointKind::Overlap: return "overlap";
                   case JointKind::Abutting: return "abutting"; default: return "corner"; } }

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

    // What each end should be told, to make the shape continuous. Only ends that are free to move
    // and are not already at their vertex produce one.
    struct EndCorrection
    {
        int   seg = -1;
        bool  high_end = false;      // which end of that run (false = face_a side)
        float delta = 0.0f;          // signed move along +u: >0 extend outward, <0 retract
        Eigen::Vector2f target{0, 0};// where the end should be, room frame
        JointKind cause = JointKind::Corner;
        int   other = -1;            // the run it must meet
    };
    std::vector<EndCorrection> end_corrections() const
    {
        std::vector<EndCorrection> out;
        const auto push = [&](const OutlineJoint& j, int self, int other, float gap, bool high)
        {
            if (self < 0 or not segs_[static_cast<std::size_t>(self)].ends_free) return;
            if (std::abs(gap) < kJoinTolM) return;                       // already meets: nothing to say
            const auto& g = segs_[static_cast<std::size_t>(self)];
            EndCorrection c;
            c.seg = self; c.other = other; c.high_end = high; c.cause = j.kind;
            // gap > 0 ⇒ the end stops short and must EXTEND outward; gap < 0 ⇒ it overshoots and must
            // RETRACT. Either way the end moves outward by `gap` along its own outward direction.
            c.delta  = gap;
            const float sign = high ? 1.0f : -1.0f;
            c.target = g.centre + sign * (0.5f * g.length + gap) * g.u;
            out.push_back(c);
        };
        for (const auto& j : joints_)
        {
            // A TEE's through-run is doing the right thing; only the side that should meet is told.
            if (j.kind == JointKind::Tee)
            {
                if (not j.passes_i) push(j, j.i, j.j, j.gap_i, j.end_i_high);
                if (not j.passes_j) push(j, j.j, j.i, j.gap_j, j.end_j_high);
                continue;
            }
            if (j.kind == JointKind::Abutting or j.kind == JointKind::Corner)
                continue;                                                 // already continuous
            // Crossing, Hole and collinear Overlap: every FREE end involved moves to its vertex.
            push(j, j.i, j.j, j.gap_i, j.end_i_high);
            push(j, j.j, j.i, j.gap_j, j.end_j_high);
        }
        return out;
    }

    static bool self_test();

private:
    // Two near-parallel members on the same wall line: project both onto the shared axis and report
    // how much of it they claim in common. This is the case face-line intersection cannot reach.
    // Returns false when they are not parallel, or are parallel but on DIFFERENT lines (facing
    // counters across a galley are parallel and must not be joined — they share no wall).
    bool collinear_joint(std::size_t i, std::size_t j, OutlineJoint& out) const
    {
        const auto& a = segs_[i];
        const auto& b = segs_[j];
        const float sin_ab = std::abs(a.u.x() * b.u.y() - a.u.y() * b.u.x());
        if (sin_ab > kParallelSin)
            return false;                                   // not parallel — the intersection path handles it

        // Same LINE? Compare the perpendicular offset of b's face from a's face line. Two members of
        // one wall run differ by at most a carcass depth; anything more is a different wall.
        const Eigen::Vector2f fa = a.centre + 0.5f * a.depth * a.n;
        const Eigen::Vector2f fb = b.centre + 0.5f * b.depth * b.n;
        const float lateral = std::abs((fb - fa).dot(a.n));
        if (lateral > std::max(a.depth, b.depth))
            return false;                                   // parallel but not the same surface

        // Overlap of the two intervals along the shared axis.
        const float ca = 0.0f,                    ha = 0.5f * a.length;
        const float cb = (b.centre - a.centre).dot(a.u), hb = 0.5f * b.length;
        const float lo = std::max(ca - ha, cb - hb);
        const float hi = std::min(ca + ha, cb + hb);
        const float shared = hi - lo;                       // >0 overlap, <0 a gap between them
        if (shared < -reach_)
            return false;                                   // far apart along the wall — unrelated

        out.i = static_cast<int>(i); out.j = static_cast<int>(j);
        out.vertex = a.centre + (0.5f * (lo + hi)) * a.u + 0.5f * a.depth * a.n;
        out.cross  = sin_ab;
        // Sign convention as everywhere else: negative = bodies share space, positive = a hole.
        out.gap_i = out.gap_j = -shared;
        out.passes_i = out.passes_j = false;
        // ★WHICH end of each run the shared stretch sits at. The first version left these at their
        // default (false = low end), so a correction for an overlap at a run's HIGH end was aimed at
        // the opposite end of the run — it would have lengthened the very run it was trying to shorten.
        const float shared_c = 0.5f * (lo + hi);                  // in a's frame
        const float sgn      = (a.u.dot(b.u) >= 0.0f) ? 1.0f : -1.0f;
        out.end_i_high = shared_c > 0.0f;
        out.end_j_high = (sgn * (shared_c - cb)) > 0.0f;
        out.kind = (shared > 0.01f) ? JointKind::Overlap
                 : (shared > -0.01f) ? JointKind::Abutting
                                     : JointKind::Hole;
        return true;
    }

    void rebuild()
    {
        joints_.clear();
        for (std::size_t i = 0; i < segs_.size(); ++i)
            for (std::size_t j = i + 1; j < segs_.size(); ++j)
            {
                OutlineJoint col;
                if (collinear_joint(i, j, col)) { joints_.push_back(col); continue; }
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
                jt.passes_i = di < -kJoinTolM;
                jt.passes_j = dj < -kJoinTolM;
                const bool short_i = di > kJoinTolM, short_j = dj > kJoinTolM;
                jt.kind = (jt.passes_i and jt.passes_j) ? JointKind::Crossing
                        : (short_i or short_j)          ? JointKind::Hole
                        : (jt.passes_i or jt.passes_j)  ? JointKind::Tee
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

    // ── (g2) BOTH sides well inside ⇒ CROSSING, whatever the carcass depth happens to be ─────────
    // Live: gaps of −0.507 and −0.699 were reported as a clean TEE because 0.507 fell just under the
    // run's own depth (~0.53). Two runs half a metre into each other are a crossing either way.
    {
        OutlineSeg a; a.name = "a"; a.u = {1, 0}; a.n = {0, -1};
        a.depth = 0.60f; a.length = 3.00f; a.centre = Eigen::Vector2f(0.0f, 2.20f);
        OutlineSeg b; b.name = "b"; b.u = {0, 1}; b.n = {1, 0};
        b.depth = 0.60f; b.length = 3.00f; b.centre = Eigen::Vector2f(0.0f, 1.90f);
        // Shorten a so the vertex is 0.50 inside it — just UNDER its 0.60 depth.
        a.length = 2.0f * (0.30f + 0.50f);
        KitchenOutline o; o.set_segments({a, b});
        check(o.joints().size() == 1, "(g2) one joint");
        if (o.joints().size() == 1)
            check(o.joints().front().kind == JointKind::Crossing,
                  "(g2) ★0.50 m inside is a CROSSING even though it is less than the 0.60 m depth");
    }

    // ── (h) COLLINEAR OVERLAP — the case face-line intersection cannot reach ─────────────────────
    // Live: a 3.58 m cabinet run at 89.01° with refrigerator_1 at 88.36° INSIDE it. 0.65° apart, so
    // their face lines meet nowhere near either of them and the first version of this file reported
    // no joint at all. On a single wall everything is parallel, so this is the common violation.
    {
        const auto wall_member = [](const char* nm, float t_centre, float len)
        {
            OutlineSeg g; g.name = nm; g.u = {0, 1}; g.n = {1, 0};
            g.depth = 0.60f; g.length = len;
            g.centre = Eigen::Vector2f(0.0f, t_centre);          // all on the wall line x=0.30
            return g;
        };
        // A 3.58 m run with a 0.66 m fridge standing inside it.
        auto run = wall_member("cabinet_run", 1.79f, 3.58f);      // spans y 0.00 .. 3.58
        auto fri = wall_member("refrigerator", 2.00f, 0.66f);     // spans y 1.67 .. 2.33 — INSIDE
        fri.u = Eigen::Vector2f(std::sin(0.65f * 3.14159f / 180.0f),
                                std::cos(0.65f * 3.14159f / 180.0f)).normalized();   // 0.65° off, as live
        KitchenOutline o; o.set_segments({run, fri});
        check(o.joints().size() == 1, "(h) a run and an appliance on the same wall form ONE joint");
        if (o.joints().size() == 1)
        {
            const auto& j = o.joints().front();
            check(j.kind == JointKind::Overlap,
                  "(h) ★a cabinet run containing a fridge is an OVERLAP — face lines never meet");
            check(std::abs(o.worst_penetration() - 0.66f) < 0.02f,
                  "(h) the shared stretch of wall is the fridge's whole width");
        }
        // Two members ABUTTING end to end on the same wall: the normal way a run continues.
        KitchenOutline o2; o2.set_segments({wall_member("left", 0.0f, 2.00f),
                                            wall_member("right", 2.00f, 2.00f)});
        check(o2.joints().size() == 1 and o2.joints().front().kind == JointKind::Abutting,
              "(h) two members meeting end-to-end on one wall are ABUTTING, not overlapping");
        check(o2.worst_penetration() == 0.0f, "(h) and abutting is not an interpenetration");
        // Facing counters across a galley: parallel, but NOT the same wall line ⇒ still no joint.
        {
            OutlineSeg a = wall_member("north", 1.5f, 3.0f);
            OutlineSeg b = wall_member("south", 1.5f, 3.0f);
            b.centre = Eigen::Vector2f(2.40f, 1.5f); b.n = {-1, 0};   // 2.4 m away, facing back
            KitchenOutline o3; o3.set_segments({a, b});
            check(o3.joints().empty(),
                  "(h) parallel counters on OPPOSITE walls share no line and form no joint");
        }
    }

    // ── (i) END CORRECTIONS — what step 2 would actually say ─────────────────────────────────────
    {
        // The live overlap: a cabinet run sharing 0.554 m of wall with the fridge. The fridge does
        // not move (its width is the object); the run retracts by exactly the shared stretch.
        OutlineSeg run; run.name = "run"; run.u = {0, 1}; run.n = {1, 0};
        run.depth = 0.60f; run.length = 3.58f; run.centre = Eigen::Vector2f(0.0f, 1.79f);
        OutlineSeg fri; fri.name = "fridge"; fri.u = {0, 1}; fri.n = {1, 0};
        // run spans y 0.00 .. 3.58; fridge spans 3.026 .. 3.686 ⇒ they share 0.554, as measured live
        fri.depth = 0.60f; fri.length = 0.66f; fri.centre = Eigen::Vector2f(0.0f, 3.356f);
        fri.ends_free = false;                                    // an appliance's extent is the object
        KitchenOutline o; o.set_segments({run, fri});
        const auto cs = o.end_corrections();
        check(cs.size() == 1, "(i) ★only the RUN is corrected — the fridge's width is not an interval");
        if (cs.size() == 1)
        {
            check(cs.front().seg == 0, "(i) and it is the run, not the appliance");
            check(cs.front().delta < 0.0f, "(i) it RETRACTS (negative), it does not grow");
            check(std::abs(std::abs(cs.front().delta) - 0.554f) < 0.02f,
                  "(i) by exactly the stretch of wall the two of them share");
            check(cs.front().high_end,
                  "(i) ★and at the END the overlap is actually at — aiming at the other end would "
                  "have LENGTHENED the run it is trying to shorten");
        }
        // A clean corner produces no correction at all — nothing to say when it already meets.
        OutlineSeg a; a.name = "a"; a.u = {1, 0}; a.n = {0, -1};
        a.depth = 0.60f; a.length = 2.00f; a.centre = Eigen::Vector2f(1.00f, 0.30f);
        OutlineSeg b; b.name = "b"; b.u = {0, 1}; b.n = {1, 0};
        b.depth = 0.60f; b.length = 2.00f; b.centre = Eigen::Vector2f(-0.30f, 1.00f);
        KitchenOutline o2; o2.set_segments({a, b});
        check(o2.joints().size() == 1 and o2.joints().front().kind == JointKind::Corner,
              "(i) the control is a clean corner");
        check(o2.end_corrections().empty(), "(i) ★a continuous shape is told NOTHING");
    }

    if (ok) std::printf("[kitchen_outline::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
