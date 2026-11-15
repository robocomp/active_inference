#!/usr/bin/env python3
"""furniture_priors_patch.py <obj> — FURNITURE PRIORS for a refrigerator-family clone (owner 2026-11-14):
"furniture objects have to be created complying with some priors, or the priors have to be forced after —
adjacent to a wall, aligned with the room axis, etc."  Both, as model terms:

  BIRTH   the cold-start box is born ON the priors: yaw = the room axis nearest the cloud's direction, back on the
          nearest wall, front facing the room, width/depth measured ALONG/ACROSS that wall (depth from the prior
          until the cloud spans it).
  AFTER   (a) WallFlushPrior: the class probability that the object stands against a wall (was 0 = never assumed);
          (b) the flush weight's gap scale includes the belief's OWN pose uncertainty (J Σ Jᵀ), so a young,
              uncertain object is still drawn to the wall instead of being ignored beyond the 0.15 m reach;
          (c) a ROOM-AXIS alignment prior on yaw, 4-fold (residual ¼·sin 4Δ) so it never decides front/back.
Run from the repo root. Idempotence is not attempted: each replacement must match exactly once.
"""
import pathlib, sys

o = sys.argv[1]; O = o.capitalize(); d = pathlib.Path(f"{o}_concept/src")


def sub(fname, old, new):
    p = d / fname; t = p.read_text(); c = t.count(old)
    if c != 1:
        sys.exit(f"{o}: {fname}: {c}x {old[:80]!r}")
    p.write_text(t.replace(old, new))


# ── belief.h: frame carries the room axis; params carry the axis std ─────────────────────────────────
sub(f"{o}_belief.h", "    WallRef wall;\n",
    "    WallRef wall;\n"
    "    // The room's principal axis (rad, room frame; defined modulo 90°), from the room polygon's edges.\n"
    "    bool  room_axis_ok = false;\n"
    "    float room_axis    = 0.0f;\n")
sub(f"{o}_belief.h", "    float wall_flush_prior        = 0.0f;\n",
    "    float wall_flush_prior        = 0.0f;\n"
    "    // ROOM-AXIS ALIGNMENT (furniture prior, owner 2026-11-14): yaw prefers the room's axes, 4-fold, so it never\n"
    "    // decides front/back. Residual ¼·sin(4Δ) ≈ Δ near alignment ⇒ this is the angle's std (rad). ≤0 = OFF.\n"
    "    float room_axis_std           = 0.0f;\n")

# ── belief.cpp: flush weight with the belief's own gap variance; axis prior in accumulate_extra ─────
sub(f"{o}_belief.cpp", """    const float gap = (back_centre(s) - f.wall.p).dot(f.wall.n);
    const float rel = gap / std::max(1e-3f, params_.wall_reach_m);
    return std::exp(-rel * rel);
}""", """    // ★THE GAP IS UNCERTAIN TOO (2026-11-14). exp(−(gap/reach)²) treats the gap as known exactly, so a young
    // belief whose back face sits 0.5 m off the wall — because its POSE is still uncertain by that much — got
    // weight ~0 and was never drawn to the wall it stands against. The flush hypothesis predicts gap = 0 up to
    // the reach; what it is compared with is the belief's gap, whose variance is J Σ Jᵀ. Adding the two is the
    // marginal likelihood of the flush component: wide while the pose is uncertain, the bare reach once settled.
    const auto gap_of = [&](const {O}BeliefState& st) {{ return (back_centre(st) - f.wall.p).dot(f.wall.n); }};
    const float gap = gap_of(s);
    Eigen::Matrix<float, 6, 1> J;
    const Eigen::Matrix<float, 6, 1> v = s.vec();
    for (int k = 0; k < 6; ++k)
    {{
        Eigen::Matrix<float, 6, 1> vp = v, vm = v;
        vp(k) += params_.fd_eps; vm(k) -= params_.fd_eps;
        J(k) = (gap_of({O}BeliefState::from_vec(vp)) - gap_of({O}BeliefState::from_vec(vm))) / (2.0f * params_.fd_eps);
    }}
    const float var_gap = std::max(0.0f, J.dot(Sigma_ * J));
    const float reach   = std::max(1e-3f, params_.wall_reach_m);
    return std::exp(-gap * gap / (reach * reach + 2.0f * var_gap));
}}""".replace("{O}", O).replace("{{", "{").replace("}}", "}"))
sub(f"{o}_belief.cpp", "    accumulate_wall(s, f, Id, bd);\n",
    "    accumulate_wall(s, f, Id, bd);\n"
    "\n"
    "    // ROOM-AXIS ALIGNMENT (furniture prior): furniture is aligned with the room's axes whether or not it touches\n"
    "    // a wall. ¼·sin(4Δ) vanishes at every multiple of 90°, so the four yaws a box can take on the room grid are\n"
    "    // equally good and the front/back (and width/depth) decision stays with the data and the clearance prior.\n"
    "    if (f.room_axis_ok and params_.room_axis_std > 0.0f)\n"
    "    {\n"
    "        const float dlt = s.yaw - f.room_axis;\n"
    "        const float r   = 0.25f * std::sin(4.0f * dlt);\n"
    "        const float dr  = std::cos(4.0f * dlt);                 // ∂r/∂yaw\n"
    "        const float lam = 1.0f / (params_.room_axis_std * params_.room_axis_std);\n"
    "        Id(5, 5) += lam * dr * dr;\n"
    "        bd(5)    += -lam * dr * r;\n"
    "    }\n")

# ── config ─────────────────────────────────────────────────────────────────────────────────────────
sub(f"{o}_config.h", "    float ai2_prior_depth_m       = -1.0f;",
    "    float ai2_wall_flush_prior    = 0.0f;    // class P(stands against a wall) — furniture prior\n"
    "    float ai2_room_axis_std_deg   = 0.0f;    // room-axis alignment std (deg); <=0 = off\n"
    "    float ai2_prior_depth_m       = -1.0f;")
sub(f"{o}_config.cpp", "    out.ai2_prior_depth_m       = getf(",
    f'    out.ai2_wall_flush_prior    = getf("{O}Model.WallFlushPrior", 0.0f,\n'
    '            "class probability that the object stands against a wall (furniture prior); 0 = never assumed");\n'
    f'    out.ai2_room_axis_std_deg   = getf("{O}Model.RoomAxisStdDeg", 0.0f,\n'
    '            "room-axis alignment std (deg), 4-fold; <=0 = off");\n'
    "    out.ai2_prior_depth_m       = getf(")

# ── fitter: params, per-frame room axis, birth snap ──────────────────────────────────────────────────
sub(f"{o}_fitter.cpp", "        p.prior_depth_m       = cfg_.ai2_prior_depth_m;\n",
    "        p.prior_depth_m       = cfg_.ai2_prior_depth_m;\n"
    "        p.wall_flush_prior    = cfg_.ai2_wall_flush_prior;\n"
    "        p.room_axis_std       = cfg_.ai2_room_axis_std_deg * 0.017453293f;\n")
sub(f"{o}_fitter.cpp", "            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));\n",
    "            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));\n"
    "            if (const auto ax = room_axis(); ax) { frame.room_axis_ok = true; frame.room_axis = *ax; }\n")
sub(f"{o}_fitter.cpp", """                    if (sum > 1e-4f and (mom.ext_major - mom.ext_minor) / sum > 0.10f)
                        s0.yaw = mom.phi;
""", """                    if (sum > 1e-4f and (mom.ext_major - mom.ext_minor) / sum > 0.10f)
                        s0.yaw = mom.phi;
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
""")
sub(f"{o}_fitter.cpp", f"WallRef {O}Fitter::nearest_wall(const Eigen::Vector2f& q) const\n", f'''// The room's principal axis: the length-weighted circular mean of the edge directions on the 4-fold circle
// (4θ), so walls at 0°, 90°, 180°, 270° all vote for the same axis. nullopt without a polygon.
std::optional<float> {O}Fitter::room_axis() const
{{
    const std::size_t n = room_polygon_.size();
    if (not have_room_geometry_ or n < 3) return std::nullopt;
    double sx = 0.0, sy = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {{
        const Eigen::Vector2f e = room_polygon_[(i + 1) % n] - room_polygon_[i];
        const double len = e.norm(), th = std::atan2(e.y(), e.x());
        sx += len * std::cos(4.0 * th); sy += len * std::sin(4.0 * th);
    }}
    if (sx * sx + sy * sy < 1e-12) return std::nullopt;
    return static_cast<float>(0.25 * std::atan2(sy, sx));
}}

// BIRTH ON THE PRIORS. A wall-standing object's first view is its FRONT; the principal-axis yaw of that view is
// noise on a near-square footprint and the depth is unseen. So the birth takes what the priors already know:
//  · the wall: the one nearest the cloud (the class prior says it stands against one — WallFlushPrior);
//  · yaw: back face onto that wall, front into the room (local +Y = back = −n);
//  · width/depth: the cloud's spread ALONG and ACROSS that wall (p5–p95), depth from the prior unless the
//    cloud really spans it (the same band the depth-observability prior uses);
//  · centre: the cloud's middle along the wall, half a depth in from it.
// Not a gate on the fit: the belief starts here and the data then moves it (an object that is genuinely away
// from the wall loses the flush weight as its pose tightens — see flush_weight).
void {O}Fitter::birth_on_priors({O}BeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi,
                                const std::string& name) const
{{
    std::vector<Eigen::Vector2f> xy;
    for (const auto& p : pts) if (p.z() >= z_lo and p.z() <= z_hi) xy.emplace_back(p.x(), p.y());
    if (xy.size() < 10) return;
    Eigen::Vector2f c = Eigen::Vector2f::Zero();
    for (const auto& q : xy) c += q;
    c /= static_cast<float>(xy.size());
    const WallRef w = nearest_wall(c);
    if (not w.ok) return;
    const Eigen::Vector2f n = w.n, t(-n.y(), n.x());          // into the room, and along the wall
    std::vector<float> a, b;
    for (const auto& q : xy) {{ a.push_back((q - w.p).dot(t)); b.push_back((q - w.p).dot(n)); }}
    const auto pct = [](std::vector<float> v, float f)
    {{ const std::size_t k = static_cast<std::size_t>(f * (v.size() - 1)); std::nth_element(v.begin(), v.begin() + k, v.end()); return v[k]; }};
    const float a0 = pct(a, 0.05f), a1 = pct(a, 0.95f), b0 = pct(b, 0.05f), b1 = pct(b, 0.95f);
    const float prior_d = cfg_.ai2_prior_depth_m > 0.0f ? cfg_.ai2_prior_depth_m : cfg_.ai2_prior_footprint_m;
    const float across  = b1 - b0;
    s0.w   = std::max(0.10f, a1 - a0);
    s0.h   = across > cfg_.ai2_depth_obs_band_m ? std::max(across, b1) : prior_d;   // a seen side reaches the wall
    s0.yaw = std::atan2(n.x(), -n.y());                       // R(yaw)·ŷ = −n: the back faces the wall
    const Eigen::Vector2f ctr = w.p + t * (0.5f * (a0 + a1)) + n * (0.5f * s0.h);
    std::print("[{{}}] birth on priors: wall n=({{:.2f}},{{:.2f}}) at {{:.2f}} m from the cloud → yaw={{:.1f}}° "
               "w={{:.2f}} h={{:.2f}} centre ({{:.2f}},{{:.2f}}) [cloud centre ({{:.2f}},{{:.2f}})]\\n",
               name, n.x(), n.y(), 0.5f * (b0 + b1), s0.yaw * 57.29578f, s0.w, s0.h, ctr.x(), ctr.y(), c.x(), c.y());
    s0.cx = ctr.x(); s0.cy = ctr.y();
}}

WallRef {O}Fitter::nearest_wall(const Eigen::Vector2f& q) const
''')
sub(f"{o}_fitter.h", "    bool has_room_polygon() const { return room_polygon_.size() >= 3; }\n",
    "    bool has_room_polygon() const { return room_polygon_.size() >= 3; }\n"
    "    std::optional<float> room_axis() const;   // the room's principal axis (rad, mod 90°)\n"
    f"    void birth_on_priors({O}BeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi,\n"
    "                         const std::string& name) const;\n")
t = (d / f"{o}_fitter.h").read_text()
if "#include <optional>" not in t:
    (d / f"{o}_fitter.h").write_text(t.replace("#pragma once\n", "#pragma once\n#include <optional>\n", 1))
print(o, "furniture priors patched")
