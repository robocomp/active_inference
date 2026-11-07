// door_visibility.cpp — see door_visibility.h. The sample classification is compute_silhouette_existence's, moved.
#include "door_visibility.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include "ricoh_door_birth.h"                                     // ricoh_looked
#include "../door_geometry.h"
#include "../door_phi_posterior.h"                                // phi_field_nuisance, door_wall_edges (one own-wall definition)
#include "../door_room_band.h"
#include "../../../common/occlusion/occlusion.h"                 // rc::occlusion::{cone_blocks, walls_block} (SHARED)

namespace rc::door::core
{
MaskCells build_mask_cells(const MaskPacket& pkt, MaskScope scope)
{
    MaskCells c;
    if (not pkt.valid)
        return c;
    for (const auto& sl : pkt.slices)
    {
        if (scope == MaskScope::RicohOnly and sl.source == 0)
            continue;
        const std::size_t b = std::min<std::size_t>(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min<std::size_t>(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "door") ? c.door : c.occluder;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(MaskCells::key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }
    return c;
}

std::vector<std::size_t> own_wall_edges(const rc::DoorFitState& inst, const std::vector<Eigen::Vector2f>& polygon)
{
    const auto nz = rc::door::phi_field_nuisance(inst.aperture);
    return rc::door::door_wall_edges(inst.aperture, polygon, nz.depth_lo, nz.depth_hi);
}

ViewRig::ViewRig(const CameraView& v, const ExistenceScene& sc, const rc::DoorFitState& inst, const MaskCells& cells)
    : v_(v), sc_(sc), inst_(inst), cells_(cells),
      cam_T_room_(v.room_T_cam.inverse()),
      O_(static_cast<float>(v.room_T_cam.coeff(0, 3)), static_cast<float>(v.room_T_cam.coeff(1, 3)),
         static_cast<float>(v.room_T_cam.coeff(2, 3))),
      Oxy_(O_.x(), O_.y()),
      own_edges_(own_wall_edges(inst, sc.polygon)),   // Task D-C: every wall but the door's own occludes
      occ_margin_(std::max(0.0f, sc.cfg.exist_occlusion_margin_m)),
      has_poly_(sc.polygon.size() >= 3),
      W_(static_cast<float>(v.cam.width)), H_(static_cast<float>(v.cam.height))
{
    // ── THE LEAF BEYOND THE APERTURE (door_room_band.h): a sample past the room-side face is seen THROUGH the hole, a view the
    // silhouette model does not represent; each sample carries weight 1 − P(it lies past that face), measured from the
    // aperture line. See the long comment this was moved from (git: door_existence.cpp, compute_silhouette_existence).
    const float delta_ap = has_poly_
                         ? rc::door::signed_boundary_distance(sc.polygon, inst.aperture.centre_xy())
                         : std::numeric_limits<float>::quiet_NaN();
    band_known_ = std::isfinite(delta_ap);
    if (band_known_)
        n_out_ = rc::door::outward_normal(sc.polygon, inst.aperture.centre_xy(), inst.aperture.across_u(), sc.band);
    r0_ = rc::door::face_offset_from_aperture(delta_ap);
    windows_.assign(v.looked_windows.begin(), v.looked_windows.end());
}

SampleView ViewRig::at(const door::LeafPose& L, float lx, float lz) const
{
    SampleView sv;
    const Eigen::Vector3f Ps = door::leaf_point(L, lx, 0.0f, lz);   // face at mid-thickness
    const Eigen::Vector4d Pr(Ps.x(), Ps.y(), Ps.z(), 1.0);
    const Eigen::Vector4d Pc = cam_T_room_ * Pr;
    sv.X = Pc.x(); sv.Y = Pc.y(); sv.Z = Pc.z();
    const bool pinhole = v_.cam.model == Projection::Pinhole;
    // behind / at the near clip: the pinhole's forward axis, a 360 camera's range
    if (pinhole ? sv.Y <= 0.20 : Eigen::Vector3d(sv.X, sv.Y, sv.Z).norm() <= 0.20)
        return sv;
    const auto uv = project(v_.cam, Eigen::Vector3d(sv.X, sv.Y, sv.Z));
    if (not uv) return sv;
    sv.col = static_cast<float>(uv->x()); sv.row = static_cast<float>(uv->y());
    if (sv.col < 0.f or sv.col >= W_ or sv.row < 0.f or sv.row >= H_) return sv;   // out of the REAL frustum ⇒ not detectable
    const Eigen::Vector3f Pw(static_cast<float>(Pr.x()), static_cast<float>(Pr.y()), static_cast<float>(Pr.z()));
    sv.Pw = Pw;
    // A 360 detector processes ONE strip per frame: a bearing outside every looked window was not looked at (≠ not seen);
    // windows unknown ⇒ nothing looked at (Review Focus 3).
    if (not pinhole)
    {
        if (not v_.windows_known or not rc::door::core::ricoh_looked(windows_, Oxy_, Eigen::Vector2f(Pw.x(), Pw.y())))
            return sv;
    }
    sv.key = MaskCells::key(sv.col, sv.row);
    if (cells_.occluder.contains(sv.key) and not cells_.door.contains(sv.key))
    { sv.fate = SampleFate::Occluded; return sv; }                  // a nearer MASKED object hides it
    // GEOMETRIC occluders, which carry no YOLO mask: (a) a room WALL crossing this sightline (own wall skipped), (b) another
    // door instance's panel standing in front of this sample. Per SAMPLE, not per instance.
    if (rc::occlusion::walls_block(Oxy_, {Pw.x(), Pw.y()}, sc_.polygon, own_edges_))
    { sv.fate = SampleFate::Occluded; return sv; }
    {
        Eigen::Vector3f dc = Pw - O_;
        const float rc_len = dc.norm();
        if (rc_len > 1e-3f)
        {
            dc /= rc_len;
            for (const auto& [jid, jp] : sc_.doors)
            {
                const rc::DoorFitState& jinst = *jp;
                if (jid == inst_.node_id or not jinst.ai2_initialized) continue;
                const auto& js = jinst.model.state();
                const Eigen::Vector3f Cj(js.cx, js.cy, js.cz + 0.5f * js.h);
                if (rc::occlusion::cone_blocks(O_, dc, rc_len, Cj, 0.5f * std::max(js.w, js.thickness),
                                               (Cj - O_).norm(), occ_margin_))
                { sv.fate = SampleFate::Occluded; return sv; }
            }
        }
    }
    float w_room = 1.0f;
    if (band_known_)
    {
        const float a = (Eigen::Vector2f(Pw.x(), Pw.y()) - inst_.aperture.plane_O()).dot(n_out_);   // from the LEAF's line (5b)
        w_room = 1.0f - rc::door::p_beyond_far_face(a, r0_, sc_.band.sigma_m);
    }
    sv.w_room = w_room;
    sv.fate = (w_room > 0.0f) ? SampleFate::Visible : SampleFate::Occluded;   // wholly through the opening: not this view
    return sv;
}

ViewGeometry door_view(const CameraView& v, const ExistenceScene& sc, const rc::DoorFitState& inst)
{
    ViewGeometry g;
    if (not inst.ai2_initialized or v.cam.width == 0 or v.cam.height == 0)
        return g;
    const MaskCells cells = build_mask_cells(sc.pkt, v.cam.model == Projection::Pinhole ? MaskScope::All : MaskScope::RicohOnly);
    const ViewRig rig(v, sc, inst, cells);
    const auto& L0 = inst.leaf_pose;
    const auto& s = inst.model.state();
    const float hw = L0.half_w;

    // The leaf-face centre must have a pixel for anything to be projected at all.
    const Eigen::Vector3f Pc3 = door::leaf_point(L0, 0.0f, 0.0f, L0.centre_z);
    const Eigen::Vector4d cc = rig.cam_T_room() * Eigen::Vector4d(Pc3.x(), Pc3.y(), Pc3.z(), 1.0);
    if (not project(v.cam, cc.head<3>()) or (v.cam.model == Projection::Pinhole and cc.y() <= 0.20))
        return g;
    g.projectable = true;
    g.range_m = static_cast<float>(cc.head<3>().norm());
    {   // |cos| between the horizontal view ray and the aperture plane's normal
        const Eigen::Vector2f r = Eigen::Vector2f(Pc3.x(), Pc3.y()) - rig.camera_position().head<2>();
        g.aspect = (r.norm() < 1e-6f) ? 1.0f : std::abs(r.normalized().dot(inst.aperture.across_u()));
    }

    // vis: one pass per opening angle; the modal pass also yields the pixel samples.
    const auto sweep = [&](const door::LeafPose& L, std::vector<cv::Point2f>* px)
    {
        double det_w = 0.0;
        for (int ix = 0; ix < kLeafGridNX; ++ix)
            for (int iz = 0; iz < kLeafGridNZ; ++iz)
            {
                const auto sv = rig.at(L, (-1.0f + 2.0f * (ix + 0.5f) / kLeafGridNX) * hw,
                                       s.cz + s.h * (iz + 0.5f) / kLeafGridNZ);
                if (sv.fate != SampleFate::Visible) continue;
                det_w += sv.w_room;
                if (px) px->emplace_back(sv.col, sv.row);
            }
        return det_w / static_cast<double>(kLeafGridNX * kLeafGridNZ);
    };
    const double modal = sweep(L0, &g.samples);
    double w_sum = 0.0;
    for (const auto& [phi, sup] : inst.phi_curve)
        w_sum += std::max(0.0f, sup);
    if (inst.phi_curve.size() > 1 and w_sum > 0.0)
    {
        double e = 0.0;
        for (const auto& [phi, sup] : inst.phi_curve)
        {
            const double w = std::max(0.0f, sup) / w_sum;
            if (w <= 0.0) continue;
            door::LeafState ls = inst.leaf;
            ls.phi = phi;
            e += w * sweep(door::leaf_pose(inst.aperture, ls), nullptr);
        }
        g.vis = static_cast<float>(e);
    }
    else
        g.vis = static_cast<float>(modal);

    // fill: the projected extent of the leaf face's four corners over the image extent (a 360 camera: over one detector strip).
    // A corner with no pixel (behind a pinhole) falls back to the visible samples' extent.
    const float half_h = L0.half_h, cz = L0.centre_z;
    std::vector<cv::Point2f> pts;
    bool all = true;
    for (const auto& [lx, lz] : {std::pair{-hw, cz - half_h}, {hw, cz - half_h}, {hw, cz + half_h}, {-hw, cz + half_h}})
    {
        const Eigen::Vector3f P = door::leaf_point(L0, lx, 0.0f, lz);
        const Eigen::Vector4d c = rig.cam_T_room() * Eigen::Vector4d(P.x(), P.y(), P.z(), 1.0);
        const auto uv = (v.cam.model == Projection::Pinhole and c.y() <= 0.20) ? std::nullopt : project(v.cam, c.head<3>());
        if (not uv) { all = false; break; }
        pts.emplace_back(static_cast<float>(uv->x()), static_cast<float>(uv->y()));
    }
    if (not all)
        pts = g.samples;
    if (not pts.empty())
    {
        const bool wrap = v.cam.model != Projection::Pinhole;
        const float W = static_cast<float>(v.cam.width), H = static_cast<float>(v.cam.height);
        float c_lo = 0.f, c_hi = 0.f, r_lo = pts[0].y, r_hi = pts[0].y;
        for (const auto& p : pts)
        {
            float dc = p.x - pts[0].x;
            if (wrap) dc -= W * std::round(dc / W);                   // unwrap across the seam, relative to the first point
            c_lo = std::min(c_lo, dc); c_hi = std::max(c_hi, dc);
            r_lo = std::min(r_lo, p.y); r_hi = std::max(r_hi, p.y);
        }
        const float strips = wrap ? static_cast<float>(std::max(1, v.strips)) : 1.0f;
        g.fill_h = (c_hi - c_lo) / W * strips;
        g.fill_v = (r_hi - r_lo) / H;
    }
    return g;
}
}  // namespace rc::door::core
