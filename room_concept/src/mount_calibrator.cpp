#include "mount_calibrator.h"

#include "room_viewer.h"

#include <dsr/api/dsr_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDateTime>
#include <QDebug>
#include <QString>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>

namespace rc
{
void MountCalibrator::push_mount_correction(rc::CameraIngestor &ing, const Eigen::Vector4d &applied,
                                           const std::string &cam, const char *why)
{
    // Prior-sigma units -> radians and metres, on the camera's own axes (see set_mount_correction).
    const float pitch  = static_cast<float>(applied(0)) * params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    const float height = static_cast<float>(applied(1)) * params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    const float yaw    = static_cast<float>(applied(2)) * params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    if (std::abs(pitch) + std::abs(height) + std::abs(yaw) <= 0.f) return;
    ing.set_mount_correction(pitch, height, yaw);
    qInfo().nospace().noquote()
        << "[camcal] " << QString::fromStdString(cam) << " mount correction " << why << ": pitch "
        << QString::number(pitch * 180.0 / M_PI, 'f', 4) << " deg, height "
        << QString::number(height, 'f', 4) << " m, yaw "
        << QString::number(yaw * 180.0 / M_PI, 'f', 4) << " deg  (total against the graph extrinsic)";
}

/// Feed the pooled solve back into the camera mount.
///
/// ★ THERE IS NO GATE HERE, AND THAT IS THE DESIGN. The increment applied is the POSTERIOR MEAN,
///   which is already shrunk toward the prior by exactly how much the data informs the axis: the
///   zed's yaw is 98% data and applies nearly all of itself, the ricoh's is 45% and applies 45%,
///   and an axis the data says nothing about applies nothing. A threshold on "informed enough"
///   would be a second, cruder copy of a judgement the posterior has already made.
/// ★ Accum::apply_correction moves the evidence AND the prior's anchor together, so the total is
///   always the posterior mean of the error relative to the graph extrinsic and cannot ratchet.
/// ⚠ It REFUSES unless the solve was marginalised. An estimate taken with the per-vertex nuisance
///   off has absorbed each corner's own detection bias, and writing that into an extrinsic would
///   put a detector's error into the robot's geometry — silently, and with a tightening sigma to
///   go with it. That refusal, not a magnitude limit, is what makes this safe to leave running.
void MountCalibrator::apply_mount_solve(rc::camcal::Estimator &pool, rc::CameraIngestor &ing,
                                       const rc::mount::Accum::Solution &sol, const std::string &cam)
{
    if (not params.IMAGE_EDGE_MOUNT_APPLY or not sol.ok) return;
    if (not sol.marginalised)
    {
        if (not mount_apply_refused_logged_)
        {
            mount_apply_refused_logged_ = true;
            qWarning() << "[camcal] mountApply is on but the solve is NOT marginalised — set"
                       << "ImageEdge.mountVertexOffsetSigmaPx > 0, or delete evidence that carries"
                       << "no per-vertex partials. Refusing to write an unmarginalised estimate"
                       << "into the extrinsic.";
        }
        return;
    }
    // ⚠ THE SIGN. `Solution::p` is `-x`, and every place that reports it negates it again, so the
    //   correction that cancels the reported error is `-p` and not `p`. Applying `+p` runs the loop
    //   away linearly instead of converging (measured: -53 degrees in 12 cycles on a 1 degree truth).
    Eigen::Vector4d dp = -sol.p;
    dp(3) = 0.0;                       // the dt column is not a mount parameter and has no observation
    if (not dp.allFinite() or dp.isZero()) return;
    pool.apply_correction(dp);         // evidence first: the ingestor must never lead the evidence
    push_mount_correction(ing, pool.applied(), cam, "applied");
    // The shared extrinsic LAST, and only ever mirroring what the local correction already is.
    publish_mount_to_graph(pool, ing, cam);
}

/// Decide what the ingestor's base must be: the nominal the evidence was measured against, or the
/// extrinsic just read from the graph.
///
/// ★★★ THIS IS WHAT MAKES PUBLISHING AND RESUMING BOTH CORRECT. `applied` is a total relative to a
///     nominal. Binding takes the base from the graph, which is right only until something writes
///     that edge — and mountPublish writes exactly it. Without this the next restart binds to
///     `nominal ⊕ applied`, pushes `applied` again, and the correction lands TWICE with the prior
///     anchored at the doubled total: one correction per restart, which is the ratchet.
/// ★ The stored nominal WINS over the graph, and that is the point. It is also checked rather than
///   trusted: the graph should read either the nominal (nothing published, or robot_concept reseeded
///   the robot's JSON over it) or `nominal ⊕ applied` (our publish still standing). Anything else came
///   from OUTSIDE this loop.
/// ⚠ An external nominal is ADOPTED and the correction it absorbs is dropped; our own published value
///   never is. Re-centring the prior on the estimator's own last answer removes the prior's pull, and
///   publishes happen every window — that asymmetry is the whole reason this function is not two
///   lines.
void MountCalibrator::reconcile_mount_nominal(rc::camcal::Estimator &pool, rc::CameraIngestor &ing,
                                             const std::string &cam)
{
    const Eigen::Matrix3f graph_R = ing.base_R();       // as just bound, i.e. what the graph says
    const Eigen::Vector3f graph_t = ing.base_t();
    const Eigen::Vector4d ap = pool.applied();
    const bool has_corr = ap.head<3>().cwiseAbs().maxCoeff() > 1e-12;

    if (not pool.have_base())
    {
        // First run on this camera, or a pre-format-3 file. The graph is the only nominal there is.
        if (has_corr and params.IMAGE_EDGE_MOUNT_PUBLISH)
        {
            // ⚠ AMBIGUOUS, AND IT MUST NOT BE GUESSED. The file carries a correction, publishing is
            //   on, and nothing records whether the graph already contains it. Applying it again
            //   doubles the mount; not applying it discards a measurement. So the correction is
            //   dropped and the graph adopted: the evidence (H, b) is KEPT, only its anchor moves,
            //   and the next save records the nominal so this cannot recur.
            pool.adopt_external_nominal(graph_R, graph_t, params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_YAW_SIGMA);
            qWarning().noquote() << QString::asprintf(
                "[camcal] %s: evidence predates the stored nominal (format < 3) and carries a"
                " correction of pitch %+.4f / height %+.4f / yaw %+.4f prior sigmas while mountPublish"
                " is ON. Nothing records whether the graph already holds it, so the correction is"
                " DROPPED and the current extrinsic adopted as the nominal. H and b are kept; the"
                " total re-converges from here and the nominal is saved from now on.",
                cam.c_str(), ap(0), ap(1), ap(2));
        }
        else
            pool.set_base(graph_R, graph_t);
        return;
    }

    // What the graph would read under each hypothesis.
    const Eigen::Matrix3f nom_R = pool.base_R();
    const Eigen::Vector3f nom_t = pool.base_t();
    const float d_nominal = (graph_R - nom_R).cwiseAbs().maxCoeff()
                          + (graph_t - nom_t).cwiseAbs().maxCoeff();
    // `nominal ⊕ applied`, built the way the ingestor builds it (rebuild_extrinsic_).
    const float pitch = static_cast<float>(ap(0)) * params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    const float height = static_cast<float>(ap(1)) * params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    const float yaw = static_cast<float>(ap(2)) * params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    const Eigen::Matrix3f Rc = (Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitX())
                              * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())).toRotationMatrix();
    const Eigen::Matrix3f pub_R = Rc * nom_R;
    const Eigen::Vector3f pub_t = Rc * nom_t + height * Eigen::Vector3f::UnitZ();
    const float d_published = (graph_R - pub_R).cwiseAbs().maxCoeff()
                            + (graph_t - pub_t).cwiseAbs().maxCoeff();

    constexpr float kTol = 1e-4f;      // 1e-4 rad is 0.006 deg, far below any correction here
    if (d_nominal < kTol or d_published < kTol)
    {
        // Either the graph still holds the nominal or it holds our own published total. Both mean the
        // stored nominal is the mount this evidence describes, so it becomes the base and the
        // correction is pushed on top exactly as it was measured.
        ing.set_base(nom_R, nom_t);
        qInfo().noquote() << QString::asprintf(
            "[camcal] %s: base taken from the EVIDENCE's nominal, not the graph (graph %s;"
            " |graph-nominal| %.2e, |graph-published| %.2e). This is what stops the resumed"
            " correction being applied twice.", cam.c_str(),
            d_published < kTol ? "still holds our published total" : "holds the nominal",
            static_cast<double>(d_nominal), static_cast<double>(d_published));
        return;
    }
    // Neither: the extrinsic changed from outside this loop — the robot's JSON reseeded with a mount
    // that now carries a measurement, or someone edited it. Adopt it, and drop the correction it
    // absorbs rather than stacking ours on top of theirs.
    Eigen::Vector3f delta = Eigen::Vector3f::Zero();
    float res = 0.f;
    const bool rebased = pool.adopt_external_nominal(
        graph_R, graph_t, params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
        params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA, params.IMAGE_EDGE_MOUNT_YAW_SIGMA, &delta, &res);
    qWarning().noquote() << QString::asprintf(
        "[camcal] %s: the extrinsic changed OUTSIDE this loop (|graph-nominal| %.2e,"
        " |graph-published| %.2e, both over %.0e). Adopting it as the new nominal. The change is"
        " pitch %+.4f deg / height %+.4f m / yaw %+.4f deg with a family residual of %.2e, so the"
        " evidence was %s.",
        cam.c_str(), static_cast<double>(d_nominal), static_cast<double>(d_published),
        static_cast<double>(kTol), delta.x() * 180.0 / M_PI, delta.y(), delta.z() * 180.0 / M_PI,
        static_cast<double>(res),
        rebased ? "RE-ANCHORED onto it — b is untouched because the MOUNT did not move, only the point it is measured from, and no information is lost"
                : "RESET: the change is not expressible as a mount correction, so evidence measured"
                  " against the old mount is not evidence about this one");
}

/// Mirror the measured mount into the SHARED body->camera RT edge, so retina, the controller and
/// every other consumer read the corrected extrinsic instead of the nominal one. Until this existed
/// the loop corrected only room_concept's own copy, which was deliberate containment while the loop
/// was young and never the end state.
///
/// ★★★ THIS IS AN OUTPUT AND NOTHING BUT AN OUTPUT, AND THAT IS THE WHOLE SAFETY ARGUMENT. The
///     estimator keeps measuring against the extrinsic it read at bind: CameraIngestor freezes that
///     base, Accum's prior stays anchored on it, and neither is re-read after a write. So publishing
///     cannot move a single number the loop produces — it only tells the fleet what the loop already
///     decided. Re-reading the edge after writing it would make the graph both the input and the
///     output of one estimator and re-centre the prior on its own last answer, which is exactly the
///     ratchet the feedback path was designed to avoid (a weakly informed axis then walks away one
///     honest step at a time).
/// ⚠⚠ ACROSS A RESTART IT STOPS BEING ONLY AN OUTPUT, and that is the open end of this feature. A
///     new process binds to whatever the edge now holds, so today's published value becomes the next
///     session's prior anchor and its nominal. Within a session the loop is stationary; across
///     sessions it can random-walk by its own posterior sigma, because nothing outside the loop
///     still holds the mount's independently measured value. THE CURE IS NOT HERE: the nominal
///     belongs in the robot's JSON — the description of the physical mount — so that the prior is
///     anchored on a measurement the loop never writes. See ImageEdge.mountPublish in etc/config.toml
///     for the persistence step that closes it, which is NOT BUILT.
/// ★ Composes from the NOMINAL captured on the first write, never from the edge's current value: the
///   edge is always `nominal x correction`, so a second window cannot compound the first one's write.
/// ★ Refuses an unmarginalised solve for the same reason apply_mount_solve does, and refuses if
///   composing the edge does not reproduce cortex's own chain — a convention this file believed
///   rather than checked would land a pitch on a roll and still look plausible.
void MountCalibrator::publish_mount_to_graph(rc::camcal::Estimator &pool,
                                            const rc::CameraIngestor &ing, const std::string &cam)
{
    if (not params.IMAGE_EDGE_MOUNT_PUBLISH) return;
    const Eigen::Vector3f corr = ing.mount_correction();          // pitch rad, height m, yaw rad
    if (not corr.allFinite()) return;
    MountPublishState &st = mount_publish_[cam];
    // Nothing moved since the last write. An RT write is a graph signal to every peer, so a window
    // that changed the mount by less than a microradian must not cost one.
    if (st.have_last and (corr - st.last).cwiseAbs().maxCoeff() < 1e-6f) return;

    auto cn = G->get_node(cam);
    if (not cn.has_value()) return;                               // ALWAYS check (CLAUDE.md)
    const auto pid = G->get_attrib_by_name<parent_att>(cn.value());
    if (not pid.has_value()) return;
    auto pn = G->get_node(pid.value());
    if (not pn.has_value()) return;
    const auto e = G->get_edge(pn->id(), cn->id(), "RT");
    if (not e.has_value()) return;
    const auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
    const auto tr  = G->get_attrib_by_name<rt_translation_att>(e.value());
    if (not rot.has_value() or not tr.has_value()
        or rot.value().get().size() < 3 or tr.value().get().size() < 3) return;

    // Rx * Ry * Rz, the composition the whole fleet uses for rt_rotation_euler_xyz.
    const auto euler_to_R = [](const Eigen::Vector3f &r) -> Eigen::Matrix3f
    {
        return (Eigen::AngleAxisf(r.x(), Eigen::Vector3f::UnitX())
              * Eigen::AngleAxisf(r.y(), Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(r.z(), Eigen::Vector3f::UnitZ())).toRotationMatrix();
    };

    if (not st.have_nominal and pool.have_edge_nominal())
    {
        // ★ FROM THE EVIDENCE, not from the edge. After a room_concept restart the edge holds LAST
        //   session's published total, so capturing it here would compose this session's correction
        //   on top of it and compound the publish once per restart. The nominal travels with the
        //   evidence for the same reason the base does.
        st.nominal_t = pool.edge_nominal_t();
        st.nominal_r = pool.edge_nominal_r();
        st.have_nominal = true;
        qInfo().noquote() << QString::asprintf(
            "[camcal] %s: publish nominal restored from the evidence, t [%+.4f %+.4f %+.4f]"
            " euler [%+.5f %+.5f %+.5f] — the edge's current value is NOT used as a base.",
            cam.c_str(), st.nominal_t.x(), st.nominal_t.y(), st.nominal_t.z(),
            st.nominal_r.x(), st.nominal_r.y(), st.nominal_r.z());
    }
    if (not st.have_nominal)
    {
        st.nominal_t = Eigen::Vector3f(tr.value().get()[0],  tr.value().get()[1],  tr.value().get()[2]);
        st.nominal_r = Eigen::Vector3f(rot.value().get()[0], rot.value().get()[1], rot.value().get()[2]);
        // Persist it with the evidence so the next session composes from the same place.
        pool.set_edge_nominal(st.nominal_t, st.nominal_r);
        st.have_nominal = true;
    }
    // ★ THE CONVENTION CHECK RUNS ONCE PER CAMERA AND INDEPENDENTLY OF WHERE THE NOMINAL CAME FROM.
    //   It used to sit inside the capture branch, so a nominal restored from the evidence skipped it
    //   — i.e. it would only ever run on a camera's first-ever session, which is the one run where a
    //   convention error is least likely to have been introduced since.
    if (not st.checked)
    {
        st.checked = true;
        // ── The convention, checked against cortex's own chain instead of asserted ───────────────
        // ★ MAIN THREAD ONLY (ts == 0 touches InnerEigenAPI's unlocked cache — CLAUDE.md). Both call
        //   sites are in compute()'s pump; if that ever changes, skip the check rather than corrupt
        //   the cache, and say the check was skipped rather than let silence imply it passed.
        if (QThread::currentThread() == QCoreApplication::instance()->thread())
        {
            if (auto inner = G->get_inner_eigen_api())
            {
                const auto chain = inner->get_transformation_matrix(
                    cam, pn->name(), 0, "RT", DSR::RT_API::TimeQuery::Nearest);
                if (chain.has_value())                            // ALWAYS check the optional
                {
                    Eigen::Matrix4f parent_T_cam = Eigen::Matrix4f::Identity();
                    parent_T_cam.block<3, 3>(0, 0) = euler_to_R(st.nominal_r);
                    parent_T_cam.block<3, 1>(0, 3) = st.nominal_t;
                    const Eigen::Matrix4f prod =
                        chain.value().matrix().cast<float>() * parent_T_cam;
                    const float err = (prod - Eigen::Matrix4f::Identity()).cwiseAbs().maxCoeff();
                    if (err > 1e-3f)
                    {
                        if (not mount_publish_refused_logged_)
                        {
                            mount_publish_refused_logged_ = true;
                            qWarning().noquote() << QString::asprintf(
                                "[camcal] mountPublish REFUSES on %s: composing the body->camera edge "
                                "as Rx*Ry*Rz does not reproduce cortex's own chain (max residual "
                                "%.2e). A mount written under the wrong euler convention puts a pitch "
                                "on a roll and looks plausible. Fix the composition, do not relax "
                                "this.", cam.c_str(), static_cast<double>(err));
                        }
                        return;
                    }
                }
            }
        }
        else
            qInfo() << "[camcal] mountPublish: euler-convention check SKIPPED for"
                    << QString::fromStdString(cam) << "— not on the main thread, so the ts==0 chain"
                    << "query is unsafe here (CLAUDE.md). The write below is unverified.";
    }

    // ── The correction, as the ingestor applies it ───────────────────────────────────────────────
    // rebuild_extrinsic_(), verbatim: p_cam' = Rx(pitch)*Rz(yaw) * p_cam + height * z_cam. So the
    // camera<-parent transform is premultiplied by A, and parent->camera is postmultiplied by A^-1.
    // The correction lives entirely in the CAMERA frame, so it localises to this one link whatever
    // sits above it in the tree — which is why nothing here needs to know about Shadow->body.
    Eigen::Matrix4f A = Eigen::Matrix4f::Identity();
    A.block<3, 3>(0, 0) = (Eigen::AngleAxisf(corr.x(), Eigen::Vector3f::UnitX())
                         * Eigen::AngleAxisf(corr.z(), Eigen::Vector3f::UnitZ())).toRotationMatrix();
    A.block<3, 1>(0, 3) = corr.y() * Eigen::Vector3f::UnitZ();
    Eigen::Matrix4f T_nom = Eigen::Matrix4f::Identity();
    T_nom.block<3, 3>(0, 0) = euler_to_R(st.nominal_r);
    T_nom.block<3, 1>(0, 3) = st.nominal_t;
    const Eigen::Matrix4f T_new = T_nom * A.inverse();
    const Eigen::Vector3f t_new = T_new.block<3, 1>(0, 3);
    const Eigen::Matrix3f R_new = T_new.block<3, 3>(0, 0);
    // ★ NOT Eigen::eulerAngles(0,1,2). It constrains the MIDDLE angle to [0,pi], and both of this
    //   robot's cameras have a nominal Y of about -2e-5 rad, so it returns the other valid
    //   decomposition — roughly (pi, +y, pi). That composes to the same rotation and would pass any
    //   round-trip check, while writing a triple in which r[2] is no longer readable as a yaw. Every
    //   consumer that reads a component by index — robot_concept's boresight write does — would then
    //   be reading nonsense from a mathematically correct edge. Closed form instead, valid for
    //   |pitch| < pi/2, which a camera mount is:
    //     R = Rx(a)Ry(b)Rz(c)  =>  b = asin(R02), a = atan2(-R12, R22), c = atan2(-R01, R00)
    const Eigen::Vector3f r_new(std::atan2(-R_new(1, 2), R_new(2, 2)),
                                std::asin(std::clamp(R_new(0, 2), -1.f, 1.f)),
                                std::atan2(-R_new(0, 1), R_new(0, 0)));
    // Round-trip the extraction: a write whose own decomposition does not reproduce the matrix is a
    // different mount, not a rounding. (Measured on both cameras' real values: 4.7e-07.)
    if ((euler_to_R(r_new) - R_new).cwiseAbs().maxCoeff() > 1e-5f)
    {
        qWarning() << "[camcal] mountPublish: euler decomposition did not round-trip for"
                   << QString::fromStdString(cam) << "— refusing to write.";
        return;
    }
    if (not t_new.allFinite() or not r_new.allFinite()) return;

    if (auto rt = G->get_rt_api())
    {
        // ★ STATIC, NOT TIMESTAMPED. robot_concept's reasoning applies unchanged: this is a fixed
        //   physical mount whose ESTIMATE is being refined, so the newest value is the best answer
        //   for every frame including older ones — and a timestamped write would stamp the blocks
        //   with the moment of calibration, putting every later query through body->camera outside
        //   the ring for ever and reporting a clamp it did not earn.
        rt->insert_or_assign_edge_RT_static(pn.value(), cn->id(),
            std::vector<float>{t_new.x(), t_new.y(), t_new.z()},
            std::vector<float>{r_new.x(), r_new.y(), r_new.z()});
        st.last = corr;
        st.have_last = true;
        qInfo().noquote() << QString::asprintf(
            "[camcal] %s mount PUBLISHED into %s->%s: pitch %+.4f deg, height %+.4f m, yaw %+.4f deg "
            "on top of the nominal — t [%+.4f %+.4f %+.4f] euler [%+.5f %+.5f %+.5f]. Every agent "
            "reads this now.", cam.c_str(), pn->name().c_str(), cam.c_str(),
            corr.x() * 180.0 / M_PI, corr.y(), corr.z() * 180.0 / M_PI,
            t_new.x(), t_new.y(), t_new.z(), r_new.x(), r_new.y(), r_new.z());
    }
}

void MountCalibrator::open_pair_log(std::ofstream &csv, const std::string &cam,
                                   const rc::CameraIngestor &ing)
{
    // ★ KEYED BY CAMERA, like the evidence file beside it (camera_calib_<robot>_<camera>.txt).
    //   A fixed filename let a second camera's run destroy the first one's rows; the estimator's
    //   own evidence was already keyed, only this diagnostic was not.
    csv.open("etc/image_edge_pair_" + cam + ".csv", std::ios::out | std::ios::trunc);
    if (not csv.is_open()) return;
    csv.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
    // ★ FULL float precision, for the reason camera_calibration.h::save already gives about the
    //   evidence file — and this file needed it MORE, not less. The solve weights by cov^-1, and a
    //   near-singular 2x2 amplifies input error by its condition number. MEASURED on the 09-03 tour
    //   at the default 6 significant figures: typical cond 17 (harmless), but 259 of 91152 rows above
    //   1e6 and 9 rows that came back NOT positive-definite once rounded — those were dropped by a
    //   replay though they counted live, leaving H off by 6% on the yaw-height cross term while b and
    //   rTr agreed to 1e-2. 9 digits round-trips a float exactly, so the delta=0 replay can be exact
    //   rather than approximately right.
    csv << std::setprecision(std::numeric_limits<float>::max_digits10);
    csv << "ts_ms,camera,vertex,"
        // WHICH corner of the vertical edge: the loop closure keys on vertex*2 + ceiling, so a
        // replay without this column would difference a floor corner against a ceiling one and
        // recompute a closure the agent never measured.
           "ceiling,"
           "u_img,v_img,u_lidar,v_lidar,ru,rv,"
        // sigu/sigv are the DIAGONAL of the pair covariance; cuv is its off-diagonal. All three,
        // because the solve weights by the full 2x2 inverse: a replay handed only the diagonal
        // would compute a different H from the same rows, and then a bug in the replay could not be
        // told apart from that difference. With cuv the zero-injection replay must reproduce the
        // live solve exactly, which is the only self-check the replay has.
           "sigu,sigv,cuv,assoc_prob,range_m,angle_deg,assoc_chi2,"
        // ── the association's INPUTS, beside its verdict ──────────────────────────────────────
        // assoc_chi2 is TRUNCATED to [0, CornerDetector::Params::assoc_chi2] by the gate itself,
        // so its distribution cannot be used to judge the gate. These two can: n_rivals is how
        // many model corners were in gate for this detection (0 = no choice to get wrong), and
        // runnerup_chi2 is how far away the best loser sat. The MARGIN runnerup_chi2/assoc_chi2 is
        // the correspondence's real confidence — a match is trustworthy when the second-best
        // candidate is FAR, not when the best one is CLOSE.
           "n_rivals,runnerup_chi2,"
        // The LiDAR corner in the ROBOT frame — what uv_lidar was computed FROM. With it (and the
        // sidecar's mount) an extrinsic injection is a replay of this file rather than another
        // 200 m of driving (VALIDATION_THREE_DEVICE_CORNERS §2b, arm 7).
           "px_robot,py_robot,pz_robot,"
        // The self-calibration correction in force WHEN THIS ROW WAS WRITTEN. With mountApply on the
        // mount moves during a run, so the sidecar's single extrinsic describes only the moment the
        // file opened; a replay that recomputed uv_lidar from it would be reconstructing a mount the
        // later rows were never measured against. Per row, because that is where the truth is.
           "corr_pitch,corr_height,corr_yaw,"
        // The LIDAR HALF of the covariance, on its own. The other half (the image corner's) is the
        // remainder. A replay needs the split because the LiDAR half is the part that moves with the
        // mount: with it the injected weighting is rebuilt exactly instead of held fixed and
        // apologised for.
           "cl_uu,cl_uv,cl_vv\n";

    rc::mount::ReplayContext rc_ctx;
    rc_ctx.robot        = params.LIDAR_ROBOT_FRAME;
    rc_ctx.camera       = cam;
    rc_ctx.cam          = ing.model();
    rc_ctx.cam_R_robot  = ing.cam_R_robot();
    rc_ctx.cam_t_robot  = ing.cam_t_robot();
    rc_ctx.sigma_pitch  = params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    rc_ctx.sigma_height = params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    rc_ctx.sigma_yaw    = params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    rc_ctx.offset_sigma_px = params.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX;
    rc_ctx.applied         = ing.mount_correction();   // cam_R_robot above already includes it
    // The LiDAR's origin in the robot frame: a LiDAR mount error rotates every corner about THAT
    // point, not about the robot origin, and the parallax that leaves is precisely what decides
    // whether a LiDAR injection really cancels in the camera-vs-camera closure. If the chain does
    // not resolve, the flag stays false and the replay REFUSES that leg rather than assuming zero.
    if (auto inner = G->get_inner_eigen_api())
        if (const auto T = inner->get_transformation_matrix(params.LIDAR_ROBOT_FRAME,
                                                            params.LIDAR_HELIOS_NAME, 0, "RT",
                                                            DSR::RT_API::TimeQuery::Nearest);
            T.has_value())   // ALWAYS check the optional (CLAUDE.md)
        {
            rc_ctx.lidar_t_robot = T.value().matrix().block<3, 1>(0, 3).cast<float>();
            rc_ctx.lidar_known   = true;
        }
    const std::string side = "etc/image_edge_replay_" + cam + ".txt";
    if (not rc::mount::write_replay_context(side, rc_ctx))
        qWarning() << "[camcal] could not write" << QString::fromStdString(side)
                   << "— the pair rows will not be replayable offline";
    else if (not rc_ctx.lidar_known)
        qWarning() << "[camcal]" << QString::fromStdString(params.LIDAR_HELIOS_NAME) << "<-"
                   << QString::fromStdString(params.LIDAR_ROBOT_FRAME)
                   << "did not resolve; the LiDAR-injection leg of arm 7 will refuse to run";
}

void MountCalibrator::write_pair_row(std::ofstream &csv, const std::string &cam, std::int64_t ts,
                                    const rc::mount::PairObs &pr, bool ceiling, float angle_deg,
                                    float assoc_chi2, int n_rivals, float runnerup_chi2,
                                    const Eigen::Vector3f &corr)
{
    if (not csv.is_open()) return;
    csv << ts << ',' << cam << ',' << pr.vertex << ',' << (ceiling ? 1 : 0) << ','
        << pr.uv_image.x() << ',' << pr.uv_image.y() << ','
        << pr.uv_lidar.x() << ',' << pr.uv_lidar.y() << ','
        << pr.r.x() << ',' << pr.r.y() << ','
        << std::sqrt(std::max(0.f, pr.cov(0, 0))) << ','
        << std::sqrt(std::max(0.f, pr.cov(1, 1))) << ','
        << pr.cov(0, 1) << ','
        << pr.assoc_prob << ',' << pr.range_m << ','
        << angle_deg << ',' << assoc_chi2 << ','
        << n_rivals << ','
        // A match with no rival has an INFINITE margin, not a huge finite one. Writing the 1e9
        // sentinel would put a number into an average that means "no rival".
        << (n_rivals > 0 ? runnerup_chi2 : -1.f) << ','
        << pr.p_robot.x() << ',' << pr.p_robot.y() << ',' << pr.p_robot.z() << ','
        << corr.x() << ',' << corr.y() << ',' << corr.z() << ','
        << pr.cov_lidar(0, 0) << ',' << pr.cov_lidar(0, 1) << ',' << pr.cov_lidar(1, 1) << '\n';
}

void MountCalibrator::mount_pair_update(const rc::ImageEdgeObs &obs,
                                       const std::vector<rc::CornerDetector::CornerMatch> &matches,
                                       std::int64_t timestamp_ms)
{
    // The driving camera. Auxiliary channels call feed_calib_pairs() with their own name and model.

    if (obs.triple_points.empty() or matches.empty() or not driving_) return;
    if (mp_win_start_ms_ == 0) mp_win_start_ms_ = timestamp_ms;
    if (not mp_loaded_)
    {
        // Resume the measurement from the last session. Loading H and b is not a ratchet: the prior
        // is re-added at solve time, so this is the same evidence arriving earlier.
        mp_loaded_ = true;
        // ★ ONE FILE PER CAMERA. A single etc/camera_calib.txt accumulated 523844 pairs across a
        //   zed -> ricoh switch on 2026-08-29 before this was noticed: two different mounts summed
        //   into one information matrix, which estimates neither. The name keys the file AND is
        //   checked inside it, so a copy or a rename is caught too.
        mp_pool_.set_camera(params.IMAGE_EDGE_CAMERA, params.LIDAR_ROBOT_FRAME);
        // The per-vertex offset nuisance, on the pool AND on the per-window accumulator, so the two
        // columns on the same log line are produced by the same model. Set BEFORE load(), which
        // preserves it. 0 = off = the pre-2026-09-02 solve exactly.
        const double vox = params.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX;
        mp_pool_.set_vertex_offset_sigma_px(vox);
        mp_win_.offset_sigma_px = vox;
        if (vox > 0.0)
            qInfo().nospace() << "[camcal] per-vertex offset nuisance ON, prior sigma " << vox
                              << " px — mount sigmas are now cluster-honest and will read LARGER; "
                                 "yaw approaches the between-vertex SEM by construction";
        const std::string path = mp_pool_.path();
        if (const std::size_t k = mp_pool_.load(path); k > 0)
            qInfo().nospace() << "[camcal] resumed from " << QString::fromStdString(path)
                              << " (" << k << " pairs, camera "
                              << QString::fromStdString(params.IMAGE_EDGE_CAMERA) << ")";
        // A correction restored from disk must reach the mount BEFORE the first frame is measured
        // against it, or this session's first window is referenced to an extrinsic the evidence
        // does not describe.
        reconcile_mount_nominal(mp_pool_, *driving_, params.IMAGE_EDGE_CAMERA);
        push_mount_correction(*driving_, mp_pool_.applied(), params.IMAGE_EDGE_CAMERA, "resumed");
    }

    for (const auto &tp : obs.triple_points)
    {
        ++mp_seen_;
        // EXACT association: both sides index the ORIGINAL polygon vertex list. No gate, no
        // nearest-neighbour, so no misassociation mode to defend against.
        const auto it = std::ranges::find_if(matches,
            [&](const auto &m) { return m.model_index == tp.vertex; });
        if (it == matches.end()) continue;
        // A SUPPRESSED match is a retired landmark: still detected, still carrying numbers, but
        // deliberately kept out of the loss (corner_detector.h). It must stay out of this one too.
        if (it->suppressed) continue;

        const auto pr = rc::mount::make_pair(tp, *it, driving_->model(),
                                             driving_->cam_R_robot(),
                                             driving_->cam_t_robot(),
                                             params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                             params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                             params.IMAGE_EDGE_MOUNT_YAW_SIGMA);
        if (not pr.ok) continue;
        ++mp_paired_;
        mp_win_.add(pr);
        mp_pool_.add(pr);
        // ★ IN RADIANS, not pixels. A pixel is 0.128 deg on the zed and 0.188 on the ricoh, so a
        //   pixel residual cannot be compared across cameras and an angular one can.
        {
            // ★ AT THE MEASUREMENT, not at the principal point. The closure differences the two
            //   cameras' residuals in radians; the panorama's scale is constant but the ZED is a
            //   pinhole, whose true local scale is fx·sec^2(theta). Measured on a synthetic drive of
            //   the real pair, that constant was 82% of the leakage a LiDAR error puts into the
            //   closure — a bias in the comparison, not in either camera (mount_replay --selftest).
            const Eigen::Vector2f ppr = rc::img::px_per_rad_at(obs.cam, pr.uv_image);
            if (ppr.x() > 0.f and ppr.y() > 0.f)
                loop_closure_observe(params.IMAGE_EDGE_CAMERA, tp.vertex,
                                     tp.from == rc::ContourClass::WallCeiling,
                                     static_cast<double>(pr.r.x() / ppr.x()),
                                     static_cast<double>(pr.r.y() / ppr.y()), timestamp_ms);
        }

        if (not mp_csv_.is_open()) open_pair_log(mp_csv_, params.IMAGE_EDGE_CAMERA, *driving_);
        write_pair_row(mp_csv_, params.IMAGE_EDGE_CAMERA, timestamp_ms, pr,
                       tp.from == rc::ContourClass::WallCeiling,
                       it->angle_deg, it->assoc_chi2_val, it->n_rivals, it->runnerup_chi2,
                       driving_->mount_correction());
    }

    if (timestamp_ms - mp_win_start_ms_ < 5000) return;
    mp_win_start_ms_ = timestamp_ms;
    if (mp_csv_.is_open()) mp_csv_.flush();

    const auto win  = mp_win_.solve();
    const auto pool = mp_pool_.solve();
    apply_mount_solve(mp_pool_, *driving_, pool, params.IMAGE_EDGE_CAMERA);
    mp_pool_.save(mp_pool_.path());   // per (robot, camera); a kill -9 costs at most one window
    if (viewer() != nullptr and pool.ok)
    {
        Eigen::Matrix<float, rc::camcal::P_COUNT, 1> pv, sv;
        const double psig[4] = {params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0};
        for (int i = 0; i < rc::camcal::P_COUNT; ++i)
        {
            // The estimator works in units of the PRIOR SIGMA (h carries it), so convert back to
            // physical here — and scale the posterior sigma the same way, or the popup would show a
            // physical value with a dimensionless uncertainty beside it.
            pv(i) = static_cast<float>(pool.p(i)     * psig[i]);
            sv(i) = static_cast<float>(pool.sigma(i) * psig[i]);
        }
        viewer()->set_camera_calibration(pv, sv, pool.informed, static_cast<float>(pool.cond),
                                        mp_pool_.pairs(), params.IMAGE_EDGE_CAMERA);
        if (loop_n_ > 0)
        {
            const double R = 180.0 / M_PI;
            const double mu = loop_du_sum_ / loop_n_, mv = loop_dv_sum_ / loop_n_;
            viewer()->set_loop_closure(
                mu * R, mv * R,
                std::sqrt(std::max(0.0, loop_du_sq_ / loop_n_ - mu * mu)) * R,
                std::sqrt(std::max(0.0, loop_dv_sq_ / loop_n_ - mv * mv)) * R, loop_n_);
        }
    }
    mp_win_.reset();
    if (not win.ok) return;
    ++mp_wins_;
    mp_sum_  += win.p;
    mp_sum2_ += win.p.cwiseProduct(win.p);
    ++mp_sum_n_;

    const double sig[4] = {params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                           params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                           params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0};
    const char *nm[4] = {"pitch", "height", "yaw", "dt"};
    const auto phys = [&](double v, int i)
    { return (i == 0 or i == 2) ? v * sig[i] * 180.0 / M_PI : v * sig[i]; };

    QString body;
    for (int i = 0; i < 3; ++i)      // dt is not estimated here; do not print an unmoved prior
        body += QString(" | %1 %2 (%3%4)").arg(nm[i])
                    .arg(phys(win.p(i), i), 0, 'f', 4).arg(win.sigma(i), 0, 'f', 3)
                    .arg((win.informed >> i) & 1 ? ", INF" : "");
    qInfo().nospace().noquote()
        << "[mount/pair] window " << mp_wins_ << " (" << win.chi2_dof * 0 + mp_paired_
        << " pairs of " << mp_seen_ << " triple points"
        // ★ THE CLUSTER COUNT IS THE REAL SAMPLE SIZE FOR THE MOUNT'S LEVEL, and printing it beside
        //   the pair count is what stops "395171 pairs" being read as 395171 measurements.
        << ", " << win.clusters << " corners" << (win.marginalised ? ", offset marginalised" : "")
        << ")" << body
        << " | chi2/dof " << QString::number(win.chi2_dof, 'f', 2)
        << " | cond " << QString::number(win.cond, 'f', 1)
        << " (" << nm[win.rho_i] << "/" << nm[win.rho_j] << " rho "
        << QString::number(win.rho, 'f', 3) << ")";

    // ★ The pooled solve is the CLAIM under test: with the pose gone from the residual, windows
    //   should be comparable draws of one static quantity, so summing their information is legitimate
    //   and buys the range diversity one window never has. If the between-window scatter does NOT
    //   collapse toward the pooled sigma, the pose was not the dominant nuisance and pooling is still
    //   wrong — which is why both numbers are on the same line rather than only the flattering one.
    if (pool.ok and mp_sum_n_ >= 2)
    {
        QString pb;
        for (int i = 0; i < 3; ++i)
        {
            const double m = mp_sum_(i) / mp_sum_n_;
            const double sc = std::sqrt(std::max(0.0, mp_sum2_(i) / mp_sum_n_ - m * m));
            pb += QString(" | %1 %2 (%3%4) [scatter %5]").arg(nm[i])
                      .arg(phys(pool.p(i), i), 0, 'f', 4).arg(pool.sigma(i), 0, 'f', 3)
                      .arg((pool.informed >> i) & 1 ? ", INF" : "")
                      .arg(phys(sc, i), 0, 'f', 4);
        }
        qInfo().nospace().noquote()
            << "[mount/pool] " << mp_pool_.pairs() << " pairs over " << mp_wins_ << " windows" << pb
            << " | chi2/dof " << QString::number(pool.chi2_dof, 'f', 2)
            << " | cond " << QString::number(pool.cond, 'f', 1)
            << " (" << nm[pool.rho_i] << "/" << nm[pool.rho_j] << " rho "
            << QString::number(pool.rho, 'f', 3) << ")"
            << (pool.cond < 50.0 && win.cond > 50.0
                    ? "   <- POOLING BROKE THE RIDGE: the pair is separable across poses and was not"
                      " within one window" : "");
    }
}

// One camera's angular disagreement with the LiDAR, for one corner. When a SECOND camera reports the
// same corner close enough in time, the two are differenced: the LiDAR corner's own error is common
// to both and cancels, leaving camera-vs-camera.
void MountCalibrator::loop_closure_observe(const std::string &cam, int vertex, bool ceiling,
                                          double du_rad, double dv_rad, std::int64_t ts)
{
    const int key = vertex * 2 + (ceiling ? 1 : 0);
    for (auto &[k, v] : loop_last_)
    {
        if (k.second != key or k.first == cam) continue;
        // ★ NEAR-SIMULTANEOUS ONLY. The two cameras free-run at different rates, and differencing
        //   observations 200 ms apart would fold the robot's own motion into what is meant to be a
        //   static mount comparison. 60 ms is about one frame at these rates.
        if (std::abs(ts - v.ts) > 60) continue;
        const double ddu = du_rad - v.du_rad, ddv = dv_rad - v.dv_rad;
        loop_du_sum_ += ddu; loop_dv_sum_ += ddv;
        loop_du_sq_  += ddu * ddu; loop_dv_sq_ += ddv * ddv;
        ++loop_n_;
        if (not loop_csv_.is_open())
        {
            loop_csv_.open("etc/camera_loop.csv", std::ios::out | std::ios::trunc);
            if (loop_csv_.is_open())
            {
                loop_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                loop_csv_ << "ts_ms,vertex,ceiling,cam_a,cam_b,"
                             "du_a_deg,dv_a_deg,du_b_deg,dv_b_deg,ddu_deg,ddv_deg\n";
            }
        }
        if (loop_csv_.is_open())
        {
            const double R = 180.0 / M_PI;
            loop_csv_ << ts << ',' << vertex << ',' << (ceiling ? 1 : 0) << ','
                      << cam << ',' << k.first << ','
                      << du_rad * R << ',' << dv_rad * R << ','
                      << v.du_rad * R << ',' << v.dv_rad * R << ','
                      << ddu * R << ',' << ddv * R << '\n';
        }
        if (loop_n_ % 500 == 0)
        {
            const double R = 180.0 / M_PI;
            const double mu = loop_du_sum_ / loop_n_, mv = loop_dv_sum_ / loop_n_;
            const double su = std::sqrt(std::max(0.0, loop_du_sq_ / loop_n_ - mu * mu));
            const double sv = std::sqrt(std::max(0.0, loop_dv_sq_ / loop_n_ - mv * mv));
            qInfo().nospace().noquote()
                << "[loop] " << loop_n_ << " shared corners | camera-vs-camera du "
                << QString::number(mu * R, 'f', 4) << " +/- " << QString::number(su * R, 'f', 4)
                << " deg, dv " << QString::number(mv * R, 'f', 4) << " +/- "
                << QString::number(sv * R, 'f', 4) << " deg"
                << "   <- the LiDAR corner's own error CANCELS here; what is left is the two"
                   " cameras disagreeing with each other";
        }
        break;
    }
    loop_last_[{cam, key}] = CornerAngle{du_rad, dv_rad, ts};
}

}   // namespace rc
