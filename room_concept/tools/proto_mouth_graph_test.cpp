// Task M graph-level test: the code the proto BIRTH runs (rc::crossing::write_mouth, called from
// RoomSceneGraph::step_proto_room_impl) writes onto a real DSR node attributes equal to proto_frame's pose in the old
// room, with the covariance and the frame-room id. Run from the room_concept root:
//   ./bin/proto_mouth_graph_test ../ltsm_agent/etc/stage_two_rooms.json
// (Limit: this is the write the birth calls, not the whole birth path, which needs a RoomConcept + live localiser.)
#include "../src/proto_mouth.h"
#include <QCoreApplication>
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char *w) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", w); }

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    auto G = std::make_shared<DSR::DSRGraph>("mouth_live", 193, argc > 1 ? argv[1] : "../ltsm_agent/etc/stage_two_rooms.json", true, 43);
    using Eigen::Vector2f;
    for (const double yaw : {0.0, 0.7, -2.4})
    {
        const Eigen::Rotation2Df R(static_cast<float>(yaw));
        std::vector<Vector2f> outline;   // 32-vertex outline: rectangle densified
        const Vector2f corners[4] = {{-3, -2}, {3, -2}, {3, 3}, {-3, 3}};
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) outline.push_back(R * (corners[i] + (corners[(i + 1) % 4] - corners[i]) * (j / 8.f)));
        rc::DoorAperture ap; ap.a = R * Vector2f(-0.5f, 3); ap.b = R * Vector2f(0.5f, 3); ap.p_open = 1.f; ap.sigma_along = 0.03f;
        const auto c = rc::crossing::evaluate(ap, outline, R * Vector2f(0.2f, 3.8f), Eigen::Matrix2f::Identity() * 4e-4f, 0.25f);
        DSR::Node proto = DSR::Node::create<room_node_type>("room_mouth_test_" + std::to_string(int(yaw * 10)));
        rc::crossing::write_mouth(*G, proto, c, ap.sigma_along, 0.06f, 7777, Eigen::Affine2f::Identity());
        const auto id = G->insert_node(proto);
        check(id.has_value(), "proto node inserted");
        const auto n = G->get_node(id.value()).value();
        const auto t = G->get_attrib_by_name<rt_translation_att>(n), r = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(n);
        const auto cv = G->get_attrib_by_name<rt_covariance_att>(n);
        const auto f = G->get_attrib_by_name<interaction_node_id_att>(n);
        check(t.has_value() and r.has_value() and cv.has_value() and f.has_value(), "mouth attributes present on the node");
        if (not (t and r and cv and f)) continue;
        const auto Tp = rc::crossing::proto_frame(c);
        const Eigen::Matrix2f Rm = Eigen::Rotation2Df(r->get()[2]).toRotationMatrix();
        check(std::abs(t->get()[0] - Tp.translation().x()) < 1e-5f and std::abs(t->get()[1] - Tp.translation().y()) < 1e-5f, "translation = proto origin in the old room");
        check((Rm - Tp.linear()).cwiseAbs().maxCoeff() < 1e-5f and std::abs(Rm.determinant() - 1.f) < 1e-5f, "rotation = proto frame rotation (orthonormal)");
        check(cv->get().size() == 36 and cv->get()[0] > 0.f and cv->get()[35] > 0.f, "6x6 covariance stated");
        check(f.value() == 7777, "frame room id recorded");
    }
    // ── THE OLD ROOM IS NOT THE FIRST ROOM (clean6 frame report §3): its published frame is a rigid transform away from the
    //    internal frame the crossing is computed in, so the mouth, stated "in the old room's frame" and read by ltsm as
    //    T(old <- new), must be composed with T_pub_int of the OLD room. A case where the old code's seam is metres wrong.
    {
        const Eigen::Rotation2Df Ry(1.2f);
        Eigen::Affine2f T_old_pub_int = Eigen::Affine2f::Identity();
        T_old_pub_int.linear() = Ry.toRotationMatrix();
        T_old_pub_int.translation() = Vector2f(3.1f, -5.2f);
        std::vector<Vector2f> outline;
        const Vector2f corners[4] = {{-3, -2}, {3, -2}, {3, 3}, {-3, 3}};
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) outline.push_back(corners[i] + (corners[(i + 1) % 4] - corners[i]) * (j / 8.f));
        rc::DoorAperture ap; ap.a = Vector2f(-0.5f, 3); ap.b = Vector2f(0.5f, 3); ap.p_open = 1.f; ap.sigma_along = 0.03f;
        const auto c = rc::crossing::evaluate(ap, outline, Vector2f(0.2f, 3.8f), Eigen::Matrix2f::Identity() * 4e-4f, 0.25f);
        DSR::Node proto = DSR::Node::create<room_node_type>("room_mouth_test_second");
        rc::crossing::write_mouth(*G, proto, c, ap.sigma_along, 0.06f, 7777, T_old_pub_int);
        const auto id = G->insert_node(proto);
        check(id.has_value(), "proto node inserted (old room is not the first room)");
        const auto n = G->get_node(id.value()).value();
        const auto t = G->get_attrib_by_name<rt_translation_att>(n), r = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(n);
        const auto cv = G->get_attrib_by_name<rt_covariance_att>(n);
        check(t.has_value() and r.has_value() and cv.has_value(), "mouth attributes present (old room is not the first room)");
        if (t and r and cv)
        {
            const Eigen::Affine2f Tp = rc::crossing::proto_frame(c);                    // the proto frame in the INTERNAL frame
            const Eigen::Affine2f expected = T_old_pub_int * Tp;                       // the proto frame in the old room's published frame
            Eigen::Affine2f mouth = Eigen::Affine2f::Identity();
            mouth.linear() = Eigen::Rotation2Df(r->get()[2]).toRotationMatrix();
            mouth.translation() = Vector2f(t->get()[0], t->get()[1]);
            check((mouth.translation() - expected.translation()).norm() < 1e-4f and (mouth.linear() - expected.linear()).cwiseAbs().maxCoeff() < 1e-4f,
                  "mouth = T_pub_int(old) * proto frame");
            // the seam ltsm derives: T(new <- old) = inverse(mouth). The truth, and what the old (internal-frame) mouth gave.
            const Eigen::Affine2f seam_truth = expected.inverse(), seam_old_code = Tp.inverse();
            const float err_seam = (mouth.inverse().translation() - seam_truth.translation()).norm();
            const float err_old = (seam_old_code.translation() - seam_truth.translation()).norm();
            check(err_seam < 1e-3f, ("the seam read as inverse(mouth) is the true T(new<-old) (err " + std::to_string(err_seam) + " m)").c_str());
            check(err_old > 1.f, ("this case discriminates: the internal-frame mouth is off by " + std::to_string(err_old) + " m").c_str());
            // covariance: the xy block rotated with the frame, yaw variance unchanged
            const auto mi = rc::crossing::mouth_of(c, ap.sigma_along, 0.06f);
            Eigen::Matrix2f Cin; Cin << mi.cov[0], mi.cov[1], mi.cov[6], mi.cov[7];
            const Eigen::Matrix2f Cex = T_old_pub_int.linear() * Cin * T_old_pub_int.linear().transpose();
            check(std::abs(cv->get()[0] - Cex(0, 0)) < 1e-5f and std::abs(cv->get()[1] - Cex(0, 1)) < 1e-5f and std::abs(cv->get()[7] - Cex(1, 1)) < 1e-5f, "xy covariance rotated with the frame");
            check(std::abs(cv->get()[35] - mi.cov[35]) < 1e-7f, "yaw variance unchanged by a constant rotation");
        }
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
