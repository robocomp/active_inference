// Plan Task 2.3: proposal publisher. A: pure (round trip, layout, rate limiter). B: graph on a PRIVATE domain (92).
// Build+run:  make -C build proposal_publisher_test && ../bin/proposal_publisher_test
#include "../src/proposal_writer.h"

#include <QCoreApplication>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <print>

using namespace sc::calib;

namespace
{
int failures = 0;
void check(bool ok, int line, std::string_view what) { if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)

MountPosteriors make_post()
{
    MountPosteriors p;
    for (std::size_t s = 0; s < kMountSensors; ++s)
    {
        p[s] = MountPosterior::prior(0.05, 0.0175);
        for (std::size_t i = 0; i < kMountDoF; ++i) p[s].mean[i] = 0.001 * double(s + 1) * double(i + 1);
        p[s].cov[1] = p[s].cov[6] = 1e-5 * double(s + 1);   // off-diagonal, symmetric
    }
    return p;
}
}

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);

    // A1 layout + round trip
    const auto post = make_post();
    const auto v = pack_proposal(post);
    CHECK(v.size() == 168 and kProposalFloats == 4 * 42);
    CHECK(v[0] == float(post[0].mean[0]) and v[42] == float(post[1].mean[0]) and v[84] == float(post[2].mean[0]) and v[126] == float(post[3].mean[0]));
    CHECK(v[6 + 1] == float(post[0].cov[1]));                 // row-major (0,1)
    const auto back = unpack_proposal(v);
    CHECK(back.has_value());
    if (back)
        for (std::size_t s = 0; s < kMountSensors; ++s)
            for (std::size_t i = 0; i < 36; ++i)
            {
                if (i < 6) CHECK(std::abs((*back)[s].mean[i] - post[s].mean[i]) < 1e-7);
                CHECK(std::abs((*back)[s].cov[i] - post[s].cov[i]) < 1e-7);
            }
    CHECK(not unpack_proposal(std::vector<float>(167)));
    std::println("A1 layout 4x42=168 + round trip ok");

    // A2 50 Hz changing posterior for 10 s: <= 1 write / s
    {
        ProposalPublisher pub; auto p = post; int writes = 0;
        for (int k = 0; k < 500; ++k)   // 20 ms steps; the mean drifts by 5 sigma per step
        {
            for (auto& m : p) m.mean[0] += 5.0 * m.sigma(0);
            writes += pub.offer(p, 1'700'000'000'000ll + 20ll * k).has_value();
        }
        std::println("A2 {} writes in 10 s at 50 Hz", writes);
        CHECK(writes >= 1 and writes <= 10);
    }
    // A3 small move: none; big move: exactly one, version +1; ready false
    {
        ProposalPublisher pub; auto p = post; std::int64_t t = 1'700'000'000'000ll;
        auto first = pub.offer(p, t);
        CHECK(first and first->version == 1 and not first->ready and first->mount_proposal.size() == 168);
        t += 2000;
        p[2].mean[3] += 0.05 * p[2].sigma(3);
        CHECK(not pub.offer(p, t));                                   // < 0.1 sigma
        t += 2000;
        p[2].mean[3] += 0.2 * p[2].sigma(3);                          // total 0.25 sigma from the published one
        auto w = pub.offer(p, t);
        CHECK(w and w->version == 2);
        t += 2000;
        CHECK(not pub.offer(p, t));                                   // unchanged since the write
        t += 2000;
        p[0].mean[5] += 0.5 * p[0].sigma(5);
        CHECK(pub.offer(p, t) and pub.version() == 3);
        std::println("A3 move gating + version ok");
    }

    // B. graph, PRIVATE domain 92 (never the live 0)
    auto G = std::make_shared<DSR::DSRGraph>("proposal_publisher_test", 992, std::string(SHADOW_JSON), true,
                                             static_cast<int8_t>(92), DSR::SignalMode::QT);
    auto node = DSR::Node::create<metaconcept_node_type>("self_calibration 27");
    const auto id = G->insert_node(node);
    CHECK(id.has_value());
    ProposalPublisher pub;
    const auto w = pub.offer(post, 1'700'000'000'000ll);
    CHECK(w.has_value());
    if (id and w)
    {
        CHECK(sc::write_proposal(*G, *id, *w));
        auto n = G->get_node(*id);
        CHECK(n.has_value());
        if (n)
        {
            const auto a = G->get_attrib_by_name<mount_proposal_att>(*n);
            const auto ver = G->get_attrib_by_name<mount_proposal_version_att>(*n);
            const auto rdy = G->get_attrib_by_name<mount_proposal_ready_att>(*n);
            CHECK(a.has_value() and ver.has_value() and rdy.has_value());
            if (a and ver and rdy)
            {
                const std::vector<float> got = *a;
                CHECK(got == v);
                CHECK(*ver == 1 and *rdy == false);
            }
        }
    }
    std::println("B graph attributes on the agent node ok");
    G.reset();
    std::println("{}", failures == 0 ? "PASS" : "FAILED");
    std::fflush(stdout);
    // _Exit: the Ice static teardown in this link throws at normal exit (same as sensor_inputs_test).
    std::_Exit(failures == 0 ? 0 : 1);
}
