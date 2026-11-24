// engine_no_beam_test — the shared persistent engine with a LiDAR ray-set but a BeamConfig that was never built by
// BeamConfig::make (shared-engine Task 4 (a), Review Focus 1). The camera frames carry the truth, so θ must converge toward it
// with no NaN, and the engine must refuse the unready beam loudly ONCE: exactly one stderr line containing "beam not ready"
// over the whole run. Runs on the minimal toy model (toy_model.h). Run: ./engine_no_beam_test
#include <cmath>
#include <cstdio>
#include <string>
#include <unistd.h>

#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

// Everything written to fd 2 while `fn` runs.
template <class Fn>
static std::string capture_stderr(Fn&& fn)
{
    std::fflush(stderr);
    FILE* tmp = std::tmpfile();
    const int saved = dup(2);
    dup2(fileno(tmp), 2);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    std::rewind(tmp);
    for (int c = std::fgetc(tmp); c != EOF; c = std::fgetc(tmp)) out.push_back(char(c));
    std::fclose(tmp);
    return out;
}

static int count_lines_with(const std::string& text, const std::string& needle)
{
    int n = 0;
    std::size_t start = 0;
    while (start < text.size())
    {
        std::size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        if (text.substr(start, end - start).find(needle) != std::string::npos) ++n;
        start = end + 1;
    }
    return n;
}

int main()
{
    using Engine = rc::ai::engine::PersistentEngine<toy::ToyBox>;
    static_assert(rc::ai::engine::PersistentModel<toy::ToyBox>);

    const toy::ToyState truth{1.0f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f};
    toy::ToyBox m;
    m.s = toy::ToyState{1.08f, 1.94f, 0.9f, 0.52f, 0.46f, 0.4f};   // the seed, off the truth on every component
    m.prior_mean << 0.f, 0.f, 0.75f, 0.55f, 0.45f, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();

    // A ray-set that WOULD measure (non-empty, Σq > 0, one m per ray): only the config is unready.
    rc::ai::beam::BeamRays rays;
    rays.origin = {0.0, 0.0, 0.3};
    for (int i = 0; i < 8; ++i)
    {
        const double a = 1.0 + 0.04 * i;
        rays.endpoints.emplace_back(std::cos(a) * 2.2, std::sin(a) * 2.2, 0.3);
        rays.m.push_back(0.9);
    }
    rays.q_sum = 1.0;
    CHECK(rc::ai::beam::beam_readiness(rays, m.beam_config()) == rc::ai::beam::Readiness::kUnready);

    rc::ai::engine::FrameLatents<6> fl;
    fl.rho_cam.fill(1.0f);  fl.sig2_cam.fill(1e-4f);
    fl.rho_L.fill(1.0f);    fl.sig2_L.fill(1e-4f);
    fl.rho_chain = 1.0f;
    fl.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 1e-4).asDiagonal();
    fl.robot_xy = {0.0f, 0.0f};
    fl.camera_measured = true;
    fl.lidar_measured  = true;   // the caller believes the rays measure; the engine must find the config unready

    const Eigen::Matrix<float, 6, 1> err0 = m.s.vec() - truth.vec();
    Engine eng;
    bool finite = true;
    const std::string log = capture_stderr([&]
    {
        for (int t = 0; t < 10; ++t)
        {
            const auto f = toy::make_frame(toy::surface_points(truth, 6), rays);
            eng.update(m, f, fl);
            finite = finite and m.s.vec().allFinite() and eng.P().allFinite() and eng.latent_mean().allFinite();
        }
    });
    std::fputs(log.c_str(), stdout);

    const Eigen::Matrix<float, 6, 1> err = m.s.vec() - truth.vec();
    for (int k = 0; k < 6; ++k)
        std::printf("theta[%d]: seed error %+.5f, final error %+.5f, posterior sigma %.5f\n", k, err0(k), err(k),
                    std::sqrt(eng.P()(k, k)));

    CHECK(finite);
    CHECK(err.norm() < err0.norm());   // toward the truth
    for (int k = 0; k < 6; ++k)        // and consistent with the engine's own posterior
        CHECK(std::abs(err(k)) <= 3.0 * std::sqrt(eng.P()(k, k)));
    CHECK(count_lines_with(log, "beam not ready") == 1);
    CHECK(eng.last_beam_staged() == 0);   // the LiDAR channel did not measure

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
