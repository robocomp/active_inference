/*
 * atlas_io_test.cpp — archive round-trip, locale, corruption; constants loader. PARSES FILES, so it sets
 * the agents' locale first (CLAUDE.md): a "C"-locale harness would answer a different question.
 */
#include "atlas_io.h"

#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace rc::atlas;
static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}

static ArchiveRecord make_record(std::uint64_t uid)
{
    AtlasParams P;
    ViewAtlas a(P, 4, 3);
    ViewObs o; o.shape = Eigen::Vector4f(0.1f, 0.2f, 0.3f, 0.9f); o.colour = Eigen::Vector3f(0.3f, 0.3f, 0.9f);
    o.bearing_rad = 0.123456789f; o.range_m = 1.987654321f; o.quality = 0.7f; o.stamp_ms = 1790000000123ull;
    a.update(o);
    o.bearing_rad = -2.5f; a.update(o);
    ArchiveRecord r;
    r.object_uid = uid; r.agent = "chair_concept"; r.cls = "chair"; r.name = "chair_2";
    r.pose = Eigen::Vector3f(-6.90769f, 1.43326f, 1.0472f);
    r.cov = Eigen::Matrix3f::Identity() * 0.0123456f; r.cov(0, 1) = r.cov(1, 0) = 0.001f;
    r.born_ms = 1790000000000ull; r.died_ms = 1790000099999ull;
    r.atlas = a.state();
    return r;
}

/// Overwrites one little-endian u32 at byte offset `at` of a file (blob header: magic 0, version 4,
/// shape_dim 8, colour_dim 12, n_bins 16, n_records 20, generation u64 24).
static void patch_u32(const std::string& path, std::streamoff at, std::uint32_t v)
{
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(at);
    f.write(reinterpret_cast<const char*>(&v), 4);
}

static bool same(const ArchiveRecord& a, const ArchiveRecord& b)
{
    if (a.object_uid != b.object_uid or a.agent != b.agent or a.cls != b.cls or a.name != b.name) return false;
    if (a.pose != b.pose or a.cov != b.cov or a.born_ms != b.born_ms or a.died_ms != b.died_ms) return false;
    if (a.atlas.n_c != b.atlas.n_c or a.atlas.colour != b.atlas.colour) return false;
    for (std::size_t i = 0; i < a.atlas.bins.size(); ++i)
    {
        const auto &x = a.atlas.bins[i], &y = b.atlas.bins[i];
        if (x.n != y.n or x.bearing != y.bearing or x.stamp_ms != y.stamp_ms or x.mean != y.mean) return false;
        if (x.poses.size() != y.poses.size()) return false;
        for (std::size_t k = 0; k < x.poses.size(); ++k) if (x.poses[k] != y.poses[k]) return false;
    }
    return true;
}

int main()
{
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    std::printf("locale: %s\n", std::setlocale(LC_NUMERIC, nullptr));
    const auto dir = std::filesystem::temp_directory_path() / "atlas_io_test";
    std::filesystem::create_directories(dir);
    const std::string base = (dir / "atlas_chair_concept_22").string();
    const AtlasParams P;

    std::printf("round trip\n");
    std::vector<ArchiveRecord> recs = {make_record(7), make_record(8)};
    std::string why;
    check(save_archive(base, recs, &why), "save succeeds");
    std::vector<ArchiveRecord> back;
    check(load_archive(base, P, 4, 3, back, &why), "load succeeds");
    check(back.size() == 2 and same(back[0], recs[0]) and same(back[1], recs[1]), "records read back bit-exact under es_ES");
    check(save_archive(base, {}, &why) and load_archive(base, P, 4, 3, back, &why) and back.empty(),
          "empty archive saves and reloads (agent with no dead objects)");
    check(save_archive(base, recs, &why), "re-save the two records");

    std::printf("refusals\n");
    back.clear();
    check(not load_archive(base, P, 5, 3, back, &why) and why.find("shape_dim") != std::string::npos,
          "wrong shape dimension refused, reason names it");
    check(back.empty(), "refused load leaves output empty");
    {
        std::ifstream in(base + ".blob", std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), {});
        std::ofstream out(base + ".blob", std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), std::streamsize(bytes.size() / 2));
    }
    check(not load_archive(base, P, 4, 3, back, &why) and why.find("blob short") != std::string::npos,
          "truncated blob refused");
    {
        std::ofstream out(base + ".blob", std::ios::binary | std::ios::trunc);
        out.write("XXXX", 4);
    }
    check(not load_archive(base, P, 4, 3, back, &why) and why.find("magic") != std::string::npos,
          "wrong magic refused");
    check(not load_archive((dir / "missing").string(), P, 4, 3, back, &why), "missing files refused");
    ArchiveRecord bad = make_record(9); bad.name = "chair,2";
    check(not save_archive(base, {bad}, &why) and why.find("comma") != std::string::npos,
          "a comma in a text field is refused at save time");

    std::printf("header field refusals (patched blob)\n");
    {
        const std::string blob = base + ".blob";
        auto refused_with = [&](const char* field) {
            back.clear(); why.clear();
            return not load_archive(base, P, 4, 3, back, &why) and why.find(field) != std::string::npos and back.empty();
        };
        auto fresh = [&] { return save_archive(base, recs, &why); };
        fresh(); patch_u32(blob, 4, 99);
        check(refused_with("version"), "wrong version refused, reason names version");
        fresh(); patch_u32(blob, 12, 2);
        check(refused_with("colour_dim"), "wrong colour_dim refused, reason names colour_dim");
        fresh(); patch_u32(blob, 16, 5);
        check(refused_with("n_bins"), "wrong n_bins refused, reason names n_bins");
        fresh(); patch_u32(blob, 20, 3);
        check(refused_with("record count"), "bad record count refused, reason names record count");
        fresh(); patch_u32(blob, 4, 1);
        check(refused_with("version"), "version-1 blob refused, reason names version");
        fresh();
    }

    std::printf("torn pair (crash between the two renames)\n");
    {
        std::vector<ArchiveRecord> A = {make_record(1), make_record(2)}, B = {make_record(3), make_record(4)};
        B[0].atlas.colour = Eigen::Vector3f(0.9f, 0.1f, 0.1f);   // same count, different content
        const std::string a = (dir / "tornA").string(), bb = (dir / "tornB").string();
        check(save_archive(a, A, &why) and save_archive(bb, B, &why), "save A and B");
        std::filesystem::copy_file(bb + ".blob", a + ".blob", std::filesystem::copy_options::overwrite_existing);
        back.clear();
        check(not load_archive(a, P, 4, 3, back, &why) and why.find("generation") != std::string::npos and back.empty(),
              "A.idx + B.blob refused, reason names generation");
        std::filesystem::copy_file(bb + ".idx", a + ".idx", std::filesystem::copy_options::overwrite_existing);
        check(load_archive(a, P, 4, 3, back, &why) and back.size() == 2, "a consistent pair loads");
    }

    std::printf("constants loader\n");
    const std::string cpath = (dir / "constants.txt").string();
    {
        std::ofstream f(cpath);
        f << "# fitted test\nell_rad = 0.873\nmu_own_d = 0.9512\ncov_bg_dc = -0.00125\nrepeat_ell_rad=0.0912\n";
    }
    AtlasParams q;
    check(load_params(cpath, q, &why), "constants file loads");
    check(q.ell_rad == 0.873f and q.mu_own.x() == 0.9512f, "decimals parsed with points under es_ES");
    check(q.cov_bg(0, 1) == -0.00125f and q.cov_bg(1, 0) == -0.00125f, "off-diagonal set symmetrically");
    check(q.repeat_ell_rad == 0.0912f, "no spaces around '=' is fine");
    { std::ofstream f(cpath); f << "ell_radd = 0.5\n"; }
    check(not load_params(cpath, q, &why) and why.find("ell_radd") != std::string::npos, "unknown key refused");
    { std::ofstream f(cpath); f << "ell_rad = 0,5\n"; }
    check(not load_params(cpath, q, &why), "a comma decimal is refused, not read as 0");

    std::printf("constants invariants\n");
    {
        std::ofstream f(cpath); f << "n0 = 0\n";
    }
    AtlasParams r;
    check(not load_params(cpath, r, &why) and why.find("n0") != std::string::npos, "n0 = 0 refused, reason names n0");
    { std::ofstream f(cpath); f << "cov_bg_dd = 0.001\ncov_bg_dc = 0.01\n"; }
    check(not load_params(cpath, r, &why) and why.find("cov_bg") != std::string::npos,
          "cov_bg not positive definite refused, reason names cov_bg");
    {
        std::ofstream f(cpath); f << "ell_rad = 0.5\nn0 = 0\n";
    }
    AtlasParams s;
    s.ell_rad = 0.6f;
    check(not load_params(cpath, s, &why) and s.ell_rad == 0.6f, "refused load leaves the passed AtlasParams unchanged");
    { std::ofstream f(cpath); f << "cov_bg_dd = nan\n"; }
    check(not load_params(cpath, r, &why) and why.find("cov_bg_dd") != std::string::npos,
          "cov_bg_dd = nan refused, reason names cov_bg_dd");
    { std::ofstream f(cpath); f << "ell_rad = inf\n"; }
    check(not load_params(cpath, r, &why) and why.find("ell_rad") != std::string::npos,
          "ell_rad = inf refused, reason names ell_rad");

    std::printf("atomic save\n");
    check(save_archive(base, recs, &why), "baseline archive saved");
    std::filesystem::create_directories(base + ".idx.tmp");   // the index temp cannot be created as a file
    check(not save_archive(base, {make_record(42)}, &why), "save fails when the index temp cannot be written");
    std::vector<ArchiveRecord> still;
    check(load_archive(base, P, 4, 3, still, &why) and still.size() == recs.size(), "previous archive still loads");
    check(not std::filesystem::exists(base + ".blob.tmp"), "blob temp removed after the failed save");
    std::filesystem::remove_all(base + ".idx.tmp");
    check(save_archive(base, recs, &why) and not std::filesystem::exists(base + ".idx.tmp"), "normal save leaves no temp files");

    std::filesystem::remove_all(dir);
    std::printf("\natlas_io_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
