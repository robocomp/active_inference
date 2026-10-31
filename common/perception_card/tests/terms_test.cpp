// Golden test for the closed term library. Every table below was PRINTED by a scratch harness that called the
// agents' CURRENT (pre-move) code on 20 fixed input tuples per term; the terms must reproduce them to 1e-6.
// Runs under the AGENT's locale:   LANG=es_ES.UTF-8 ./terms_test
#include "perception_card/terms.h"
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;

static bool near(float a, float b) { return std::abs(a - b) <= 1e-6f; }
#define EXPECT_NEAR(name, i, got, want) do { if (not near((got), (want))) { std::printf("FAIL %s[%d]: got %.9g want %.9g\n", name, i, double(got), double(want)); ++bad; } } while (0)

struct Env4 { float fmax, fmin, vis, lo, hi, soft, out; };
struct Env3 { float fill, vis, lo, hi, soft, out; };
struct Rng  { float range, ref, power, out; };
struct Obl  { float rx, ry, nx, ny, out; };
struct Res  { int cells, det; float out; };
struct Kap  { double var_b; float range, half_diag, out; };

// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<Env4,20> kEnv4{{
    {0.426833093f, 0.0703623965f, 0.124689914f, 0.18361491f, 0.577911615f, 0.0972006023f, 0.0244713333f},
    {0.343562841f, 0.311715871f, 0.985067129f, 0.168281063f, 0.797448397f, 0.0546695404f, 0.918218791f},
    {0.575247586f, 0.359238505f, 0.746475339f, 0.137504563f, 0.40177995f, 0.0637212545f, 0.0446596257f},
    {0.163248613f, 0.159148112f, 0.594456911f, 0.161398783f, 0.755241513f, 0.0585470311f, 0.291504264f},
    {0.489752889f, 0.289889514f, 1.07577264f, 0.0604887456f, 0.563178718f, 0.0546362251f, 0.781398416f},
    {0.389755577f, 0.138753816f, 0.721064448f, 0.119636059f, 0.596500754f, 0.0239344593f, 0.497235775f},
    {0.823115826f, 0.379609555f, 0.411333114f, 0.0593430847f, 0.728623271f, 0.0537727103f, 0.0603647232f},
    {0.702670455f, 0.0595027283f, 0.368371218f, 0.0788645819f, 0.728889108f, 0.0794792473f, 0.0941596031f},
    {0.522074997f, 0.321307868f, 0.359320402f, 0.181016892f, 0.716585279f, 0.0755805671f, 0.288738102f},
    {0.814159095f, 0.812873423f, 0.817529917f, 0.0515643694f, 0.698040962f, 0.0477135181f, 0.0659274459f},
    {0.34753868f, 0.311873913f, 0.00641468912f, 0.0937846676f, 0.6107288f, 0.06202906f, 0.00614132918f},
    {0.21464926f, 0.121479563f, 1.04611397f, 0.121173218f, 0.680363178f, 0.0953104645f, 0.49705115f},
    {0.0640017241f, 0.0124136806f, 0.00879215449f, 0.133811027f, 0.524703622f, 0.0899458379f, 0.00179978192f},
    {0.131057322f, 0.0951626673f, -0.036090605f, 0.0902260244f, 0.598720789f, 0.0599330403f, 0.0f},
    {0.0545081794f, 0.0542544201f, 0.619185269f, 0.0758538246f, 0.520247698f, 0.0958284885f, 0.272735506f},
    {0.208491474f, 0.135723114f, 0.683921456f, 0.0832487345f, 0.45963186f, 0.0896995813f, 0.414043099f},
    {0.591745615f, 0.0481917113f, 0.775730371f, 0.13662684f, 0.49227038f, 0.0246171951f, 0.000359126396f},
    {0.138024122f, 0.0541303828f, 0.878146589f, 0.139306054f, 0.701624036f, 0.0432575643f, 0.107566178f},
    {0.193889603f, 0.186111182f, 0.388420135f, 0.5f, 0.200000003f, 0.0599999987f, 0.388420135f},
    {0.823887527f, 0.400425643f, 0.787724078f, 0.100000001f, 0.600000024f, 9.99999975e-06f, 0.787724078f},
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<Env3,20> kEnv3{{
    {0.786953926f, 1.0f, 0.0810240805f, 0.781225085f, 0.0341038704f, 0.458102942f},
    {0.173394471f, 0.808860123f, 0.0793864429f, 0.506762326f, 0.0511794314f, 0.696668267f},
    {0.651446819f, 0.0287490264f, 0.155394912f, 0.435746819f, 0.0283491947f, 1.42560511e-05f},
    {0.654431701f, 1.0f, 0.110716209f, 0.401969403f, 0.0396572314f, 0.00171588163f},
    {0.646547079f, 1.01269376f, 0.0681401193f, 0.44423154f, 0.054225333f, 0.0234069005f},
    {0.572132707f, 0.543952465f, 0.088067174f, 0.624599218f, 0.0592760704f, 0.384944975f},
    {0.372670382f, 1.0f, 0.143138796f, 0.464876652f, 0.0252944492f, 0.97443974f},
    {0.32814756f, 0.568730772f, 0.0804738849f, 0.641895413f, 0.0564916879f, 0.559557974f},
    {0.208855703f, 1.01038122f, 0.0868511498f, 0.454399884f, 0.0584977493f, 0.876324475f},
    {0.331222713f, 1.0f, 0.180316791f, 0.632027328f, 0.0812037364f, 0.844321251f},
    {0.165290177f, 0.169500321f, 0.192151964f, 0.494237363f, 0.0452052876f, 0.0602441318f},
    {0.733323932f, 0.683544993f, 0.191005468f, 0.440446794f, 0.0888959467f, 0.0243866444f},
    {0.518643439f, 1.0f, 0.115859613f, 0.682054877f, 0.0785648227f, 0.88369447f},
    {0.823224008f, -0.00568573177f, 0.0554815903f, 0.470674515f, 0.0687057003f, 0.0f},
    {0.773902953f, 0.525915027f, 0.137215376f, 0.419437736f, 0.0515570305f, 0.000542751804f},
    {0.669760227f, 1.0f, 0.104121074f, 0.459080935f, 0.0395470038f, 0.00483362144f},
    {0.75991112f, 0.668072879f, 0.111953616f, 0.499237984f, 0.0473361239f, 0.00270064711f},
    {0.428812295f, 0.949134946f, 0.196003228f, 0.401036203f, 0.0320985951f, 0.28095907f},
    {0.602228463f, 1.0f, 0.128152952f, 0.44811976f, 0.0900205225f, 0.152128354f},
    {0.747475922f, 0.193855435f, 0.5f, 0.200000003f, 0.0599999987f, 0.193855435f},
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<Rng,20> kRng{{
    {10.8076229f, 1.91268182f, 0.740701079f, 0.277286053f},
    {4.77511549f, 2.8933835f, 2.00916553f, 0.365468591f},
    {4.3303256f, 3.51605248f, 0.739788413f, 0.857185602f},
    {4.96265888f, 3.89820766f, 1.59269714f, 0.680779636f},
    {8.10584927f, 4.64367771f, 1.95229149f, 0.337030739f},
    {7.49112034f, 3.47674441f, 1.44680452f, 0.329361618f},
    {8.51913452f, 5.07564259f, 0.763219714f, 0.673516929f},
    {6.90794468f, 5.08523035f, 2.74545813f, 0.431269944f},
    {6.06570721f, 2.715065f, 2.73774004f, 0.110726513f},
    {9.04131031f, 1.8916409f, 2.34591007f, 0.0254803784f},
    {0.424814761f, 3.94682121f, 0.777249932f, 1.0f},
    {2.67707682f, 1.64257574f, 1.08831859f, 0.587663949f},
    {6.57745266f, 3.76719356f, 2.90498805f, 0.198096707f},
    {8.17692184f, 5.28240299f, 2.72903848f, 0.303488493f},
    {8.37175274f, 5.8536582f, 0.959750712f, 0.709357619f},
    {9.53363514f, 2.05041313f, 2.96122003f, 0.0105591975f},
    {7.01852608f, 0.0f, 0.69945848f, 1.0f},
    {2.96869993f, 4.0669899f, 0.0f, 1.0f},
    {0.0f, 1.76910269f, 2.66589594f, 1.0f},
    {0.5f, 5.07525444f, 0.781533122f, 1.0f},
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<Obl,20> kObl{{
    {2.04790306f, -3.33972621f, 0.765151978f, -0.64384973f, 0.948853433f},
    {0.463448048f, 1.18509102f, 0.525044799f, -0.851074576f, 0.60139668f},
    {-2.95055413f, 4.24330139f, -0.840037882f, 0.542527795f, 0.925000846f},
    {1.29101658f, -4.05149841f, -0.970659077f, 0.240459993f, 0.52381134f},
    {-0.453801632f, 4.56976795f, -0.873495221f, -0.486832768f, 0.398131907f},
    {-0.569581985f, 0.0453233719f, 0.181117654f, 0.98346144f, 0.102536522f},
    {-1.571913f, -2.8233881f, -0.995407224f, 0.0957314447f, 0.400562137f},
    {-4.85376596f, 4.12017441f, -0.995378435f, 0.0960302353f, 0.820989549f},
    {-3.07210326f, 3.68489552f, 0.748993099f, 0.662577808f, 0.0292954743f},
    {3.54361343f, 0.831949234f, -0.234677494f, 0.972073257f, 0.00628872216f},
    {-2.58495688f, 3.99589252f, 0.726065695f, -0.687625289f, 0.97171998f},
    {-1.34134388f, 2.99765539f, 0.854758918f, 0.519025207f, 0.124641865f},
    {-2.68276191f, -3.20365477f, 0.724613845f, 0.689155102f, 0.993584216f},
    {4.85058308f, 3.90823174f, -0.771436214f, -0.636306703f, 0.999934196f},
    {-1.45614684f, -2.98449659f, 0.535494149f, 0.844538927f, 0.993826985f},
    {-0.356541634f, 1.13128853f, -0.698256791f, -0.715847373f, 0.47285378f},
    {-0.82805407f, 1.84146309f, 0.313128889f, 0.949710667f, 0.737748861f},
    {-3.69199777f, 3.41505527f, 0.0233040974f, 0.999728441f, 0.661745429f},
    {4.0f, 0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, -0.497853637f, 0.867261052f, 1.0f},
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<Res,20> kRes{{
    {756, 775, 0.975483894f},
    {4, 279, 0.0143369175f},
    {224, 734, 0.305177122f},
    {78, 247, 0.315789461f},
    {3, 334, 0.00898203626f},
    {115, 589, 0.195246175f},
    {124, 128, 0.96875f},
    {28, 93, 0.30107528f},
    {151, 535, 0.282242984f},
    {9, 110, 0.0818181783f},
    {144, 306, 0.470588237f},
    {13, 31, 0.419354826f},
    {652, 791, 0.82427305f},
    {45, 505, 0.0891089141f},
    {563, 659, 0.854324758f},
    {21, 62, 0.338709682f},
    {345, 421, 0.819477439f},
    {720, 727, 0.990371406f},
    {770, 770, 1.0f},
    {0, 0, 0.0f},
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<float,20> kInFov{{
    0.352412969f,
    0.0f,
    0.0476696081f,
    1.0f,
    0.958051562f,
    0.544541895f,
    0.819926679f,
    0.0f,
    0.146450654f,
    0.106684729f,
    0.619071782f,
    0.438299656f,
    0.8768152f,
    0.187863424f,
    0.301279306f,
    0.334326684f,
    0.761795461f,
    0.65635848f,
    0.217304319f,
    1.0f,
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<float,20> kCentral{{
    1.0f,
    0.192041814f,
    0.698853374f,
    0.0702776313f,
    0.124744415f,
    0.256475568f,
    0.420080841f,
    0.150885582f,
    0.0421508551f,
    0.399103343f,
    0.356537342f,
    0.370223045f,
    0.0f,
    0.115897834f,
    0.302245438f,
    0.0f,
    0.818705738f,
    0.962259054f,
    0.0f,
    0.0f,
}};
// golden: generated from the pre-move agent code (door_concept core/door_existence, cabinet_existence), 20 tuples
inline constexpr std::array<Kap,20> kKap{{
    {0.01477515806340509, 7.5f, 1.28725183f, 0.715261281f},
    {0.1268875339812886, 8.5f, 0.285966516f, 0.0751061291f},
    {0.17882640470978969, 8.25f, 1.04828227f, 0.231611982f},
    {0.87193136413892114, 0.75f, 0.83913976f, 0.573031008f},
    {0.083073737289623126, 8.25f, 0.551540196f, 0.18155849f},
    {0.082479125003755829, 4.5f, 0.534012318f, 0.310833305f},
    {0.14199728751555085, 8.0f, 0.727782071f, 0.188468471f},
    {0.17383307402231254, 8.75f, 1.04302931f, 0.22111252f},
    {0.18213666754740257, 6.75f, 0.629033625f, 0.171029642f},
    {0.039685248500770993, 6.0f, 1.24979389f, 0.619976044f},
    {0.059857245350693478, 4.5f, 0.417157888f, 0.287894845f},
    {0.24350628712453132, 2.75f, 0.794246793f, 0.411138356f},
    {0.10552218936156699, 4.75f, 1.37222373f, 0.55875057f},
    {0.14198962319642305, 4.0f, 0.304103941f, 0.158534557f},
    {0.064715476706624031, 4.0f, 0.202375233f, 0.156494558f},
    {0.039681936651468276, 10.0f, 1.19114113f, 0.425833851f},
    {1.7225313037633896, 0.5f, 0.671472192f, 0.487099469f},
    {0.14215594651747723, 1.75f, 0.682890534f, 0.604945064f},
    {1.8446101099252701, 0.25f, 1.57921171f, 0.623285949f},
    {1.036281179138322e-09, 5.25f, 1.20250642f, 0.999942958f},
}};

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    for (int i = 0; i < 20; ++i)
    {
        { const auto& t = kEnv4[i]; const rc::detect::DetectorEnvelope e{t.lo, t.hi, t.soft};
          EXPECT_NEAR("envelope4", i, term_envelope(t.fmax, t.fmin, t.vis, e), t.out); }
        { const auto& t = kEnv3[i]; const rc::detect::DetectorEnvelope e{t.lo, t.hi, t.soft};
          EXPECT_NEAR("envelope3", i, term_envelope(t.fill, t.vis, e), t.out); }
        { const auto& t = kRng[i];  EXPECT_NEAR("range_conf", i, term_range_conf(t.range, t.ref, t.power), t.out); }
        { const auto& t = kObl[i];  EXPECT_NEAR("leaf_obliquity", i, term_leaf_obliquity(t.rx, t.ry, t.nx, t.ny), t.out); }
        { const auto& t = kRes[i];  EXPECT_NEAR("resolvability", i, term_resolvability(t.cells, t.det), t.out); }
        EXPECT_NEAR("in_fov", i, term_in_fov(kInFov[i]), kInFov[i]);
        EXPECT_NEAR("central", i, term_central(kCentral[i]), kCentral[i]);
        { const auto& t = kKap[i];  EXPECT_NEAR("kappa", i, term_kappa(t.var_b, t.range, t.half_diag), t.out); }
    }
    // coverage passes the visible fraction through
    CHECK(term_coverage(0.37f) == 0.37f);

    // eval_term dispatch: by name, parameters from the spec, quantities from the inputs.
    TermInputs in;  in.fill_max = 0.3f; in.fill_min = 0.25f; in.visible = 0.8f; in.in_fov = 0.7f; in.central = 0.6f;
    in.range = 3.0f; in.obliquity = -0.5f; in.resolvability = 0.9f; in.kappa = 0.4f;
    const TermSpec env{"envelope", {{"min_fill", 0.1f}, {"max_fill", 0.6f}, {"soft", 0.06f}}};
    CHECK(near(eval_term(env, in), term_envelope(0.3f, 0.25f, 0.8f, rc::detect::DetectorEnvelope{0.1f, 0.6f, 0.06f})));
    CHECK(near(eval_term({"in_fov", {}}, in), 0.7f));
    CHECK(near(eval_term({"central", {}}, in), 0.6f));
    CHECK(near(eval_term({"range_conf", {{"ref_m", 2.0f}, {"power", 1.0f}}}, in), term_range_conf(3.0f, 2.0f, 1.0f)));
    CHECK(near(eval_term({"leaf_obliquity", {}}, in), 0.5f));
    CHECK(near(eval_term({"resolvability", {}}, in), 0.9f));
    CHECK(near(eval_term({"kappa", {}}, in), 0.4f));
    CHECK(near(eval_term({"coverage", {}}, in), 0.8f));
    CHECK(near(eval_term({"motion", {}}, in), 1.0f));   // Task 3 supplies the real one; neutral until then

    if (bad) { std::printf("terms_test: %d FAILURES\n", bad); return 1; }
    std::printf("terms_test: ok\n");
    return 0;
}
