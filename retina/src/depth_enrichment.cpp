#include "depth_enrichment.h"

#include "room_envelope_depth.h"

#include <dsr/api/dsr_camera_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <locale>
#include <print>
#include <span>

namespace rc::depth
{

namespace
{

// ★LOCALE-INDEPENDENT PARSING (CLAUDE.md). These machines run es_ES.UTF-8 and Qt activates it for the
// C library, so strtof/atof stop at the '.' and every float silently truncates to its integer part.
// std::from_chars is locale independent by definition. Same helper shape as depth_dataset.cpp.
template <typename T>
const char* parse_field(const char* p, const char* end, T& out)
{
    while (p < end and (*p == ' ' or *p == '\t')) ++p;
    const auto r = std::from_chars(p, end, out);
    if (r.ec != std::errc{})
        out = T{};
    p = (r.ec == std::errc{}) ? r.ptr : p;
    while (p < end and *p != ',') ++p;
    return (p < end) ? p + 1 : end;
}

const char* parse_token(const char* p, const char* end, std::string& out)
{
    const char* q = p;
    while (q < end and *q != ',') ++q;
    out.assign(p, q);
    return (q < end) ? q + 1 : end;
}

// ── ADE20K-150 class ids the shell can wear ──────────────────────────────────────────────────────
// The label set is in yolo_semantic.cpp::default_class_names(), in model class-id order.
constexpr int kWall = 0, kFloor = 3, kCeiling = 5, kWindow = 8, kEarth = 13, kDoor = 14,
              kCurtain = 18, kPainting = 22, kMirror = 27, kRug = 28, kScreenDoor = 58,
              kBlind = 63, kPoster = 100;

enum class Shell { Floor, Ceiling, Wall };

// Is the segmenter's winning class one that the envelope surface could legitimately wear? A window,
// a door, a picture or a blind ARE the wall as far as range is concerned (they live in its plane); a
// rug IS the floor. A "chair" over a floor hit is not — that is the case the precision must handle.
bool compatible(Shell k, int label)
{
    switch (k)
    {
        case Shell::Floor:   return label == kFloor or label == kRug or label == kEarth;
        case Shell::Ceiling: return label == kCeiling;
        case Shell::Wall:    return label == kWall or label == kWindow or label == kDoor
                                 or label == kCurtain or label == kPainting or label == kMirror
                                 or label == kScreenDoor or label == kBlind or label == kPoster;
    }
    return false;
}

// ★PRIOR probability that the envelope's own hit is what the pixel really shows, BEFORE the segmenter
// speaks. It is not the same for the three surfaces and that asymmetry is physical, not tuning: a
// ceiling is nearly free (nothing hangs there), a wall is contested by shelves and radiators, a floor
// is contested by every piece of furniture in the room. Used (a) when the segmenter has no confident
// class at all and (b) to spread the non-argmax probability mass over classes it did not name.
constexpr double kPriorCeiling = 0.85, kPriorWall = 0.60, kPriorFloor = 0.50;

double prior_of(Shell k)
{
    switch (k)
    {
        case Shell::Ceiling: return kPriorCeiling;
        case Shell::Wall:    return kPriorWall;
        case Shell::Floor:   return kPriorFloor;
    }
    return 0.5;
}

// Which envelope surface did the ray land on, and at what incidence? This is NOT a second ray-caster:
// the intersection has already been solved by room_envelope_range_equirect(); this only asks which of
// the envelope's surfaces the resulting point lies on (the nearest one, no epsilon needed) and reads
// off that surface's normal so the incidence angle can grade the precision.
struct ShellHit
{
    Shell  kind    = Shell::Wall;
    int    wall    = -1;
    double cos_inc = 0.0;
};

ShellHit classify_hit(const Eigen::Vector3d& P, const Eigen::Vector3d& D,
                      std::span<const float> px, std::span<const float> py, double height)
{
    ShellHit out;
    double best = std::abs(P.z());
    out.kind    = Shell::Floor;
    out.cos_inc = std::abs(D.z());
    if (const double dc = std::abs(P.z() - height); dc < best)
    {
        best        = dc;
        out.kind    = Shell::Ceiling;
        out.cos_inc = std::abs(D.z());
    }
    const std::size_t n = px.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t j = (i + 1) % n;
        const double ax = px[i], ay = py[i];
        const double ex = px[j] - ax, ey = py[j] - ay;
        const double L2 = ex * ex + ey * ey;
        if (L2 < 1e-12)
            continue;
        double s = ((P.x() - ax) * ex + (P.y() - ay) * ey) / L2;
        s = std::clamp(s, 0.0, 1.0);
        const double qx = ax + s * ex - P.x(), qy = ay + s * ey - P.y();
        const double d  = std::sqrt(qx * qx + qy * qy);
        if (d < best)
        {
            best        = d;
            out.kind    = Shell::Wall;
            out.wall    = static_cast<int>(i);
            const double inv = 1.0 / std::sqrt(L2);
            const double nx = -ey * inv, ny = ex * inv;
            out.cos_inc = std::abs(D.x() * nx + D.y() * ny);
        }
    }
    return out;
}

// Circular column pad — a panorama wraps, so a strip at the seam must see across it. Six lines, and
// deliberately local: the depth path has its own copy for its own geometry, and sharing one would
// couple two pipelines that only happen to agree today.
cv::Mat circular_pad(const cv::Mat& src, int ov)
{
    if (ov <= 0)
        return src;
    const int W = src.cols, H = src.rows;
    cv::Mat out(H, W + 2 * ov, src.type());
    src.copyTo(out(cv::Rect(ov, 0, W, H)));
    src(cv::Rect(W - ov, 0, ov, H)).copyTo(out(cv::Rect(0, 0, ov, H)));
    src(cv::Rect(0, 0, ov, H)).copyTo(out(cv::Rect(W + ov, 0, ov, H)));
    return out;
}

}   // namespace

// ─── Room-geometry sidecar ───────────────────────────────────────────────────────────────────────

bool append_room_geometry(const std::string& path, std::uint64_t stamp_ms, const RoomGeometry& g)
{
    if (not g.valid())
        return false;
    const bool fresh = not std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (not f.is_open())
        return false;
    f.imbue(std::locale::classic());   // decimal POINT always — the reader assumes it
    if (fresh)
        f << "# per-frame room belief for the offline depth-enrichment pass (depth_enrichment.h).\n"
             "# The dataset CSV cannot hold this: its rows are per-SAMPLE and a polygon is per-FRAME.\n"
             "# room_T_cam is the FULL 3x4 — (rx,ry,rtheta) in the dataset loses the camera height and\n"
             "# any pitch, and the envelope ray-cast needs all of it.\n"
             "stamp_ms,room,height,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,nverts,x0,y0,...\n";
    f << stamp_ms << ',' << g.room << ',' << g.height;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            f << ',' << g.room_T_cam(r, c);
    f << ',' << g.poly_x.size();
    for (std::size_t i = 0; i < g.poly_x.size(); ++i)
        f << ',' << g.poly_x[i] << ',' << g.poly_y[i];
    f << '\n';
    return true;
}

bool load_room_geometry(const std::string& path, std::map<std::uint64_t, RoomGeometry>& out)
{
    std::ifstream f(path);
    if (not f.is_open())
        return false;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line[0] == '#' or line.starts_with("stamp_ms"))
            continue;
        const char* p   = line.data();
        const char* end = line.data() + line.size();
        unsigned long long stamp = 0;
        RoomGeometry g;
        p = parse_field(p, end, stamp);
        p = parse_token(p, end, g.room);
        p = parse_field(p, end, g.height);
        g.room_T_cam = Mat::RTMat::Identity();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
            {
                double v = 0.0;
                p = parse_field(p, end, v);
                g.room_T_cam(r, c) = v;
            }
        long nv = 0;
        p = parse_field(p, end, nv);
        g.poly_x.reserve(static_cast<std::size_t>(std::max(0L, nv)));
        g.poly_y.reserve(static_cast<std::size_t>(std::max(0L, nv)));
        for (long i = 0; i < nv; ++i)
        {
            float x = 0.f, y = 0.f;
            p = parse_field(p, end, x);
            p = parse_field(p, end, y);
            g.poly_x.push_back(x);
            g.poly_y.push_back(y);
        }
        g.cam_z = static_cast<float>(g.room_T_cam(2, 3));
        if (g.valid() and stamp != 0)
            out[static_cast<std::uint64_t>(stamp)] = std::move(g);
    }
    return true;
}

// ─── DatasetEnricher ─────────────────────────────────────────────────────────────────────────────

const char* DatasetEnricher::phase_name(Phase p)
{
    switch (p)
    {
        case Phase::Idle:      return "idle";
        case Phase::Loading:   return "loading dataset";
        case Phase::Parity:    return "checking model parity";
        case Phase::Sessions:  return "loading models";
        case Phase::Enriching: return "enriching";
        case Phase::Fitting:   return "fitting";
        case Phase::Done:      return "done";
        case Phase::Failed:    return "failed";
    }
    return "?";
}

DatasetEnricher::DatasetEnricher(EnrichConfig cfg) : cfg_(std::move(cfg)) {}

DatasetEnricher::~DatasetEnricher()
{
    cancel_.store(true, std::memory_order_release);
    if (worker_.joinable())
        worker_.join();
}

void DatasetEnricher::bind_camera(std::unique_ptr<DSR::CameraAPI> cam) { cam_ = std::move(cam); }

void DatasetEnricher::set_geometry(std::map<std::uint64_t, RoomGeometry> per_frame,
                                   RoomGeometry legacy_fallback)
{
    geom_     = std::move(per_frame);
    fallback_ = std::move(legacy_fallback);
}

DatasetEnricher::Progress DatasetEnricher::progress() const
{
    return Progress{static_cast<Phase>(phase_.load(std::memory_order_acquire)),
                    done_.load(std::memory_order_acquire),
                    total_.load(std::memory_order_acquire),
                    synth_.load(std::memory_order_acquire)};
}

bool DatasetEnricher::start()
{
    if (running_.load(std::memory_order_acquire) or worker_.joinable())
        return false;
    cancel_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    report_ = Report{};
    worker_ = std::thread([this] { work(); running_.store(false, std::memory_order_release); });
    return true;
}

DatasetEnricher::Report DatasetEnricher::join()
{
    if (worker_.joinable())
        worker_.join();
    return report_;
}

DatasetEnricher::Report DatasetEnricher::run()
{
    work();
    return report_;
}

bool DatasetEnricher::open_sessions()
{
    // ★OUR OWN sessions. DepthStage's and SemanticStage's belong to the ricoh worker thread and are
    // being driven by it right now; borrowing either would put two threads through one Ort::Session
    // and through one processor's cached remap tables.
    phase_.store(static_cast<int>(Phase::Sessions), std::memory_order_release);
    depth_ = std::make_unique<DepthProcessor>();
    depth_->configure(cfg_.depth_cfg);
    if (not depth_->ready())
    {
        report_.error = std::format("depth model {} did not load", cfg_.depth_cfg.model_path);
        return false;
    }
    sem_ = std::make_unique<rc::semantic::YoloSemanticProcessor>();
    sem_->configure(cfg_.sem_cfg);
    if (not sem_->ready())
    {
        report_.error = std::format("semantic model {} did not load", cfg_.sem_cfg.model_path);
        return false;
    }
    return true;
}

cv::Mat DatasetEnricher::load_frame_image(std::uint64_t stamp_ms) const
{
    const auto path = std::format("{}/{}.jpg", cfg_.frames_dir, stamp_ms);
    std::error_code ec;
    if (not std::filesystem::exists(path, ec))
        return {};
    return cv::imread(path, cv::IMREAD_COLOR);   // imread always allocates — nothing shared to clone
}

RoomGeometry DatasetEnricher::geometry_for(const DepthFrame& fr, bool& legacy) const
{
    legacy = false;
    if (const auto it = geom_.find(fr.stamp_ms); it != geom_.end() and it->second.valid())
        return it->second;
    // ★LEGACY FALLBACK. The frame predates the sidecar, so the ONLY source of a polygon is the graph
    // as it is right now. Reconstruct the pose from what the dataset does carry — (rx,ry) is the
    // ricoh's position and rtheta the yaw of room_T_ricoh — plus the camera height from the live
    // mount. The rotation is exact for this robot (the body→ricoh RT edge has zero euler angles, so
    // room_T_ricoh's linear part IS the robot's yaw); the height is the assumption.
    legacy = true;
    RoomGeometry g = fallback_;
    if (not g.valid())
        return g;
    g.reconstructed = true;
    Mat::RTMat T = Mat::RTMat::Identity();
    T.linear() = Eigen::AngleAxisd(static_cast<double>(fr.rtheta), Eigen::Vector3d::UnitZ())
                     .toRotationMatrix();
    T.translation() = Eigen::Vector3d(fr.rx, fr.ry, g.cam_z);
    g.room_T_cam = T;
    return g;
}

rc::semantic::SemanticMap DatasetEnricher::segment_360(const cv::Mat& panorama_bgr) const
{
    rc::semantic::SemanticMap out;
    if (not sem_ or not sem_->ready() or panorama_bgr.empty())
        return out;
    const int W = panorama_bgr.cols, H = panorama_bgr.rows;
    const int n = std::max(1, cfg_.n_views);
    const int strip_w = W / n;
    if (strip_w <= 0)
        return out;
    // Same strip decomposition (and the same circular context margin) the depth path uses, so the two
    // fields are pixel-aligned by construction rather than by a resize that happens to agree. A
    // full-height 320-column strip letterboxes into the model's square input with NO horizontal
    // squash, which a whole 1920x960 panorama in one pass would not.
    const int ov = std::clamp(cfg_.depth360.overlap_px, 0, strip_w / 2);

    // The segmenter wants RGB (YoloSemanticProcessor::segment forces is_rgb=true); imread gives BGR.
    cv::Mat rgb;
    cv::cvtColor(panorama_bgr, rgb, cv::COLOR_BGR2RGB);
    const cv::Mat padded = circular_pad(rgb, ov);

    out.labels = cv::Mat(H, W, CV_8UC1,  cv::Scalar(rc::semantic::IGNORE_LABEL));
    out.scores = cv::Mat(H, W, CV_32FC1, cv::Scalar(0.f));
    for (int s = 0; s < n; ++s)
    {
        const int x0 = s * strip_w;
        if (x0 + strip_w + 2 * ov > padded.cols)
            continue;
        // DEEP COPY of the crop before inference: a cv::Mat ROI is a handle into `padded`, and the
        // model path is entitled to assume a contiguous owned buffer.
        const cv::Mat strip = padded(cv::Rect(x0, 0, strip_w + 2 * ov, H)).clone();
        const auto m = sem_->segment(strip, /*want_scores=*/true);
        if (m.labels.empty() or m.labels.size() != strip.size())
            continue;
        m.labels(cv::Rect(ov, 0, strip_w, H)).copyTo(out.labels(cv::Rect(x0, 0, strip_w, H)));
        if (not m.scores.empty() and m.scores.size() == strip.size())
            m.scores(cv::Rect(ov, 0, strip_w, H)).copyTo(out.scores(cv::Rect(x0, 0, strip_w, H)));
    }
    return out;
}

// ─── (2) Is the recomputed field the SAME field the dataset was collected from? ──────────────────

DatasetEnricher::Parity DatasetEnricher::check_model_parity(const DepthDataset& ds)
{
    phase_.store(static_cast<int>(Phase::Parity), std::memory_order_release);
    return measure_model_parity(ds, *depth_, cfg_, &cancel_, &done_);
}

DatasetEnricher::Parity DatasetEnricher::measure_model_parity(const DepthDataset& ds,
                                                              DepthProcessor& depth,
                                                              const EnrichConfig& cfg,
                                                              const std::atomic<bool>* cancel,
                                                              std::atomic<int>* frames_done)
{
    Parity out;
    std::vector<double> diff, diff_reg, offsets;
    diff.reserve(1 << 16);
    diff_reg.reserve(1 << 16);

    for (std::size_t i = 0; i < ds.frame_count() and out.frames < cfg.parity_frames; ++i)
    {
        // Signed per-view differences for THIS frame, so the registerable part can be separated from
        // the part that is not.
        std::array<std::vector<double>, kMaxViews> per_view;
        if (cancel != nullptr and cancel->load(std::memory_order_acquire))
            break;
        const DepthFrame& fr = ds.frame(i);
        const auto path = std::format("{}/{}.jpg", cfg.frames_dir, fr.stamp_ms);
        std::error_code ec;
        if (not std::filesystem::exists(path, ec))
            continue;
        const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty())
            continue;
        const DepthMap dm = depth.estimate_360(img, cfg.depth360);
        if (dm.empty() or dm.log_depth.size() != img.size())
            continue;
        ++out.frames;
        const int W = dm.log_depth.cols, H = dm.log_depth.rows;
        const int strip_w = (dm.n_strips > 0) ? W / dm.n_strips : W;
        if (strip_w <= 0)
            continue;
        for (const auto& s : fr.samples)
        {
            if (s.src != kSrcLidar)
                continue;
            // ★A SATURATED COORDINATE IS NOT INVERTIBLE. Both s and t were std::clamp'ed to [-1,1]
            // at collection time, so a sample sitting on either limit could have come from any pixel
            // beyond it. Excluded and COUNTED, never silently folded in.
            if (std::abs(s.s) >= 1.f or std::abs(s.t) >= 1.f)
            {
                ++out.n_clamped;
                continue;
            }
            // Exact inverse of the push site (specificworker.cpp, log_ricoh_depth_lidar_correlation):
            //   s = (u − (view+0.5)·strip_w) / (0.5·strip_w)      t = (v − H/2) / (H/2)
            const int u = static_cast<int>(std::lround((s.view + 0.5) * strip_w
                                                       + 0.5 * strip_w * static_cast<double>(s.s)));
            const int v = static_cast<int>(std::lround(0.5 * H * (1.0 + static_cast<double>(s.t))));
            if (u < 0 or u >= W or v < 0 or v >= H)
                continue;
            const float lm = dm.log_depth.at<float>(v, u);
            if (not std::isfinite(lm) or not std::isfinite(s.log_model))
                continue;
            // A pixel whose strip ownership changed is a geometry mismatch, not model noise — it means
            // n_strips/overlap differ from the collecting run, which the caller must know about.
            if (dm.strip_id.at<unsigned char>(v, u) != s.view)
                continue;
            const double d = static_cast<double>(lm) - static_cast<double>(s.log_model);
            diff.push_back(std::abs(d));
            per_view[s.view].push_back(d);
        }
        // Per-(frame,view) offset = the median signed difference. THE SAME quantity enrich_frame
        // solves before it emits anything, so what is measured here is what will actually be mixed.
        for (auto& v : per_view)
        {
            if (v.size() < 32)
                continue;
            std::ranges::nth_element(v, v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2));
            const double off = v[v.size() / 2];
            offsets.push_back(std::abs(off));
            ++out.views_registered;
            for (const double d : v)
                diff_reg.push_back(std::abs(d - off));
        }
        if (frames_done != nullptr)
            frames_done->store(out.frames, std::memory_order_release);
    }

    const auto med = [](std::vector<double>& v)
    { std::ranges::sort(v); return v[v.size() / 2]; };

    out.n = static_cast<long>(diff.size());
    if (out.n > 8)
    {
        out.med_abs_log = med(diff);
        out.p95_abs_log = diff[static_cast<std::size_t>(0.95 * (diff.size() - 1))];
        double ss = 0.0;
        for (const double d : diff) ss += d * d;
        out.rms_abs_log = std::sqrt(ss / static_cast<double>(diff.size()));
    }
    if (diff_reg.size() > 8)
        out.med_abs_log_registered = med(diff_reg);
    if (not offsets.empty())
        out.med_offset = med(offsets);
    // ★The verdict is on the REGISTERED number. A per-image offset is a documented property of this
    // model (depth_processor.h: strips of the SAME content disagree by 2-3x), it is measurable, and
    // enrich_frame removes it. A residual in SHAPE is none of those things.
    out.ok = (out.med_abs_log_registered >= 0.0 and out.med_abs_log_registered <= cfg.parity_max_med_log);
    return out;
}

// ─── The per-frame enrichment ───────────────────────────────────────────────────────────────────

std::vector<DepthSample> DatasetEnricher::enrich_frame(const DepthFrame& fr, const RoomGeometry& g)
{
    std::vector<DepthSample> out;
    if (not cam_ or not g.valid())
        return out;
    const cv::Mat img = load_frame_image(fr.stamp_ms);
    if (img.empty())
        return out;

    const DepthMap dm = depth_->estimate_360(img, cfg_.depth360);
    if (dm.empty() or dm.log_depth.size() != img.size())
        return out;
    const auto sem = segment_360(img);
    if (sem.labels.empty() or sem.labels.size() != img.size())
        return out;

    const int W = dm.log_depth.cols, H = dm.log_depth.rows;
    const cv::Mat env = room_envelope_range_equirect(*cam_, W, H, g.room_T_cam,
                                                     g.poly_x, g.poly_y, g.height,
                                                     cfg_.envelope_decimate);
    if (env.empty() or env.size() != dm.log_depth.size())
        return out;

    const int strip_w = (dm.n_strips > 0) ? W / dm.n_strips : W;
    if (strip_w <= 0)
        return out;

    // ── ★PER-(FRAME,VIEW) SCALE REGISTRATION — without this the whole pass is nonsense ───────────
    // yolo26l-depth is scale-and-shift invariant PER INPUT IMAGE, and the panorama on disk is a JPEG,
    // not the buffer the model originally saw. Re-running it therefore re-draws the arbitrary offset:
    // MEASURED 0.325 in log, i.e. 38% in depth (see EnrichConfig::parity_max_med_log). A synthetic
    // row carrying the NEW offset and a measured row carrying the OLD one, both fed to the same
    // nonlinear response in log_model, would be two different experiments — and `b` cannot absorb it,
    // because a shift of the INPUT is not a shift of the output once the spline is nonlinear.
    // The offset is measurable from data we already hold: the median of (stored − recomputed)
    // log_model at the very pixels the LiDAR rows came from. Exactly the anchor's own logic, applied
    // to the model's input instead of its output. A view with too few measured rows cannot be
    // registered and emits NOTHING, which is the honest outcome.
    std::array<std::vector<float>, kMaxViews> dv;
    for (const auto& s : fr.samples)
    {
        if (s.src != kSrcLidar or s.view >= kMaxViews)
            continue;
        if (std::abs(s.s) >= 1.f or std::abs(s.t) >= 1.f)
            continue;                       // clamped ⇒ (u,v) is not invertible
        const int u = static_cast<int>(std::lround((s.view + 0.5) * strip_w
                                                   + 0.5 * strip_w * static_cast<double>(s.s)));
        const int v = static_cast<int>(std::lround(0.5 * H * (1.0 + static_cast<double>(s.t))));
        if (u < 0 or u >= W or v < 0 or v >= H)
            continue;
        const float lm = dm.log_depth.at<float>(v, u);
        if (not std::isfinite(lm) or dm.strip_id.at<unsigned char>(v, u) != s.view)
            continue;
        dv[s.view].push_back(s.log_model - lm);
    }
    std::array<float, kMaxViews> off{};
    std::array<bool,  kMaxViews> off_ok{};
    for (std::size_t v = 0; v < kMaxViews; ++v)
    {
        auto& d = dv[v];
        if (d.size() < static_cast<std::size_t>(std::max(1, cfg_.min_register_samples)))
            continue;
        std::ranges::nth_element(d, d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2));
        off[v]    = d[d.size() / 2];
        off_ok[v] = true;
    }
    for (int v = 0; v < std::min(dm.n_strips, kMaxViews); ++v)
        if (not off_ok[static_cast<std::size_t>(v)])
            ++report_.views_unregistered;

    const int stride = std::max(1, cfg_.pixel_stride);
    const Eigen::Matrix3d R = g.room_T_cam.linear();
    const Eigen::Vector3d o = g.room_T_cam.translation();

    // ── The generative model of a synthetic row's error (see the header, point 5) ────────────────
    // Everything below is expressed RELATIVE to a LiDAR row, whose precision is 1.0 by definition.
    const double sl2 = static_cast<double>(cfg_.sigma_lidar_log) * cfg_.sigma_lidar_log;
    const double spix2 = static_cast<double>(cfg_.sigma_pixel_log) * cfg_.sigma_pixel_log;

    out.reserve(static_cast<std::size_t>((W / stride) * (H / stride) / 2));
    for (int v = 0; v < H; v += stride)
    {
        const auto* lrow = dm.log_depth.ptr<float>(v);
        const auto* srow = dm.strip_id.ptr<unsigned char>(v);
        const auto* erow = env.ptr<float>(v);
        const auto* nrow = sem.labels.ptr<unsigned char>(v);
        const auto* crow = sem.scores.empty() ? nullptr : sem.scores.ptr<float>(v);
        const float t = std::clamp((v - 0.5f * H) / (0.5f * H), -1.f, 1.f);
        for (int u = 0; u < W; u += stride)
        {
            const float lm = lrow[u];
            const float rr = erow[u];
            if (not std::isfinite(lm) or not std::isfinite(rr) or rr <= cfg_.min_range_m)
                continue;
            const unsigned char view = srow[u];
            if (view >= kMaxViews or not off_ok[view])
                continue;                    // unregisterable view — see the registration block
            const float lm_reg = lm + off[view];

            const Eigen::Vector3d D = R * cam_->ray_from_pixel(static_cast<double>(u),
                                                               static_cast<double>(v));
            const double Dn = D.norm();
            if (not (Dn > 1e-9))
                continue;
            const Eigen::Vector3d Du = D / Dn;
            const Eigen::Vector3d P  = o + static_cast<double>(rr) * Du;
            const ShellHit hit = classify_hit(P, Du, g.poly_x, g.poly_y, g.height);

            // ── p = P(this pixel really shows the surface the envelope predicts) ────────────────
            // A mixture, not a gate. `score` is the segmenter's own argmax probability; the leftover
            // mass (1−score) sits on classes it did not name, of which a prior fraction is still the
            // shell. When the winning class AGREES the two add; when it disagrees only the leftover
            // can be shell; when there is no confident class the prior is all we have.
            const double prior = prior_of(hit.kind);
            const int    label = static_cast<int>(nrow[u]);
            const double score = crow ? std::clamp(static_cast<double>(crow[u]), 0.0, 1.0) : 0.0;
            double p = prior;
            if (label != static_cast<int>(rc::semantic::IGNORE_LABEL))
                p = compatible(hit.kind, label) ? score + (1.0 - score) * prior
                                                : (1.0 - score) * prior;
            p = std::clamp(p, 1e-3, 1.0);

            // ── The three variance channels ────────────────────────────────────────────────────
            const double cos_inc = std::clamp(hit.cos_inc, 1e-4, 1.0);
            const double tan_inc = std::sqrt(std::max(0.0, 1.0 - cos_inc * cos_inc)) / cos_inc;
            // INDEPENDENT: pixel-level registration on a slanted surface. Landing one pixel off moves
            // the range by R·tanθ·δα, so δlnR = tanθ·δα — it grows with obliquity all by itself.
            // ★+ the MEASURED error-in-variables term. The re-run's log_model is not the stored one:
            // after registration it still differs by the parity check's residual, and that error
            // reaches the response through the slope a. model_var_log2_ = (a·σ_registered)², filled
            // in by work() from the measurement and from the LiDAR-only refit's own a.
            const double sig_ind2 = spix2 + std::pow(cfg_.pixel_align_rad * tan_inc, 2.0)
                                  + model_var_log2_;
            // COMMON: the believed plane's own normal-direction σ, shared by every pixel on it. A
            // displacement δn along the normal moves the hit along the ray by δn/cosθ ⇒ δlnR =
            // σ_n/(R·cosθ). This is where a GRAZING ray fades out — continuously, with no gate.
            const double sig_n = (hit.kind == Shell::Ceiling) ? cfg_.sigma_ceiling_m
                               : (hit.kind == Shell::Floor)   ? cfg_.sigma_floor_m
                                                              : cfg_.sigma_wall_m;
            const double sig_com = sig_n / (static_cast<double>(rr) * cos_inc);
            // OUTLIER: if this pixel is NOT the shell, whatever is there can only be NEARER (the
            // envelope is an upper bound on range), by at most the class's stand-off. In log range
            // that is ln(1 + d/R), so the same absolute clutter costs less at long range — which is
            // precisely why the far wall down a corridor is the sample worth having.
            const double d_out = (hit.kind == Shell::Ceiling) ? cfg_.standoff_ceiling_m
                               : (hit.kind == Shell::Floor)   ? cfg_.standoff_floor_m
                                                              : cfg_.standoff_wall_m;
            const double sig_out = std::log1p(d_out / static_cast<double>(rr));

            // Split the mixture into the part fit() can treat as independent and the part it must
            // marginalise as a shared mode. The outlier component is independent per pixel (furniture
            // varies), the plane displacement is not.
            const double var_ind = p * sig_ind2 + (1.0 - p) * sig_out * sig_out;
            if (not (var_ind > 0.0))
                continue;
            const double w = sl2 / var_ind;
            const double h = std::sqrt(p) * sig_com / cfg_.sigma_lidar_log;
            if (not std::isfinite(w) or not std::isfinite(h) or not (w > 0.0))
                continue;

            DepthSample smp;
            smp.log_model = lm_reg;
            smp.log_range = std::log(static_cast<double>(rr));
            smp.view      = view;
            smp.s = std::clamp((u - (view + 0.5f) * strip_w) / (0.5f * strip_w), -1.f, 1.f);
            smp.t = t;
            smp.src = kSrcEnvelope;
            // Common-mode group: ONE per envelope SURFACE, because the shared unknown is that
            // surface's position. Frame-local, so ids need only be unique within the frame.
            smp.region = static_cast<std::uint16_t>(
                hit.kind == Shell::Floor   ? 1
              : hit.kind == Shell::Ceiling ? 2
                                           : 3 + std::min(hit.wall < 0 ? 0 : hit.wall, 250));
            smp.w = static_cast<float>(w);
            smp.h = static_cast<float>(h);
            out.push_back(smp);
        }
    }
    return out;
}

// ─── The pass ────────────────────────────────────────────────────────────────────────────────────

void DatasetEnricher::work()
{
    const auto fail = [this](std::string why)
    {
        report_.ok    = false;
        report_.error = std::move(why);
        phase_.store(static_cast<int>(Phase::Failed), std::memory_order_release);
        std::println("[depth-enrich] FAILED — {}", report_.error);
    };

    phase_.store(static_cast<int>(Phase::Loading), std::memory_order_release);
    DepthDataset ds;
    if (not ds.load_csv(cfg_.dataset_csv))
        return fail(std::format("cannot open {}",
                                std::filesystem::absolute(cfg_.dataset_csv).string()));
    if (ds.frame_count() == 0)
        return fail("dataset has no frames");
    report_.frames_total = static_cast<int>(ds.frame_count());
    total_.store(report_.frames_total, std::memory_order_release);
    if (not cam_)
        return fail("no CameraAPI bound — the envelope ray-cast needs the ricoh's projection model");
    if (not open_sessions())
        return fail(report_.error);

    // ── (2) parity, BEFORE anything is mixed ────────────────────────────────────────────────────
    total_.store(std::min(cfg_.parity_frames, report_.frames_total), std::memory_order_release);
    report_.parity = check_model_parity(ds);
    const auto& pa = report_.parity;
    if (pa.n <= 0)
        std::println("[depth-enrich] ⚠ PARITY UNMEASURABLE — no stored sample could be re-located in a "
                     "recomputed frame ({} frames re-run, {} samples excluded as clamped). Either the "
                     "panoramas are missing or the strip geometry changed.", pa.frames, pa.n_clamped);
    else
        std::println("[depth-enrich] MODEL PARITY over {} samples in {} frames ({} clamped, excluded)\n"
                     "               RAW        median |Δlog_model| = {:.4f} ({:.1f}% depth), p95 {:.4f}\n"
                     "               of which a per-(frame,view) OFFSET of {:.4f} — the model re-drawing\n"
                     "               its arbitrary per-image scale on a JPEG it had not seen before\n"
                     "               REGISTERED median |Δlog_model| = {:.4f} ({:.1f}% depth) over {} views\n"
                     "               {}",
                     pa.n, pa.frames, pa.n_clamped,
                     pa.med_abs_log, 100.0 * (std::exp(pa.med_abs_log) - 1.0), pa.p95_abs_log,
                     pa.med_offset, pa.med_abs_log_registered,
                     100.0 * (std::exp(pa.med_abs_log_registered) - 1.0), pa.views_registered,
                     pa.ok ? "SAME FIELD once registered — safe to mix." : "★NOT THE SAME FIELD.");
    if (not pa.ok and cfg_.parity_abort)
        return fail(std::format(
            "model parity FAILED (registered median |Δlog_model| = {:.4f} > {:.4f}). The re-run does NOT "
            "reproduce the SHAPE of the field the dataset was collected from — the per-image offset is "
            "removable and was removed, this is what is left. Synthetic rows would be paired with a "
            "DIFFERENT log_model than the measured ones. Check model_path / n_strips / gnomonic / "
            "band_half_elev_deg / overlap_px / zdepth_to_range against the collecting run before "
            "enriching.", pa.med_abs_log_registered, cfg_.parity_max_med_log));

    // ── The LiDAR-only refit FIRST: it is both the A/B baseline and where the error-in-variables
    // term gets its slope. The parity residual σ_reg is an error in log_model; it reaches the fit's
    // response through a, so the variance it adds to every synthetic row is (a·σ_reg)² — measured,
    // not assumed. Doing this before the pass is what lets the weights use it.
    phase_.store(static_cast<int>(Phase::Fitting), std::memory_order_release);
    report_.map_measured = ds.fit(cfg_.n_views, /*measured_only=*/true);
    if (pa.med_abs_log_registered > 0.0 and report_.map_measured.valid)
        model_var_log2_ = std::pow(static_cast<double>(report_.map_measured.a)
                                       * pa.med_abs_log_registered, 2.0);
    std::println("[depth-enrich] error-in-variables: a={:.3f} x σ_registered={:.4f} ⇒ σ={:.4f} log "
                 "({:.1f}% depth) added to every synthetic row's independent variance",
                 report_.map_measured.a, pa.med_abs_log_registered, std::sqrt(model_var_log2_),
                 100.0 * (std::exp(std::sqrt(model_var_log2_)) - 1.0));

    // ── the pass ────────────────────────────────────────────────────────────────────────────────
    phase_.store(static_cast<int>(Phase::Enriching), std::memory_order_release);
    done_.store(0, std::memory_order_release);
    total_.store(report_.frames_total, std::memory_order_release);
    float rlo = 1e9f, rhi = -1e9f, tlo = 1e9f, thi = -1e9f;
    for (std::size_t i = 0; i < ds.frame_count(); ++i)
    {
        if (cancel_.load(std::memory_order_acquire))
            return fail("cancelled");
        const DepthFrame& fr = ds.frame(i);
        bool legacy = false;
        const RoomGeometry g = geometry_for(fr, legacy);
        if (not g.valid())
        {
            ++report_.frames_no_geometry;
            done_.store(static_cast<int>(i + 1), std::memory_order_release);
            continue;
        }
        if (legacy)
            ++report_.frames_legacy_geom;
        // Distinguish "no panorama on disk" from "the panorama contributed nothing" — they are
        // different failures and lumping them made the log line lie about which one happened.
        std::error_code ec;
        if (not std::filesystem::exists(std::format("{}/{}.jpg", cfg_.frames_dir, fr.stamp_ms), ec))
        {
            ++report_.frames_no_image;
            done_.store(static_cast<int>(i + 1), std::memory_order_release);
            continue;
        }
        const auto extra = enrich_frame(fr, g);
        if (extra.empty())
        {
            done_.store(static_cast<int>(i + 1), std::memory_order_release);
            continue;
        }
        for (const auto& s : extra)
        {
            const float r = std::exp(s.log_range);
            rlo = std::min(rlo, r); rhi = std::max(rhi, r);
            tlo = std::min(tlo, s.t); thi = std::max(thi, s.t);
            report_.synth_weight_sum += static_cast<double>(s.w);
        }
        ds.add_samples_to_frame(fr.stamp_ms, extra);
        report_.n_synth += static_cast<long>(extra.size());
        ++report_.frames_enriched;
        synth_.store(report_.n_synth, std::memory_order_release);
        done_.store(static_cast<int>(i + 1), std::memory_order_release);
    }

    if (report_.frames_legacy_geom > 0)
        std::println("[depth-enrich] ★ASSUMPTION: {} of {} frames had NO recorded room geometry "
                     "(collected before the '{}' sidecar existed). They were ray-cast against the room "
                     "'{}' AS THE GRAPH HOLDS IT NOW, with the camera height taken from the live mount "
                     "({:.3f} m) and the orientation reconstructed from the stored yaw. If those frames "
                     "were taken in a DIFFERENT room, or room_concept's belief has moved since, their "
                     "synthetic rows are wrong. New collection writes the sidecar and does not assume.",
                     report_.frames_legacy_geom, report_.frames_total, cfg_.geometry_csv,
                     fallback_.room, fallback_.cam_z);
    if (report_.n_synth == 0)
        return fail("no synthetic samples produced — check the frames dir and the room geometry");

    report_.synth_range_lo = rlo; report_.synth_range_hi = rhi;
    report_.synth_t_lo = tlo;     report_.synth_t_hi = thi;

    // ── refit, both ways ────────────────────────────────────────────────────────────────────────
    phase_.store(static_cast<int>(Phase::Fitting), std::memory_order_release);
    report_.map_enriched = ds.fit(cfg_.n_views, /*measured_only=*/false);
    if (not report_.map_enriched.valid)
        return fail("the enriched fit did not converge");
    if (not cfg_.out_dataset.empty() and not ds.save_csv(cfg_.out_dataset))
        std::println("[depth-enrich] could not write {}", cfg_.out_dataset);

    const auto& mm = report_.map_measured;
    const auto& me = report_.map_enriched;
    std::println("[depth-enrich] {} synthetic rows from {} frames (Σw = {:.0f} LiDAR-row equivalents; "
                 "{} (frame,view) pairs emitted nothing — too few measured rows to register)\n"
                 "               synthetic range {:.2f}..{:.2f} m, band t {:+.3f}..{:+.3f}\n"
                 "               LiDAR ONLY : a={:.3f} cs={:+.3f} ct={:+.3f}{} resid_anch={:.3f} "
                 "err={:.0f}% δ1={:.0f}% range {:.2f}..{:.2f} m\n"
                 "               ENRICHED   : a={:.3f} cs={:+.3f} ct={:+.3f}{} resid_anch={:.3f} "
                 "err={:.0f}% δ1={:.0f}% range {:.2f}..{:.2f} m\n"
                 "               (both residuals are scored on the MEASURED rows only — the synthetic "
                 "rows are what the map was helped by, not what it is judged against)",
                 report_.n_synth, report_.frames_enriched, report_.synth_weight_sum,
                 report_.views_unregistered,
                 report_.synth_range_lo, report_.synth_range_hi, report_.synth_t_lo, report_.synth_t_hi,
                 mm.a, mm.cs, mm.ct, mm.ct_active ? "" : " (off)", mm.resid_anchored,
                 100.0 * mm.med_rel, 100.0 * mm.delta125, mm.range_lo, mm.range_hi,
                 me.a, me.cs, me.ct, me.ct_active ? "" : " (off)", me.resid_anchored,
                 100.0 * me.med_rel, 100.0 * me.delta125, me.range_lo, me.range_hi);

    report_.ok = true;
    phase_.store(static_cast<int>(Phase::Done), std::memory_order_release);
}

}   // namespace rc::depth
