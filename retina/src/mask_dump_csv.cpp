#include "mask_dump_csv.h"

#include <charconv>
#include <format>
#include <fstream>
#include <locale>
#include <print>
#include <string_view>
#include <system_error>

namespace rc::diag
{

namespace
{
// A label must not break the CSV: the published labels are joined with '|' on DSR, and ',' / newlines
// would split a row here. None of the real class names carry them; this is insurance, not cleaning.
std::string csv_safe(std::string s)
{
    for (auto& c : s)
        if (c == ',' or c == '|' or c == '\n' or c == '\r')
            c = ' ';
    return s;
}

std::vector<std::string_view> split(std::string_view line)
{
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (true)
    {
        const auto pos = line.find(',', start);
        if (pos == std::string_view::npos)
        {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
}

// std::from_chars ONLY (CLAUDE.md): strtof/atof would read "0.25" as 0 under LANG=es_ES once Qt has
// called setlocale(LC_ALL, "").
template <typename T>
bool parse(std::string_view s, T& v)
{
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    return ec == std::errc{} and p == s.data() + s.size();
}

std::optional<rc::MaskSegmenter> parse_segmenter(std::string_view s)
{
    if (s == "yolo")     return rc::MaskSegmenter::yolo;
    if (s == "semantic") return rc::MaskSegmenter::semantic;
    return std::nullopt;
}

bool open_classic(std::ofstream& f, const std::string& path)
{
    f.open(path, std::ios::out | std::ios::trunc);
    if (not f)
        return false;
    f.imbue(std::locale::classic());   // decimal points regardless of LANG (CLAUDE.md)
    return true;
}
} // namespace

std::string mask_dump_path(const std::string& dir, std::uint64_t stamp, const char* suffix)
{
    return std::format("{}/{}.masks{}.csv", dir, stamp, suffix);
}

bool write_mask_dump(const std::string& dir, const MaskDumpRecord& rec)
{
    const auto pix_path = mask_dump_path(dir, rec.stamp_ms, ".pix");
    const auto pts_path = mask_dump_path(dir, rec.stamp_ms, ".pts");
    const auto idx_path = mask_dump_path(dir, rec.stamp_ms);

    // Long-form files first; the index last, so a present index means a complete set.
    {
        std::ofstream pix;
        if (not open_classic(pix, pix_path))
        {
            std::println(stderr, "[zed-dump] could not write {}", pix_path);
            return false;
        }
        pix << "mask,col,row\n";
        for (std::size_t m = 0; m < rec.masks.size(); ++m)
        {
            const auto& px = rec.masks[m].pixels_xy;
            for (std::size_t i = 0; i + 1 < px.size(); i += 2)
                pix << std::format("{},{:.0f},{:.0f}\n", m, px[i], px[i + 1]);
        }
        if (not pix)
            return false;
    }
    {
        std::ofstream pts;
        if (not open_classic(pts, pts_path))
        {
            std::println(stderr, "[zed-dump] could not write {}", pts_path);
            return false;
        }
        pts << "mask,x,y,z\n";
        for (std::size_t m = 0; m < rec.masks.size(); ++m)
        {
            const auto& p = rec.masks[m].points_cam;
            for (std::size_t i = 0; i + 2 < p.size(); i += 3)
                pts << std::format("{},{:.4f},{:.4f},{:.4f}\n", m, p[i], p[i + 1], p[i + 2]);
        }
        if (not pts)
            return false;
    }
    std::ofstream idx;
    if (not open_classic(idx, idx_path))
    {
        std::println(stderr, "[zed-dump] could not write {}", idx_path);
        return false;
    }
    idx << "stamp_ms,frame_id,mask,label,label_id,confidence,segmenter,n_pixels,n_points,pixel_stride,img_w,img_h\n";
    for (std::size_t m = 0; m < rec.masks.size(); ++m)
    {
        const auto& e = rec.masks[m];
        idx << std::format("{},{},{},{},{},{:.4f},{},{},{},{},{},{}\n", rec.stamp_ms, rec.frame_id, m,
                           csv_safe(e.label), e.label_id, e.confidence, rc::to_string(e.segmenter),
                           e.pixels_xy.size() / 2, e.points_cam.size() / 3, rec.pixel_stride,
                           rec.img_w, rec.img_h);
    }
    return static_cast<bool>(idx);
}

std::optional<MaskDumpRecord> read_mask_dump(const std::string& dir, std::uint64_t stamp)
{
    std::ifstream idx(mask_dump_path(dir, stamp));
    if (not idx)
        return std::nullopt;
    MaskDumpRecord rec;
    rec.stamp_ms = stamp;
    std::string line;
    std::getline(idx, line);   // header
    while (std::getline(idx, line))
    {
        if (line.empty())
            continue;
        const auto f = split(line);
        if (f.size() != 12)
            return std::nullopt;
        MaskDumpEntry e;
        std::uint64_t st = 0;
        std::size_t   m = 0;
        if (not (parse(f[0], st) and parse(f[1], rec.frame_id) and parse(f[2], m)
                 and parse(f[4], e.label_id) and parse(f[5], e.confidence)
                 and parse(f[9], rec.pixel_stride) and parse(f[10], rec.img_w) and parse(f[11], rec.img_h)))
            return std::nullopt;
        const auto seg = parse_segmenter(f[6]);
        if (not seg or st != stamp or m != rec.masks.size())
            return std::nullopt;
        e.segmenter = *seg;
        e.label = std::string(f[3]);
        rec.masks.push_back(std::move(e));
    }

    const auto read_long = [&](const char* suffix, std::size_t width, auto member) -> bool
    {
        std::ifstream in(mask_dump_path(dir, stamp, suffix));
        if (not in)
            return false;
        std::string l;
        std::getline(in, l);   // header
        while (std::getline(in, l))
        {
            if (l.empty())
                continue;
            const auto f = split(l);
            std::size_t m = 0;
            if (f.size() != width + 1 or not parse(f[0], m) or m >= rec.masks.size())
                return false;
            for (std::size_t k = 1; k <= width; ++k)
            {
                float v = 0.f;
                if (not parse(f[k], v))
                    return false;
                (rec.masks[m].*member).push_back(v);
            }
        }
        return true;
    };
    if (not read_long(".pix", 2, &MaskDumpEntry::pixels_xy) or not read_long(".pts", 3, &MaskDumpEntry::points_cam))
        return std::nullopt;
    return rec;
}

} // namespace rc::diag
