/*
 * common/thumbs/thumb_wire.h — small JPEG views of objects, as one byte buffer in a graph attribute (compressed_data).
 * key is the slice index on retina's node, the object_uid on an agent's. Pure: no DSR, no Qt.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace rc::thumbs
{
struct Thumb
{
    std::uint64_t key = 0;
    std::int32_t bin = -1;   ///< viewing-angle bin; -1 before anyone has assigned one
    float score = 0.f;
    std::vector<std::uint8_t> jpeg;
};

struct ThumbFrame
{
    std::uint64_t frame = 0;   ///< retina's mask frame these crops were cut from; 0 on an agent's node
    std::vector<Thumb> thumbs;
};

namespace detail
{
inline constexpr std::uint32_t kMagic = 0x31424D54;   // "TMB1"

template <class T> void put(std::vector<std::uint8_t>& out, const T& v)
{
    const auto at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &v, sizeof(T));
}
template <class T> bool get(std::span<const std::uint8_t> in, std::size_t& at, T& v)
{
    if (in.size() - at < sizeof(T)) return false;
    std::memcpy(&v, in.data() + at, sizeof(T));
    at += sizeof(T);
    return true;
}
}   // namespace detail

inline std::vector<std::uint8_t> encode(const ThumbFrame& f)
{
    using detail::put;
    std::vector<std::uint8_t> out;
    put(out, detail::kMagic);
    put(out, f.frame);
    put(out, static_cast<std::uint32_t>(f.thumbs.size()));
    for (const auto& t : f.thumbs)
    {
        put(out, t.key);
        put(out, t.bin);
        put(out, t.score);
        put(out, static_cast<std::uint32_t>(t.jpeg.size()));
        out.insert(out.end(), t.jpeg.begin(), t.jpeg.end());
    }
    return out;
}

/// nullopt on anything that is not exactly one well-formed buffer.
inline std::optional<ThumbFrame> decode(std::span<const std::uint8_t> in)
{
    using detail::get;
    std::size_t at = 0;
    std::uint32_t magic = 0, count = 0;
    ThumbFrame f;
    if (not get(in, at, magic) or magic != detail::kMagic or not get(in, at, f.frame) or not get(in, at, count)) return std::nullopt;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        Thumb t;
        std::uint32_t len = 0;
        if (not get(in, at, t.key) or not get(in, at, t.bin) or not get(in, at, t.score) or not get(in, at, len)) return std::nullopt;
        if (in.size() - at < len) return std::nullopt;
        t.jpeg.assign(in.begin() + static_cast<std::ptrdiff_t>(at), in.begin() + static_cast<std::ptrdiff_t>(at + len));
        at += len;
        f.thumbs.push_back(std::move(t));
    }
    if (at != in.size()) return std::nullopt;
    return f;
}
}   // namespace rc::thumbs
