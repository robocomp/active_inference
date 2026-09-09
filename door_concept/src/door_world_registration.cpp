#include "door_world_registration.h"

#include <charconv>
#include <optional>
#include <cmath>
#include <fstream>
#include <locale>
#include <print>
#include <sstream>

namespace rc
{

namespace
{
// ★from_chars, never strtof/stof. These machines run LANG=es_ES.UTF-8 and Qt calls setlocale(LC_ALL,"")
// at startup, so the C library's decimal separator is a COMMA while std::ofstream writes POINTS through
// the C++ global locale. strtof would silently truncate "-4.769" to -4 and the registration would be
// metres wrong with no error anywhere (CLAUDE.md).
std::optional<float> parse_float(std::string_view s)
{
    float v{};
    const auto* b = s.data();
    const auto* e = s.data() + s.size();
    while (b < e and (*b == ' ' or *b == '\t')) ++b;
    if (const auto r = std::from_chars(b, e, v); r.ec == std::errc{})
        return v;
    return std::nullopt;
}
}   // namespace

void DoorWorldRegistration::load(const std::string& path)
{
    pairs_.clear();
    std::ifstream f(path);
    if (not f)
        return;
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line))
    {
        std::stringstream ss(line);
        std::string our, their, rx, ry, wx, wy;
        if (not (std::getline(ss, our, ',') and std::getline(ss, their, ',')
                 and std::getline(ss, rx, ',') and std::getline(ss, ry, ',')
                 and std::getline(ss, wx, ',') and std::getline(ss, wy, ',')))
            continue;
        const auto a = parse_float(rx), b = parse_float(ry), c = parse_float(wx), d = parse_float(wy);
        if (not (a and b and c and d))
            continue;
        Pair pr{{*a, *b}, {*c, *d}, our, their};
        std::string yr, yw;   // optional: absent in files written before the yaws were persisted
        if (std::getline(ss, yr, ',') and std::getline(ss, yw, ','))
        {
            if (const auto e = parse_float(yr)) pr.yaw_room  = *e;
            if (const auto f = parse_float(yw)) pr.yaw_world = *f;
        }
        pairs_.push_back(pr);
    }
    std::println("door_concept: [registration] loaded {} door correspondence(s) from {}",
                 pairs_.size(), path);
}

void DoorWorldRegistration::add(const Pair& p, const std::string& path)
{
    std::erase_if(pairs_, [&](const Pair& q){ return q.their_id == p.their_id; });
    pairs_.push_back(p);

    // Rewritten whole: the file is a handful of lines and a full rewrite cannot leave a stale duplicate
    // of a door that has just been re-identified.
    std::ofstream f(path, std::ios::trunc);
    if (not f)
        return;
    f.imbue(std::locale::classic());   // never emit a comma decimal separator
    f << "our_name,their_id,room_x,room_y,world_x,world_y,yaw_room,yaw_world\n";
    for (const auto& q : pairs_)
        f << q.our_name << ',' << q.their_id << ',' << q.room_m.x() << ',' << q.room_m.y() << ','
          << q.world_m.x() << ',' << q.world_m.y() << ',' << q.yaw_room << ',' << q.yaw_world << '\n';
}

const DoorWorldRegistration::Fit& DoorWorldRegistration::solve()
{
    fit_ = Fit{};
    fit_.n_pairs = static_cast<int>(pairs_.size());
    if (pairs_.empty())
        return fit_;

    if (pairs_.size() == 1)
    {
        // ★PROVISIONAL. Position is pinned by the pair; rotation can only come from comparing the two
        // yaws, and the two sides do not necessarily mean the same thing by "the door's angle" — ours is
        // the aperture normal, the provider's is a Webots rotation field. Usable, flagged, and replaced
        // by a measured fit the moment a second door is identified.
        const auto& p = pairs_.front();
        if (not std::isfinite(p.yaw_room) or not std::isfinite(p.yaw_world))
        {
            // No yaws recorded ⇒ rotation is unknown ⇒ there is NO usable transform. Assuming zero
            // rotation here is what produced a fit that was metres wrong and looked plausible.
            std::println("door_concept: [registration] 1 pair but no yaws recorded — rotation unknown, "
                         "fit unusable. Identify a SECOND door by name to measure it.");
            return fit_;
        }
        fit_.theta = p.yaw_world - p.yaw_room;
        const float c = std::cos(fit_.theta), s = std::sin(fit_.theta);
        const Eigen::Vector2f Rp(c * p.room_m.x() - s * p.room_m.y(),
                                 s * p.room_m.x() + c * p.room_m.y());
        fit_.t = p.world_m - Rp;
        fit_.valid = true;
        fit_.provisional = true;
        return fit_;
    }

    // ≥2 pairs: 2-D Procrustes with the scale FIXED at 1. Scale is not a free parameter — both frames
    // are metric, and letting it float would absorb a bad correspondence into a plausible-looking fit
    // instead of showing up as residual.
    Eigen::Vector2f pbar = Eigen::Vector2f::Zero(), qbar = Eigen::Vector2f::Zero();
    for (const auto& q : pairs_) { pbar += q.room_m; qbar += q.world_m; }
    pbar /= static_cast<float>(pairs_.size());
    qbar /= static_cast<float>(pairs_.size());

    float a = 0.0f, b = 0.0f;
    for (const auto& q : pairs_)
    {
        const Eigen::Vector2f dp = q.room_m - pbar, dq = q.world_m - qbar;
        a += dp.x() * dq.x() + dp.y() * dq.y();
        b += dp.x() * dq.y() - dp.y() * dq.x();
    }
    fit_.theta = std::atan2(b, a);
    const float c = std::cos(fit_.theta), s = std::sin(fit_.theta);
    fit_.t = qbar - Eigen::Vector2f(c * pbar.x() - s * pbar.y(), s * pbar.x() + c * pbar.y());
    fit_.valid = true;

    // Residual over the pairs. With exactly 2 it is 0 by construction (a rigid transform fits 2 points
    // exactly) and says nothing — reported anyway, with n_pairs beside it, so nobody reads 0.000 as
    // confirmation. From 3 on it is a real consistency check, and a mis-identified door shows as metres.
    double acc = 0.0;
    for (const auto& q : pairs_)
    {
        const Eigen::Vector2f Rp(c * q.room_m.x() - s * q.room_m.y(),
                                 s * q.room_m.x() + c * q.room_m.y());
        acc += (q.world_m - (Rp + fit_.t)).norm();
    }
    fit_.residual_m = static_cast<float>(acc / pairs_.size());
    return fit_;
}

std::optional<Eigen::Matrix4d> DoorWorldRegistration::world_T_room() const
{
    if (not fit_.valid)
        return std::nullopt;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    const double c = std::cos(fit_.theta), s = std::sin(fit_.theta);
    M(0, 0) = c; M(0, 1) = -s; M(1, 0) = s; M(1, 1) = c;
    M(0, 3) = fit_.t.x(); M(1, 3) = fit_.t.y();
    return M;
}

}   // namespace rc
