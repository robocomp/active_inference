#pragma once
/*
 * common/birth_surprise/birth_surprise_log.h — the birth-surprise / birth-fusion diagnostic, written once.
 *
 * WHAT IT MEASURES. residual_concept publishes a grid of UNEXPLAINED occupancy. Two questions follow, and
 * this writes both to CSV every cycle the grid was read:
 *
 *   birth_surprise.csv — where is there high surprise NOT covered by a believed footprint? Those regions are
 *                        birth candidates. The `covered` rows are the free sanity check: they should be ~0 if
 *                        residual_concept's concept-subtraction is working.
 *   birth_fusion.csv   — how much residual mass sits UNDER each YOLO detection? The question this exists to
 *                        answer is whether a real detection lands on high unexplained occupancy while a
 *                        flicker lands on ~0 — i.e. whether residual could GATE or accelerate birth.
 *
 * ★DIAGNOSTIC ONLY. Nothing here touches a belief, a birth decision or a removal. That is why the probe radii
 * (0.50 / 0.30 m) and the 0.30 m footprint margin are literals rather than config keys: a config key would
 * imply they matter at runtime. If residual-gated birth is ever wired in, promote them then.
 *
 * ★★THE LOCALE PIN IS NOT COSMETIC. These machines run LANG=es_ES.UTF-8 and Qt calls setlocale(LC_ALL, "") at
 * startup; if the C++ global locale is ever imbued from it, operator<< inserts THOUSANDS SEPARATORS into
 * integers — a timestamp 1785763853131 becomes "1,785,763,853,131", one CSV field becomes five, the field
 * count varies per row, and every value past the first big integer is shifted. The log stays readable and
 * silently means something else. Four agents each carried this warning in their own words; it lives here now.
 *
 * ★★★AND IT ROTATES. Both files open through rc::diag::open_rotating, never ios::trunc — a probe whose
 * subject is intermittent must not erase the run that finally captured it.
 */

#include <cmath>
#include <cstddef>
#include <fstream>
#include <locale>
#include <print>
#include <span>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "birth_surprise_probe.h"          // rc::BirthSurpriseProbe, rc::GridField, rc::FootprintBox
#include "../diag_log/rotating_csv.h"      // rc::diag::open_rotating

namespace rc
{

// The probe's whole per-agent state: two lazily-opened CSVs and two counters. It used to be four separate
// members in each of four agents; bundling them means a new agent cannot adopt the writer and forget a counter.
class BirthSurpriseLog
{
public:
    // `footprints` are the agent's BELIEVED object footprints — built by the caller because the extent fields
    // are per-object (a cabinet run is L×d where the box-shaped agents are w×h). `detections` are this cycle's
    // detection centroids in the room frame.
    void write(const GridField& gf,
               std::span<const FootprintBox> footprints,
               std::span<const Eigen::Vector2f> detections,
               int tracker_births, std::size_t instances)
    {
        const auto cands = BirthSurpriseProbe::scan(gf, {footprints.begin(), footprints.end()});
        const long cyc = ++cycle_;          // advances only on cycles where the grid field was actually read

        int n_birth = 0;                    // uncovered high-surprise regions = birth candidates
        for (const auto& c : cands)
            if (not c.covered_by_concept) ++n_birth;

        // ── birth_surprise.csv: one row per region per cycle ──────────────────────────────────────────
        if (not surprise_csv_.is_open())
        {
            rc::diag::open_rotating(surprise_csv_, "etc/birth_surprise.csv");
            surprise_csv_.imbue(std::locale::classic());   // see the header note — NOT cosmetic
            if (surprise_csv_.is_open())
                surprise_csv_ << "cycle,region,cx,cy,cells,mass,ext_x,ext_y,mean_p,mean_var,covered,"
                                 "n_objects,tracker_births,instances\n";
        }
        if (surprise_csv_.is_open())
        {
            int r = 0;
            for (const auto& c : cands)
                surprise_csv_ << cyc << ',' << r++ << ',' << c.cx << ',' << c.cy << ',' << c.cells << ','
                              << c.mass << ',' << c.ext_x << ',' << c.ext_y << ',' << c.mean_p << ','
                              << c.mean_var << ',' << (c.covered_by_concept ? 1 : 0) << ','
                              << footprints.size() << ',' << tracker_births << ',' << instances << '\n';
            surprise_csv_.flush();
        }

        // ── birth_fusion.csv: residual mass under each detection ──────────────────────────────────────
        if (not fusion_csv_.is_open())
        {
            rc::diag::open_rotating(fusion_csv_, "etc/birth_fusion.csv");
            fusion_csv_.imbue(std::locale::classic());
            if (fusion_csv_.is_open())
                fusion_csv_ << "cycle,det,det_x,det_y,mass_r05,mass_r03,near_dist,near_mass,covered,"
                               "n_objects,tracker_births,instances\n";
        }
        if (fusion_csv_.is_open())
        {
            int di = 0;
            for (const auto& d : detections)
            {
                const float m05 = BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.50f);
                const float m03 = BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.30f);
                float nd = 1e9f, nm = 0.0f;                       // nearest region to this detection
                for (const auto& c : cands)
                    if (const float dd = std::hypot(c.cx - d.x(), c.cy - d.y()); dd < nd) { nd = dd; nm = c.mass; }
                // covered = the detection sits inside an already-believed footprint ⇒ associate, not birth.
                bool covered = false;
                for (const auto& t : footprints)
                {
                    const float cc = std::cos(t.yaw), ss = std::sin(t.yaw);
                    const float dx = d.x() - t.cx, dy = d.y() - t.cy;
                    if (std::abs(cc * dx + ss * dy) <= 0.5f * t.w + 0.30f and
                        std::abs(-ss * dx + cc * dy) <= 0.5f * t.h + 0.30f) { covered = true; break; }
                }
                fusion_csv_ << cyc << ',' << di++ << ',' << d.x() << ',' << d.y() << ',' << m05 << ',' << m03
                            << ',' << (nd > 1e8f ? -1.0f : nd) << ',' << nm << ',' << (covered ? 1 : 0) << ','
                            << footprints.size() << ',' << tracker_births << ',' << instances << '\n';
                if (tracker_births > 0 and not covered)      // something just born — print its corroboration
                    std::print("[birth-fusion] BIRTH det@({:.2f},{:.2f}) residual mass_r05={:.1f} "
                               "mass_r03={:.1f} near_region_mass={:.1f} dist={:.2f}\n",
                               d.x(), d.y(), m05, m03, nm, nd);
            }
            fusion_csv_.flush();
        }

        // Console: throttled (~every 20 cycles) OR whenever the tracker actually births — so the surprise
        // state AT THE BIRTH INSTANT is always printed, which is the correlation the probe exists for.
        if (n_birth > 0 and (tracker_births > 0 or (print_ctr_++ % 20) == 0))
        {
            const BirthCandidate* top = nullptr;   // strongest UNcovered region (cands are sorted by mass)
            for (const auto& c : cands)
                if (not c.covered_by_concept) { top = &c; break; }
            if (top)
                std::print("[birth-surprise] uncovered={} objects={} tracker_births={} | top: ({:.2f},{:.2f}) "
                           "mass={:.1f} cells={} ext={:.2f}x{:.2f} mean_p={:.2f} var={:.3f}\n",
                           n_birth, footprints.size(), tracker_births, top->cx, top->cy, top->mass,
                           top->cells, top->ext_x, top->ext_y, top->mean_p, top->mean_var);
        }
    }

private:
    std::ofstream surprise_csv_, fusion_csv_;
    long cycle_     = 0;    // probe cycle index — advances only when the grid was read
    long print_ctr_ = 0;    // console throttle
};

}  // namespace rc
