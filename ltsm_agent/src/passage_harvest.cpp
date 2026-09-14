#include "passage_harvest.h"

#include <QDebug>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <format>
#include <locale>
#include <ranges>

namespace ltsm
{

namespace
{
// door_concept's nodes: `object` type, name "door_*", class in object_subtype (door_scene_graph.cpp:128).
bool is_door(DSR::DSRGraph &g, const DSR::Node &n)
{
    if (const auto st = g.get_attrib_by_name<object_subtype_att>(n); st.has_value() and st.value() == "door")
        return true;
    return n.type() == "object" and n.name().starts_with("door_");
}

// The room a door hangs under, walked up the RT parent chain (door → wall → floor → room).
std::string room_of(DSR::DSRGraph &g, const DSR::Node &n)
{
    auto cur = n;
    for (int hop = 0; hop < 8; ++hop)
    {
        const auto p = g.get_attrib_by_name<parent_att>(cur);
        if (not p.has_value()) break;
        const auto up = g.get_node(p.value());
        if (not up.has_value()) break;
        if (up->type() == "room") return up->name();
        cur = up.value();
    }
    return "<no room>";
}

std::string csv_escape(std::string s)
{
    std::ranges::replace(s, ',', ';');      // the field separator must not appear inside a field
    std::ranges::replace(s, '\n', ' ');
    return s;
}
}   // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
void fold_llr(float &alpha, float &beta, float llr)
{
    if (not std::isfinite(llr) or llr == 0.f) return;        // discriminated nothing ⇒ exact no-op

    const double a = alpha, b = beta;
    const double mbar = a / (a + b);
    // w = mbar·L1 / (mbar·L1 + (1−mbar)·L0) with L1/L0 = exp(llr), written so neither exp overflows.
    const double w = 1.0 / (1.0 + ((1.0 - mbar) / mbar) * std::exp(-static_cast<double>(llr)));

    const double m  = (a + w) / (a + b + 1.0);
    const double e2 = w * ((a + 1.0) * (a + 2.0)) / ((a + b + 1.0) * (a + b + 2.0))
                    + (1.0 - w) * (a * (a + 1.0)) / ((a + b + 1.0) * (a + b + 2.0));
    const double v  = e2 - m * m;
    if (not(v > 0.0) or not std::isfinite(v)) return;        // degenerate: leave the belief alone
    const double k  = m * (1.0 - m) / v - 1.0;
    if (not(k > 0.0) or not std::isfinite(k)) return;

    alpha = static_cast<float>(m * k);
    beta  = static_cast<float>((1.0 - m) * k);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
PassageHarvest::PassageHarvest(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory,
                               std::filesystem::path csv_path)
    : live_(std::move(live)), mem_(std::move(memory)), csv_(std::move(csv_path)) {}

int PassageHarvest::poll()
{
    if (not live_) return 0;
    int recorded = 0;

    for (const auto &n : live_->get_nodes_by_type("object"))
    {
        if (not is_door(*live_, n)) continue;
        const auto seq = live_->get_attrib_by_name<passage_seq_att>(n);
        if (not seq.has_value() or seq.value() <= 0) continue;

        auto &last = last_seq_[n.name()];
        if (seq.value() == last) continue;                   // nothing new this cycle (the usual case)
        // ★ A GAP IS DATA. Episodes we never saw — the agent was down, or a room departed between
        // two polls — must not be silently absorbed into "one new episode".
        if (seq.value() > last + 1 and last != 0)
            qWarning() << "[passage]" << QString::fromStdString(n.name()) << "jumped seq" << last
                       << "->" << seq.value() << "--" << (seq.value() - last - 1)
                       << "episode(s) happened while nobody was harvesting; they are lost, not merged.";
        last = seq.value();

        // ★ ABSENT IS NOT ZERO, AND THIS READER HAD THE DEFECT IT WARNS ABOUT. A missing
        // `clear_span_m` read as 0.0 is indistinguishable from a door measured fully shut; a missing
        // `duration_s` read as 0.0 is indistinguishable from the instant-Timeout signature this
        // agent publishes as suspicious. So the measurement columns default to **-1 = NOT
        // MEASURED**, matching door_concept's sentinel convention (every sentinel outside its own
        // quantity's range). `llr` is the exception and keeps 0: for a log-ratio, 0 IS the neutral
        // value and folding it is an exact no-op by construction, so absent and neutral coincide
        // there honestly.
        const auto datetime = live_->get_attrib_by_name<passage_datetime_att>(n);
        const auto crossed  = live_->get_attrib_by_name<passage_crossed_att>(n);

        PassageRow r;
        r.door     = n.name();
        r.room     = room_of(*live_, n);
        r.seq      = seq.value();
        r.datetime       = datetime.value_or("");
        r.crossed        = crossed.value_or(false);
        r.passable       = live_->get_attrib_by_name<passage_passable_att>(n).value_or(false);
        r.open_requested = live_->get_attrib_by_name<passage_open_requested_att>(n).value_or(false);
        r.episode_cut    = live_->get_attrib_by_name<passage_episode_cut_att>(n).value_or(false);
        r.open_answer    = live_->get_attrib_by_name<passage_open_answer_att>(n).value_or("");
        r.outcome        = live_->get_attrib_by_name<passage_outcome_att>(n).value_or("");
        r.open_prob      = live_->get_attrib_by_name<passage_open_prob_att>(n).value_or(-1.f);
        r.llr            = live_->get_attrib_by_name<passage_llr_att>(n).value_or(0.f);
        r.clear_span_m   = live_->get_attrib_by_name<passage_clear_span_m_att>(n).value_or(-1.f);
        r.body_width_m   = live_->get_attrib_by_name<passage_body_width_m_att>(n).value_or(-1.f);
        r.duration_s     = live_->get_attrib_by_name<passage_duration_s_att>(n).value_or(-1.f);

        // An episode missing its identifying fields is RECORDED but never FOLDED: the episode did
        // happen (the sequence moved), so dropping the row would lose that, but folding from
        // defaulted values would be inventing evidence.
        const bool incomplete = not datetime.has_value() or not crossed.has_value();

        // ── THE FOLDING DECISION, RECORDED IN THE ROW ───────────────────────────────────────────
        // Written into the CSV rather than left implicit, so the file says what was done with each
        // episode and a later reader does not have to re-derive the rules from this source.
        const bool cut = r.episode_cut;
        r.fold_llr = not cut and not incomplete;    // a partial judgement must not move a belief
        if (incomplete)                r.fold_claim = "none:incomplete";
        else if (cut)                  r.fold_claim = "none:cut";
        else if (r.outcome.empty())    r.fold_claim = r.crossed ? "none:unclaimed-crossing" : "none:no-claim";   // NOLINT
        else if (r.outcome == "Satisfied") r.fold_claim = "positive";
        else if (r.outcome == "Timeout")   r.fold_claim = "negative";
        else                           r.fold_claim = "none:" + r.outcome;   // approach facts, not aperture facts

        if (incomplete)
            qWarning() << "[passage]" << QString::fromStdString(n.name()) << "seq" << r.seq
                       << "is missing passage_datetime and/or passage_crossed -- the episode is"
                       << "recorded but NOT folded. Defaulting those would have invented evidence.";

        append(r);
        apply(pending_[r.door], r);
        ++recorded;

        qInfo().noquote() << QString::fromStdString(std::format(
            "[passage] {} seq={} crossed={} llr={:+.3f} outcome='{}' -> {}{}",
            r.door, r.seq, r.crossed, r.llr, r.outcome.empty() ? "<unclaimed>" : r.outcome,
            r.fold_claim, r.fold_llr ? " +llr" : ""));
    }
    return recorded;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void PassageHarvest::apply(PassageBelief &b, const PassageRow &r)
{
    if (r.fold_llr) fold_llr(b.pass_alpha, b.pass_beta, r.llr);

    // ★ "ASSISTED" IS NOT "WE ASKED". `passage_open_requested` is a bool about US; it says a request
    // was made, not that anybody could act on it. Three cases hide inside it: the provider accepted
    // and the leaf swung; the provider REFUSED (unknown door, not actuable); or the request never
    // left this fleet at all. Crediting the `open` Beta for the last two attributes to our request
    // an outcome it had no part in — a person opening the door, or the door having been open all
    // along. `passage_open_answer` carries the provider's own words, so the discrimination is in the
    // row and does not have to be inferred.
    const bool accepted = r.open_answer == "Delivered" or r.open_answer == "InProgress";
    const bool assisted = r.open_requested and accepted;

    // An UNRECOGNISED answer is not silently treated as a refusal: that would suppress evidence
    // whenever the provider's vocabulary drifts, and suppressed evidence looks exactly like a door
    // nobody ever opened. Say it once per answer word and carry on treating it as unassisted.
    if (r.open_requested and not accepted and not r.open_answer.empty())
        if (unknown_answers_.insert(r.open_answer).second)
            qWarning() << "[passage] open_answer" << QString::fromStdString(r.open_answer)
                       << "is not one of Delivered/InProgress -- treating it as a refusal (the `open`"
                       << "Beta is not credited). If the provider's vocabulary changed, this rule needs"
                       << "updating, not the data.";

    // ★ EVERY FULL COUNT IS ANCHORED TO SOMETHING PHYSICAL THE ROBOT DID; every inference is left to
    // the weighted llr channel.
    //   Satisfied unassisted → it walked through unaided          ⇒ pass_alpha
    //   Satisfied assisted   → the request delivered a usable hole ⇒ open_alpha, and `pass` gets
    //                          NOTHING from the claim: "it must have been shut, we asked" rests on
    //                          door_open_prob, which on a flat leaf-angle curve is mostly the PRIOR.
    //                          Charging pass_beta there would drift `pass` down on every visit to a
    //                          door we habitually ask about, for a reason entirely internal to us.
    //   Timeout assisted     → asked, accepted, still could not get through ⇒ open_beta
    //   Timeout unassisted   → tried to walk through and could not          ⇒ pass_beta
    // A REFUSED request is physically identical to no request — nothing was actuated — so it lands
    // on the unassisted branches, where the evidence is about the aperture as found.
    if (r.fold_claim == "positive")
    {
        if (assisted) b.open_alpha += 1.f;
        else          b.pass_alpha += 1.f;
    }
    else if (r.fold_claim == "negative")
    {
        if (assisted) b.open_beta += 1.f;
        else          b.pass_beta += 1.f;
    }

    if (r.crossed) ++b.crossings;
    b.seq_last = std::max(b.seq_last, r.seq);
    if (not r.datetime.empty()) b.last_datetime = r.datetime;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
bool PassageHarvest::append(const PassageRow &r)
{
    std::error_code ec;
    if (const auto dir = csv_.parent_path(); not dir.empty()) std::filesystem::create_directories(dir, ec);
    const bool fresh = not std::filesystem::exists(csv_, ec);

    std::ofstream f(csv_, std::ios::app);
    if (not f) { qWarning() << "[passage] cannot open" << csv_.c_str(); return false; }
    // ★ CLASSIC LOCALE ON THE WAY OUT. These machines run es_ES and Qt calls setlocale(LC_ALL, "");
    // a file written with comma decimals and read back with from_chars is a silent data loss.
    f.imbue(std::locale::classic());

    if (fresh)
        f << "door,room,datetime,seq,crossed,passable,open_prob,llr,clear_span_m,body_width_m,"
             "open_requested,open_answer,outcome,duration_s,episode_cut,fold_claim,fold_llr\n";

    f << csv_escape(r.door) << ',' << csv_escape(r.room) << ',' << csv_escape(r.datetime) << ','
      << r.seq << ',' << (r.crossed ? 1 : 0) << ',' << (r.passable ? 1 : 0) << ','
      << r.open_prob << ',' << r.llr << ',' << r.clear_span_m << ',' << r.body_width_m << ','
      << (r.open_requested ? 1 : 0) << ',' << csv_escape(r.open_answer) << ','
      << csv_escape(r.outcome) << ',' << r.duration_s << ',' << (r.episode_cut ? 1 : 0) << ','
      << csv_escape(r.fold_claim) << ',' << (r.fold_llr ? 1 : 0) << '\n';
    ++rows_written_;
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PassageRow> PassageHarvest::read_rows(const std::string &door_name) const
{
    std::vector<PassageRow> rows;
    std::ifstream f(csv_);
    if (not f) return rows;
    f.imbue(std::locale::classic());

    std::string line;
    std::getline(f, line);                               // header
    while (std::getline(f, line))
    {
        std::vector<std::string> c;
        for (const auto part : std::views::split(line, ','))
            c.emplace_back(part.begin(), part.end());
        if (c.size() < 17 or c[0] != door_name) continue;

        // ★ from_chars, never strtof/stof: locale-independent by definition, and it REPORTS failure
        // instead of returning the integer part of "0.26" under a comma locale (CLAUDE.md).
        const auto num = [&c](std::size_t i) -> float
        {
            float v = 0.f;
            std::from_chars(c[i].data(), c[i].data() + c[i].size(), v);
            return v;
        };
        PassageRow r;
        r.door = c[0]; r.room = c[1]; r.datetime = c[2];
        r.seq = static_cast<int>(num(3));
        r.crossed = c[4] == "1"; r.passable = c[5] == "1";
        r.open_prob = num(6); r.llr = num(7); r.clear_span_m = num(8); r.body_width_m = num(9);
        r.open_requested = c[10] == "1"; r.open_answer = c[11]; r.outcome = c[12];
        r.duration_s = num(13); r.episode_cut = c[14] == "1";
        r.fold_claim = c[15]; r.fold_llr = c[16] == "1";
        rows.push_back(std::move(r));
    }
    return rows;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<PassageRow> PassageHarvest::last_row_for(const std::string &door_name) const
{
    const auto rows = read_rows(door_name);
    if (rows.empty()) return {};
    return rows.back();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
AuditVerdict PassageHarvest::audit(const std::string &door_name) const
{
    AuditVerdict v;
    const auto rows = read_rows(door_name);

    // The labelled positive set: the robot PHYSICALLY went through, so the doorway WAS passable that
    // visit, whatever the silhouette thought. `crossed == false` is NOT a label — the robot mostly
    // did not cross because it did not want to.
    std::vector<float> crossed_llr;
    for (const auto &r : rows)
    {
        if (r.crossed and not r.episode_cut) crossed_llr.push_back(r.llr);
        if (r.fold_claim == "negative") ++v.folded_negatives;
        // ★ AN ACCEPTED REQUEST THAT TIMED OUT IS NOT A FAILED OPENER. door_concept's `open` can
        // only complete on a REAL observation of the leaf, and at the range `open` is taken from,
        // YOLO drops the door (its ADE20K posterior collapses 0.995 → 0.048 as the robot closes).
        // So a working provider produces Timeouts: the door opens and we cannot see it. These rows
        // credit `open_beta`, and without this count `open_beta` climbing reads as "the opener is
        // broken" when it means "our perception is too close".
        if (r.outcome == "Timeout" and (r.open_answer == "Delivered" or r.open_answer == "InProgress"))
            ++v.accepted_unconfirmed;
    }
    v.crossed_rows = static_cast<int>(crossed_llr.size());
    if (crossed_llr.empty())
    {
        v.text = std::format("[passage audit] {}: no crossed rows yet -- INCONCLUSIVE ({} folded "
                             "negative(s), {} accepted-but-unconfirmed)", door_name,
                             v.folded_negatives, v.accepted_unconfirmed);
        return v;
    }

    float sum = 0.f;
    for (const float l : crossed_llr) sum += l;
    v.mean_llr = sum / static_cast<float>(crossed_llr.size());
    v.strongest_llr = *std::ranges::max_element(crossed_llr,
                        [](float a, float b) { return std::fabs(a) < std::fabs(b); });

    // ★ THE VERDICT COMES FROM THE STRONGEST ROW, NOT THE MEAN, AND NEVER PASSES VACUOUSLY.
    // On a near-flat silhouette channel every llr is ~0, so a mean test reads non-negative whichever
    // way the normalisation points -- an inverted channel would clear it exactly while weak, which
    // is its expected state. With no row carrying real evidence the honest answer is INCONCLUSIVE.
    if (std::fabs(v.strongest_llr) < 1e-3f)
        v.result = AuditVerdict::Result::Inconclusive;
    else
        v.result = v.strongest_llr > 0.f ? AuditVerdict::Result::Ok : AuditVerdict::Result::Inverted;

    const char *word = v.result == AuditVerdict::Result::Ok ? "OK"
                     : v.result == AuditVerdict::Result::Inverted ? "INVERTED" : "INCONCLUSIVE";
    v.text = std::format(
        "[passage audit] {}: {} -- {} crossed row(s), mean llr {:+.4f}, strongest {:+.4f}, "
        "{} folded negative(s){}{}", door_name, word, v.crossed_rows, v.mean_llr, v.strongest_llr,
        v.folded_negatives,
        v.accepted_unconfirmed > 0
            ? std::format("; {} of them are ACCEPTED-BUT-UNCONFIRMED (the provider said Delivered "
                          "and we never saw the leaf move) -- read as a perception limit, NOT a "
                          "broken opener", v.accepted_unconfirmed)
            : "",
        v.result == AuditVerdict::Result::Inverted
            ? "  <-- the doorway the robot DROVE THROUGH reads as blocked: the visibility "
              "normalisation is inverted, do not bootstrap a belief from this channel"
            : "");
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
bool PassageHarvest::fold_into(std::uint64_t passage_node_id, const std::string &door_name)
{
    const auto it = pending_.find(door_name);
    if (it == pending_.end()) return false;

    auto node = mem_->get_node(passage_node_id);
    if (not node.has_value()) return false;
    const auto &b = it->second;

    // Read-modify-write on a node with EXACTLY ONE writer (this agent, on its own memory graph), so
    // the lost-update hazard of update_node does not apply here -- and the existing values matter:
    // a passage evicted before carries counts this fold must continue, not replace.
    auto &n = node.value();
    float pa = mem_->get_attrib_by_name<passage_pass_alpha_att>(n).value_or(1.f);
    float pb = mem_->get_attrib_by_name<passage_pass_beta_att>(n).value_or(1.f);
    float oa = mem_->get_attrib_by_name<passage_open_alpha_att>(n).value_or(1.f);
    float ob = mem_->get_attrib_by_name<passage_open_beta_att>(n).value_or(1.f);
    const int  prev_cross = mem_->get_attrib_by_name<passage_crossings_att>(n).value_or(0);

    // The pending belief started from Beta(1,1); adding its excess keeps a re-fold additive rather
    // than restarting the posterior from the prior each time a room is evicted.
    pa += b.pass_alpha - 1.f;  pb += b.pass_beta - 1.f;
    oa += b.open_alpha - 1.f;  ob += b.open_beta - 1.f;

    mem_->add_or_modify_attrib_local<passage_pass_alpha_att>(n, pa);
    mem_->add_or_modify_attrib_local<passage_pass_beta_att>(n, pb);
    mem_->add_or_modify_attrib_local<passage_open_alpha_att>(n, oa);
    mem_->add_or_modify_attrib_local<passage_open_beta_att>(n, ob);
    mem_->add_or_modify_attrib_local<passage_crossings_att>(n, prev_cross + b.crossings);
    mem_->add_or_modify_attrib_local<passage_seq_last_att>(n, b.seq_last);
    if (not b.last_datetime.empty())
        mem_->add_or_modify_attrib_local<passage_last_datetime_att>(n, b.last_datetime);
    const bool ok = mem_->update_node(n);

    if (ok)
    {
        const float mean = pa / (pa + pb);
        const float sd   = std::sqrt(pa * pb / ((pa + pb) * (pa + pb) * (pa + pb + 1.f)));
        qInfo().noquote() << QString::fromStdString(std::format(
            "[passage] folded {} into {}: p(passable) = {:.3f} +/- {:.3f}  Beta({:.2f},{:.2f}), "
            "p(open works) Beta({:.2f},{:.2f}), {} crossing(s) total",
            door_name, n.name(), mean, sd, pa, pb, oa, ob, prev_cross + b.crossings));
        qInfo().noquote() << QString::fromStdString(audit(door_name).text);
        pending_.erase(it);
    }
    return ok;
}

}   // namespace ltsm
