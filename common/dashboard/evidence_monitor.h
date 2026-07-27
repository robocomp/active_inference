#pragma once

/*
 * evidence_monitor.h  —  live EVIDENCE-PIPELINE COUNTER STRIP for the concept agents
 *
 * SHARED, header-only dashboard widget (header-only, NO Q_OBJECT, built + updated on the Qt/GUI thread,
 * which in these agents is the compute() thread). Section 1 of the standard three-section concept-agent
 * dashboard:
 *
 *     ┌──────────────────────────────────────────┐
 *     │ 1. evidence-pipeline counters (THIS)     │   what arrived and what we did with it
 *     ├──────────────────────────────────────────┤
 *     │ 2. time series (FE, surprise, U(Σ), res) │   how the fit is moving
 *     ├──────────────────────────────────────────┤
 *     │ 3. belief inspector                      │   what we now believe (all DOFs, σ, Σ, modes)
 *     └──────────────────────────────────────────┘
 *
 * It answers ONE question — "is evidence flowing, and where is it going?": masks arriving (frame id +
 * freshness), how many slices, how many of OUR class, how many the tracker assigned vs DISCARDED, the
 * births/merges/removals it decided, how many instances are alive, the LiDAR sweep size, and the compute
 * rate.
 *
 * It used to also carry a 29-column per-instance table. That table is GONE: everything it showed about
 * the belief (state, σ, existence, modes) is now in the belief inspector below, in a form you can read at
 * a glance, and the rest was per-frame fit minutiae that belongs in the CSV logs, not on screen.
 *
 * Pure view: the agent fills EvidenceGlobals and calls update_view() (throttle it to a few Hz). No DSR,
 * no torch — just Qt.
 */

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>
#include <QFont>

namespace rc
{

// Per-cycle pipeline counters. Every concept agent fills the same struct, so every dashboard's top
// section reads identically regardless of which class the agent tracks.
struct EvidenceGlobals
{
    int  mask_frame_id = -1;   bool mask_stale = false;
    int  total_slices  = 0;    int  class_dets = 0;   // slices in the packet / slices of THIS agent's class
    int  assigned      = 0;    int  discarded  = 0;   // tracker: matched to an instance / dropped
    int  births = 0, merges = 0, removals = 0;                  // this cycle
    long births_cum = 0, merges_cum = 0, removals_cum = 0;      // cumulative
    int  instances = 0;        int  sweep_points = 0;
    int  seam_splits = 0;      long seam_splits_cum = 0;  // instances fusing ≥2 ricoh slices (table only)
    int  ricoh_attention = 0;  // unassigned ricoh bearings this cycle (peripheral "seek a ZED view" targets)
    float compute_hz = 0.0f;
};

// Section 1: the evidence-pipeline counter strip.
class EvidenceMonitor : public QWidget
{
public:
    explicit EvidenceMonitor(const QString& title = QStringLiteral("evidence monitor"),
                             QWidget* parent = nullptr) : QWidget(parent)
    {
        setWindowTitle(title);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(6, 4, 6, 4);
        layout->setSpacing(2);

        header_ = new QLabel(this);
        header_->setTextFormat(Qt::RichText);
        header_->setWordWrap(true);
        header_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        { QFont f = header_->font(); f.setPointSizeF(f.pointSizeF() - 0.5); header_->setFont(f); }
        layout->addWidget(header_);

        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    }

    // Refresh the strip from one cycle's counters (throttle to a few Hz).
    void update_view(const EvidenceGlobals& g) { header_->setText(format_globals(g)); }

private:
    static QString format_globals(const EvidenceGlobals& g)
    {
        const QString stale = g.mask_stale ? QStringLiteral("<b style='color:#c04040'>STALE</b>")
                                           : QStringLiteral("fresh");
        // Line 1 = INPUT (what arrived, what we kept). Line 2 = DECISIONS (what the tracker did) + rate.
        QString s = QString::asprintf(
            "<b>masks</b> frame=%d (%s) &nbsp; slices=%d &nbsp; ours=%d &nbsp; "
            "assigned=%d &nbsp; <b>discarded=%d</b> &nbsp;|&nbsp; lidar_sweep=%d pts",
            g.mask_frame_id, stale.toUtf8().constData(), g.total_slices, g.class_dets,
            g.assigned, g.discarded, g.sweep_points);
        s += QString::asprintf(
            "<br><b>instances=%d</b> &nbsp; births=%d/%ld &nbsp; merges=%d/%ld &nbsp; removals=%d/%ld"
            " &nbsp;|&nbsp; %.0f Hz",
            g.instances, g.births, g.births_cum, g.merges, g.merges_cum,
            g.removals, g.removals_cum, g.compute_hz);
        // Agent-specific extras appear only when the agent actually uses them, so the strip stays
        // identical across agents in the common case.
        if (g.seam_splits_cum > 0)
            s += QString::asprintf(" &nbsp;|&nbsp; seam-split=%d/%ld", g.seam_splits, g.seam_splits_cum);
        if (g.ricoh_attention > 0)
            s += QString::asprintf(" &nbsp;|&nbsp; ricoh-attention=%d", g.ricoh_attention);
        return s;
    }

    QLabel* header_ = nullptr;
};

}  // namespace rc
