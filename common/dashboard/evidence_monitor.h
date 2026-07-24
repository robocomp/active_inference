#pragma once

/*
 * evidence_monitor.h  —  live per-instance "evidence consuming" monitor for the concept agents
 *
 * SHARED, header-only dashboard window across the concept agents (a global counter strip + a per-instance
 * evidence table). Pure view: the agent fills the structs below and calls update_view().
 *
 * A standalone QWidget window (mirrors dashboard/custom_widget.h: header-only, NO Q_OBJECT, built + updated
 * on the Qt/GUI thread, which in these agents is the compute() thread). It surfaces, at a glance, the whole
 * evidence pipeline the agent runs each cycle instead of scrolling stdout / the CSV:
 *   • a global COUNTER strip — masks arriving, "table" slices, tracker assigned vs DISCARDED, births/merges/
 *     removals (cycle + cumulative), LiDAR sweep size, compute rate;
 *   • a per-instance SNAPSHOT table — one always-current row per table_N with its fit evidence (support split,
 *     energy, R, gate), LiDAR-reprojection evidence (rays/raw, residual, angular coverage), the free-space
 *     VACATE + COVERAGE counters, and the existence log-odds fused from the LiDAR-carve and silhouette-mask
 *     channels (occ/free/n each) plus the removal debounce streak.
 *
 * Pure view: the agent fills EvidenceGlobals + a vector<EvidenceRow> and calls update_view() (throttle it to
 * a few Hz). No DSR, no torch — just Qt.
 */

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QString>
#include <QColor>
#include <QFont>

#include <string>
#include <vector>

namespace rc
{

// One per-instance snapshot row.
struct EvidenceRow
{
    std::string node;            // table_N
    float conf        = 0.0f;    // last YOLO "table" confidence
    int   since_det   = 0;       // frames since the last fresh table mask
    int   n_zed       = 0;       // ZED slices fused this cycle
    int   n_ricoh     = 0;       // ricoh slices fused this cycle (≥2 ⇒ strip-seam split)
    int   cand        = 0;       // on-surface support pts this frame
    int   resid       = 0;       // off-surface (unexplained) support pts this frame
    bool  gated       = false;   // truncation-gated (predict-only) this frame
    float energy      = 0.0f;    // mean per-point data energy (NLL proxy)
    float R           = 0.0f;    // per-point measurement variance (m²)
    int   lidar_rays  = 0;       // LiDAR returns fed to the range factor
    int   lidar_raw   = 0;       // returns in a generous "near this table" box
    float lidar_resid = -1.0f;   // mean |dist| of returns to the model surface (m)
    float cov_ang     = -1.0f;   // angular-coverage weight (1−R)^p ∈[0,1]
    int   vacate      = 0;       // free-space beams that fired a VACATE (shrink) pull
    int   coverage    = 0;       // on-plane pts reclaimed by the COVERAGE (grow) term
    float L           = 0.0f;    // existence log-odds
    float p           = 0.5f;    // P(exists)
    // Existence channels. *_free is the RAW absence count; *_free_eff is what actually drove L after the
    // range/occlusion absence-confidence scaling + observed-guard (shown as raw→eff so degradation is visible).
    float ex_locc = 0.0f, ex_lfree = 0.0f, ex_lfree_eff = 0.0f;  int ex_ln    = 0;  // existence LiDAR-carve channel
    float ex_socc = 0.0f, ex_sfree = 0.0f, ex_sfree_eff = 0.0f;  int ex_sndet = 0;  // existence silhouette-mask channel
    int   streak      = 0;       // removal-decision debounce streak
    float w = 0.0f, h = 0.0f, H = 0.0f;   // fitted dims (m)
    std::string subtype = "?";   // inferred shape subtype (e.g. "round"/"square") from model-evidence; "?" = agent doesn't set it
};

// Global pipeline counters for the header strip.
struct EvidenceGlobals
{
    int  mask_frame_id = -1;   bool mask_stale = false;
    int  total_slices  = 0;    int  table_dets = 0;
    int  assigned      = 0;    int  discarded  = 0;
    int  births = 0, merges = 0, removals = 0;                  // this cycle
    long births_cum = 0, merges_cum = 0, removals_cum = 0;      // cumulative
    int  instances = 0;        int  sweep_points = 0;
    int  seam_splits = 0;      long seam_splits_cum = 0;        // instances fusing ≥2 ricoh slices (strip-seam split)
    int  ricoh_attention = 0;  // unassigned ricoh bearings this cycle (peripheral "seek a ZED view" attention targets)
    float compute_hz = 0.0f;
};

// Standalone monitor window: a global counter strip (header) over a per-instance evidence table.
class EvidenceMonitor : public QWidget
{
public:
    explicit EvidenceMonitor(const QString& title = QStringLiteral("evidence monitor"),
                             QWidget* parent = nullptr) : QWidget(parent)
    {
        setWindowTitle(title);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(4, 4, 4, 4);
        layout->setSpacing(4);

        header_ = new QLabel(this);
        header_->setTextFormat(Qt::RichText);
        header_->setWordWrap(true);
        { QFont f = header_->font(); f.setPointSizeF(f.pointSizeF() - 0.5); header_->setFont(f); }
        layout->addWidget(header_);

        static const QStringList cols = {
            "table", "conf", "since", "fuse", "cand", "resid", "expl%", "gate",
            "energy", "R", "Lrays", "Lraw", "Lresid", "covA", "vac", "cov+",
            "L", "p", "l.occ", "l.free", "l.n", "s.occ", "s.free", "s.ndet", "strk",
            "w", "h", "H", "shape" };
        table_ = new QTableWidget(0, cols.size(), this);
        table_->setHorizontalHeaderLabels(cols);
        table_->verticalHeader()->setVisible(false);
        table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table_->setSelectionMode(QAbstractItemView::NoSelection);
        table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        { QFont f = table_->font(); f.setPointSizeF(f.pointSizeF() - 0.5); table_->setFont(f); }
        layout->addWidget(table_);

        resize(1180, 300);
    }

    // Refresh the header strip + rebuild the per-instance table from one cycle's snapshot (throttle to a few Hz).
    void update_view(const EvidenceGlobals& g, const std::vector<EvidenceRow>& rows)
    {
        header_->setText(format_globals(g));

        table_->setRowCount(static_cast<int>(rows.size()));
        for (int r = 0; r < static_cast<int>(rows.size()); ++r)
        {
            const EvidenceRow& x = rows[r];
            // Row tint: red = removing (streak>0), amber = gated or stale (no recent mask), grey = no detection
            // yet, green = healthy/alive. Black text on a pastel fill stays readable in either app theme.
            const QColor tint =
                (x.streak > 0)            ? QColor(245, 200, 200) :
                (x.gated or x.since_det > 30) ? QColor(250, 235, 190) :
                (x.since_det >= 100000)   ? QColor(228, 228, 228) :
                                            QColor(205, 238, 205);
            const int tot = x.cand + x.resid;
            const float expl = tot > 0 ? 100.0f * static_cast<float>(x.cand) / tot : 0.0f;

            int c = 0;
            const auto put = [&](const QString& s)
            {
                auto* it = new QTableWidgetItem(s);
                it->setBackground(tint);
                it->setForeground(QColor(20, 20, 20));
                if (c > 0) it->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
                table_->setItem(r, c++, it);
            };
            put(QString::fromStdString(x.node));
            put(QString::asprintf("%.2f", x.conf));
            put(QString::number(x.since_det));
            put(QString::asprintf("%dz%dr", x.n_zed, x.n_ricoh));   // slices fused: ZED / ricoh (≥2r ⇒ seam split)
            put(QString::number(x.cand));
            put(QString::number(x.resid));
            put(QString::asprintf("%.0f", expl));
            put(x.gated ? QStringLiteral("Y") : QStringLiteral("·"));
            put(QString::asprintf("%.3f", x.energy));
            put(QString::asprintf("%.4f", x.R));
            put(QString::number(x.lidar_rays));
            put(QString::number(x.lidar_raw));
            put(x.lidar_resid < 0 ? QStringLiteral("–") : QString::asprintf("%.3f", x.lidar_resid));
            put(x.cov_ang    < 0 ? QStringLiteral("–") : QString::asprintf("%.2f", x.cov_ang));
            put(QString::number(x.vacate));
            put(QString::number(x.coverage));
            put(QString::asprintf("%+.2f", x.L));
            put(QString::asprintf("%.2f", x.p));
            put(QString::asprintf("%.1f", x.ex_locc));
            put(QString::asprintf("%.1f→%.1f", x.ex_lfree, x.ex_lfree_eff));   // raw→effective absence
            put(QString::number(x.ex_ln));
            put(QString::asprintf("%.0f", x.ex_socc));
            put(QString::asprintf("%.0f→%.0f", x.ex_sfree, x.ex_sfree_eff));   // raw→effective absence
            put(QString::number(x.ex_sndet));
            put(QString::number(x.streak));
            put(QString::asprintf("%.3f", x.w));
            put(QString::asprintf("%.3f", x.h));
            put(QString::asprintf("%.3f", x.H));
            put(QString::fromStdString(x.subtype));
        }
    }

private:
    static QString format_globals(const EvidenceGlobals& g)
    {
        const QString stale = g.mask_stale ? QStringLiteral("<b style='color:#c04040'>STALE</b>")
                                           : QStringLiteral("fresh");
        return QString::asprintf(
            "<b>masks</b> frame=%d (%s) &nbsp; slices=%d &nbsp; table_dets=%d &nbsp; "
            "assigned=%d &nbsp; <b>discarded=%d</b> &nbsp;|&nbsp; "
            "<b>instances=%d</b> &nbsp; births=%d/%ld &nbsp; merges=%d/%ld &nbsp; removals=%d/%ld "
            "&nbsp;|&nbsp; <b>seam-split=%d/%ld</b> &nbsp;|&nbsp; lidar_sweep=%d pts &nbsp; %.0f Hz",
            g.mask_frame_id, stale.toUtf8().constData(), g.total_slices, g.table_dets,
            g.assigned, g.discarded, g.instances,
            g.births, g.births_cum, g.merges, g.merges_cum, g.removals, g.removals_cum,
            g.seam_splits, g.seam_splits_cum, g.sweep_points, g.compute_hz);
    }

    QLabel*        header_ = nullptr;
    QTableWidget*  table_  = nullptr;
};

}  // namespace rc
