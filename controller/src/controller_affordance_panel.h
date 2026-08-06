#pragma once

/*
 * controller_affordance_panel.h — the affordance program, live, as a list of steps.
 *
 * Header-only, NO Q_OBJECT, built and updated on the Qt/GUI thread — the same shape as the fleet's
 * other dashboard widgets (see common/dashboard/evidence_monitor.h). A pure RENDERER: everything it
 * shows arrives in one AffordanceExecution snapshot, so it cannot disagree with the CSVs.
 *
 * WHY A WINDOW AND NOT A LOG LINE. The steps are a program with a shape — some run, some are skipped by
 * policy, the clause rows come from the contract and differ per affordance — and a program is easier to
 * read as a list than as a stream. But the reason it is worth building is narrower than "visibility":
 * every row carries the QUANTITY and its TARGET, and a stalled row says WHY. That is the difference
 * between a status light and a debugger.
 *
 * IT RECORDS WHENEVER AN AFFORDANCE IS ACTIVE, not only while open. These runs last seconds; a window
 * you have to open in time to catch the failure is a window that never catches it. Opening it shows what
 * already happened, and `recent` keeps the last finished runs.
 */

#include "controller_affordance_view.h"

#include <QDialog>
#include <QFrame>
#include <QGridLayout>
#include <QLabel>
#include <QProgressBar>
#include <QScrollArea>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>

#include <vector>

namespace rc
{

class AffordancePanel : public QDialog
{
public:
    explicit AffordancePanel(QWidget *parent = nullptr) : QDialog(parent)
    {
        setWindowTitle(QStringLiteral("Affordance"));
        // Non-modal, and it does NOT steal focus from the 2D view: this is meant to be watched WHILE
        // driving, not clicked through.
        setModal(false);
        setWindowFlag(Qt::Tool, true);
        resize(560, 420);

        auto *root = new QVBoxLayout(this);
        root->setContentsMargins(8, 8, 8, 8);
        root->setSpacing(6);

        header_ = new QLabel(this);
        header_->setTextFormat(Qt::RichText);
        header_->setWordWrap(true);
        root->addWidget(header_);

        auto *scroll = new QScrollArea(this);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        auto *host = new QWidget(scroll);
        rows_ = new QGridLayout(host);
        rows_->setContentsMargins(0, 0, 0, 0);
        rows_->setHorizontalSpacing(8);
        rows_->setVerticalSpacing(3);
        rows_->setColumnStretch(kColDetail, 1);
        scroll->setWidget(host);
        root->addWidget(scroll, 1);

        recent_ = new QLabel(this);
        recent_->setTextFormat(Qt::RichText);
        recent_->setWordWrap(true);
        root->addWidget(recent_);
    }

    void update_view(const AffordanceExecution &v)
    {
        header_->setText(header_text(v));

        // The row SET changes with the contract (clauses are per-affordance, and policy skips steps), so
        // rebuild when the shape changes and only re-text when it has not. Rebuilding every cycle would
        // fight the scroll position at 20 Hz.
        if (v.steps.size() != widgets_.size()) rebuild(v.steps.size());
        for (std::size_t i = 0; i < v.steps.size(); ++i) apply(widgets_[i], v.steps[i]);

        if (v.recent.empty())
            recent_->clear();
        else
        {
            QString s = QStringLiteral("<span style='color:#888'>recent: ");
            for (std::size_t i = 0; i < v.recent.size(); ++i)
                s += (i ? QStringLiteral(" &nbsp;·&nbsp; ") : QString())
                   + QString::fromStdString(v.recent[i]).toHtmlEscaped();
            recent_->setText(s + QStringLiteral("</span>"));
        }
    }

private:
    enum Col { kColState = 0, kColLabel, kColBar, kColDetail, kColTime, kColCount };

    struct Row { QLabel *state, *label, *detail, *time; QProgressBar *bar; };

    static QString header_text(const AffordanceExecution &v)
    {
        if (not v.active)
            return QStringLiteral("<b>idle</b> <span style='color:#888'>— no affordance is executing. "
                                  "The rows below are the last one.</span>");
        QString s = QStringLiteral("<b>%1</b>").arg(QString::fromStdString(v.affordance).toHtmlEscaped());
        if (not v.object.empty())
            s += QStringLiteral(" <span style='color:#888'>on</span> %1")
                     .arg(QString::fromStdString(v.object).toHtmlEscaped());
        s += QStringLiteral(" &nbsp; <span style='color:#888'>policy</span> %1")
                 .arg(QString::fromStdString(v.policy).toHtmlEscaped());
        if (not v.phase.empty())
            s += QStringLiteral(" &nbsp; <span style='color:#888'>phase</span> %1")
                     .arg(QString::fromStdString(v.phase).toHtmlEscaped());
        // The timeout is shown as a clock RUNNING OUT, not as an elapsed time: a run that is going to
        // fail on time should look like it is going to fail on time, before it does.
        if (v.timeout_s > 0.f)
        {
            const float left = v.timeout_s - v.elapsed_s;
            s += QStringLiteral(" &nbsp; <span style='color:%1'>%2 s left of %3</span>")
                     .arg(left < 2.f ? QStringLiteral("#c0392b") : QStringLiteral("#888"))
                     .arg(static_cast<double>(left < 0.f ? 0.f : left), 0, 'f', 1)
                     .arg(static_cast<double>(v.timeout_s), 0, 'f', 1);
        }
        else
            s += QStringLiteral(" &nbsp; <span style='color:#888'>%1 s</span>")
                     .arg(static_cast<double>(v.elapsed_s), 0, 'f', 1);
        if (not v.contract_known)
            s += QStringLiteral("<br><span style='color:#888'>contract not resolved yet — pipeline "
                                "steps only, no completion clauses</span>");
        return s;
    }

    void rebuild(std::size_t n)
    {
        while (QLayoutItem *it = rows_->takeAt(0)) { delete it->widget(); delete it; }
        widgets_.clear();
        for (std::size_t i = 0; i < n; ++i)
        {
            Row r{};
            const int row = static_cast<int>(i);
            r.state = new QLabel(this);
            r.state->setFixedWidth(16);
            r.label = new QLabel(this);
            r.label->setMinimumWidth(150);
            r.bar = new QProgressBar(this);
            r.bar->setRange(0, 1000);
            r.bar->setTextVisible(false);
            r.bar->setFixedSize(90, 10);
            r.detail = new QLabel(this);
            r.detail->setTextFormat(Qt::RichText);
            r.time = new QLabel(this);
            r.time->setStyleSheet(QStringLiteral("QLabel { color: #888; }"));
            rows_->addWidget(r.state, row, kColState);
            rows_->addWidget(r.label, row, kColLabel);
            rows_->addWidget(r.bar, row, kColBar);
            rows_->addWidget(r.detail, row, kColDetail);
            rows_->addWidget(r.time, row, kColTime);
            widgets_.push_back(r);
        }
    }

    static void apply(Row &w, const AffordanceStepView &s)
    {
        using S = AffordanceStepView::State;
        const char *glyph = s.state == S::Done    ? "✓"      // check
                          : s.state == S::Failed  ? "✗"      // cross
                          : s.state == S::Active  ? "▸"      // caret
                          : s.state == S::Skipped ? "–"      // dash
                                                  : "·";     // middle dot
        const char *colour = s.state == S::Done    ? "#27ae60"
                           : s.state == S::Failed  ? "#c0392b"
                           : s.state == S::Active  ? "#2980b9"
                                                   : "#888";
        w.state->setText(QStringLiteral("<span style='color:%1'>%2</span>")
                             .arg(QLatin1String(colour), QLatin1String(glyph)));
        w.state->setTextFormat(Qt::RichText);
        w.label->setText(QString::fromStdString(s.label).toHtmlEscaped());
        w.label->setStyleSheet(s.state == S::Active ? QStringLiteral("QLabel { font-weight: bold; }")
                                                    : QString());
        // A step with no meaningful fraction gets NO bar rather than a fake one. A clause is true or it
        // is not; drawing it half-full would be an invention.
        w.bar->setVisible(s.progress >= 0.f);
        if (s.progress >= 0.f)
            w.bar->setValue(static_cast<int>(1000.f * (s.progress < 0.f ? 0.f
                                                     : s.progress > 1.f ? 1.f : s.progress)));
        // blocked_why WINS the detail column when present. If something is stuck, the reason is the only
        // thing worth reading on that line.
        if (not s.blocked_why.empty())
            w.detail->setText(QStringLiteral("<span style='color:#c0392b'>%1</span>")
                                  .arg(QString::fromStdString(s.blocked_why).toHtmlEscaped()));
        else
            w.detail->setText(QString::fromStdString(s.detail).toHtmlEscaped());
        w.time->setText(s.state == S::Pending or s.state == S::Skipped
                            ? QString()
                            : QStringLiteral("%1 s").arg(static_cast<double>(s.elapsed_s), 0, 'f', 1));
    }

    QLabel *header_ = nullptr;
    QLabel *recent_ = nullptr;
    QGridLayout *rows_ = nullptr;
    std::vector<Row> widgets_;
};

}   // namespace rc
