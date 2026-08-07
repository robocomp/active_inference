#pragma once

/*
 * controller_affordance_panel.h — the affordance program, live, as a FLOW CHART.
 *
 * Header-only, NO Q_OBJECT, built and updated on the Qt/GUI thread — the same shape as the fleet's
 * other dashboard widgets (see common/dashboard/evidence_monitor.h). A pure RENDERER: everything it
 * shows arrives in one AffordanceExecution snapshot, so it cannot disagree with the CSVs.
 *
 * WHY A WINDOW AND NOT A LOG LINE. The steps are a program with a SHAPE — a pipeline that enters a servo
 * LOOP, gated by an AND over per-contract clauses — and the shape is the part a log cannot carry. The
 * chart itself lives in controller_affordance_flow.h; this is the frame around it: the header line
 * (which affordance, on what, under which policy, how much timeout is left) and the `recent` strip.
 * The reason it is worth building is narrower than "visibility": every node carries the QUANTITY and its
 * TARGET, and a stalled node says WHY. That is the difference between a status light and a debugger.
 *
 * IT RECORDS WHENEVER AN AFFORDANCE IS ACTIVE, not only while open. These runs last seconds; a window
 * you have to open in time to catch the failure is a window that never catches it. Opening it shows what
 * already happened, and `recent` keeps the last finished runs.
 */

#include "controller_affordance_view.h"
#include "controller_affordance_flow.h"

#include <QDialog>
#include <QFrame>
#include <QLabel>
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
        resize(660, 500);

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
        flow_ = new AffordanceFlow(scroll);
        scroll->setWidget(flow_);
        root->addWidget(scroll, 1);

        recent_ = new QLabel(this);
        recent_->setTextFormat(Qt::RichText);
        recent_->setWordWrap(true);
        root->addWidget(recent_);
    }

    void update_view(const AffordanceExecution &v)
    {
        header_->setText(header_text(v));

        flow_->set_view(v);

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

    QLabel *header_ = nullptr;
    QLabel *recent_ = nullptr;
    AffordanceFlow *flow_ = nullptr;
};

}   // namespace rc
