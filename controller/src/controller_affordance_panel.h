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
#include "controller_camera_masks.h"

#include <QDialog>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QScrollBar>
#include <QTextEdit>
#include <QScrollArea>
#include <QSplitter>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>

#include <functional>
#include <vector>

namespace rc
{

class AffordancePanel : public QDialog
{
public:
    // Called when the operator presses Skip. Wired by ControllerDisplay to the worker's command queue —
    // the button must not touch the session from the GUI thread.
    using SkipCallback = std::function<void()>;

    explicit AffordancePanel(QWidget *parent = nullptr) : QDialog(parent)
    {
        setWindowTitle(QStringLiteral("Affordance"));
        // Non-modal, and it does NOT steal focus from the 2D view: this is meant to be watched WHILE
        // driving, not clicked through.
        setModal(false);
        setWindowFlag(Qt::Tool, true);
        resize(980, 560);   // wider: the chart keeps its width and the transcript sits beside it
        // ★THE WHOLE WINDOW COMMITS TO THE DARK GROUND the flow chart's palette was drawn for. It used
        // to inherit the desktop theme, so light-grey text landed on a light fill and the panel read as
        // shaded rather than as a readout. Set here rather than per-widget: the header, the chart and
        // the recent strip must share one ground or the seams between them become the loudest thing on
        // screen. Font is one step up and semi-bold — this is read at a glance, mid-drive.
        setStyleSheet(QStringLiteral(
            "QDialog { background-color: #232528; }"
            "QLabel  { color: #f2f2f2; font-size: 11pt; font-weight: 600; }"
            "QScrollArea { background-color: #232528; border: none; }"));

        auto *root = new QVBoxLayout(this);
        root->setContentsMargins(8, 8, 8, 8);
        root->setSpacing(6);

        header_ = new QLabel(this);
        header_->setTextFormat(Qt::RichText);
        header_->setWordWrap(true);
        root->addWidget(header_);

        // ── WHAT THE CAMERA SEES, WITH THE MASKS ON IT ────────────────────────────────────────────
        // Above the flow chart, because it is the RESULT and the chart is the procedure. An affordance
        // is taken in order to perceive; the silhouettes are the only direct evidence that it worked,
        // and until now the panel could show that the robot had ARRIVED and nothing about what that
        // arrival bought.
        // ★A PEEK, NOT A VIEWER. It is capped small and the flow chart takes every spare pixel: the
        // question this image answers is "did the silhouette show up", which a thumbnail settles, while
        // the question the chart answers — which step is stalled and on what number — needs the room.
        // ★NO TEXT UNDER IT. The listing of masks-in-sight that used to sit here said in words what the
        // image already says in place: each silhouette is drawn with its own label and confidence, on
        // the pixels it was computed from, and the affordance's object is the green one. The text was a
        // second rendering of the same frame that could only ever agree with the picture or contradict
        // it, and it cost the chart two lines of height.
        // Still in a splitter, so an operator who wants a closer look can drag one out for it.
        auto *split = new QSplitter(Qt::Vertical, this);
        split->setChildrenCollapsible(true);
        split->setHandleWidth(6);

        auto *cam_panel = new QWidget(split);
        auto *cam_layout = new QVBoxLayout(cam_panel);
        cam_layout->setContentsMargins(0, 0, 0, 0);
        cam_layout->setSpacing(3);
        camera_ = new QLabel(cam_panel);
        camera_->setMinimumHeight(94);
        camera_->setMaximumHeight(172);   // +30%: still a peek, but big enough to read the overlay
        camera_->setAlignment(Qt::AlignCenter);
        camera_->setStyleSheet(QStringLiteral("QLabel { background-color: #141618; border: 1px solid #3a3d41; }"));
        camera_->setText(QStringLiteral("waiting for the camera and the masks…"));
        cam_layout->addWidget(camera_, 1);
        split->addWidget(cam_panel);

        auto *scroll = new QScrollArea(split);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        flow_ = new AffordanceFlow(scroll);
        scroll->setWidget(flow_);
        split->addWidget(scroll);
        // Stretch 0/1: every pixel the window gains goes to the CHART. Without this the splitter would
        // divide new height between them and the peek would grow back into a viewer on a tall window.
        split->setStretchFactor(0, 0);
        split->setStretchFactor(1, 1);
        split->setSizes({195, 520});

        // ── THE CONVERSATION, ON THE RIGHT ────────────────────────────────────────────────────────
        // The chart on the left says which STEP a run is on. This says what the two agents SAID to
        // each other, in order, and they are different questions: every failure this pair has had was
        // about who was waiting for whom, and none of them were visible in a step. A one-cycle state
        // that the redraw rate cannot sample — JustCompleted, which once cost a whole competing
        // traversal — appears here because the line is written when the event happens, not when the
        // window repaints.
        // ★HORIZONTAL SPLITTER, so an operator can give the transcript the whole window when reading
        // an exchange and collapse it to nothing when watching a servo.
        auto *hsplit = new QSplitter(Qt::Horizontal, this);
        hsplit->setChildrenCollapsible(true);
        hsplit->setHandleWidth(6);
        hsplit->addWidget(split);

        auto *tpanel = new QWidget(hsplit);
        auto *tlay = new QVBoxLayout(tpanel);
        tlay->setContentsMargins(0, 0, 0, 0);
        tlay->setSpacing(3);
        auto *tcap = new QLabel(QStringLiteral("protocol"), tpanel);
        tcap->setStyleSheet(QStringLiteral(
            "color:#8b9198; font-size:9pt; font-weight:600; letter-spacing:1px;"));
        tlay->addWidget(tcap);
        transcript_ = new QTextEdit(tpanel);
        transcript_->setReadOnly(true);
        transcript_->setLineWrapMode(QTextEdit::NoWrap);
        transcript_->setStyleSheet(QStringLiteral(
            "QTextEdit { background-color:#1b1d20; color:#d7dbdf; border:1px solid #33373b;"
            " font-family:monospace; font-size:9pt; font-weight:400; }"));
        tlay->addWidget(transcript_, 1);
        hsplit->addWidget(tpanel);
        hsplit->setStretchFactor(0, 3);
        hsplit->setStretchFactor(1, 2);
        hsplit->setSizes({400, 300});
        root->addWidget(hsplit, 1);

        // ── SKIP ──────────────────────────────────────────────────────────────────────────────────
        // The operator's override on the epistemic policy. The selector maximises expected information
        // gain and has no way to know that THIS look is hopeless — the object is occluded, the
        // standpoint is wrong, the servo is grinding against a contract it will not satisfy — and until
        // now the only ways out were the contract's own timeout or stopping the robot. Sits beside the
        // history, because "what happened to the last few" is the context in which you decide to
        // abandon this one.
        auto *foot = new QHBoxLayout;
        foot->setContentsMargins(0, 0, 0, 0);
        recent_ = new QLabel(this);
        recent_->setTextFormat(Qt::RichText);
        recent_->setWordWrap(true);
        foot->addWidget(recent_, 1);
        skip_btn_ = new QPushButton(QStringLiteral("Skip ▸ next"), this);
        skip_btn_->setToolTip(QStringLiteral("Abandon this affordance and let the planner choose the "
                                             "next one. It is RETIRED, not released, so the selector "
                                             "cannot hand the same one straight back."));
        skip_btn_->setEnabled(false);
        skip_btn_->setStyleSheet(QStringLiteral(
            "QPushButton { color: #f2f2f2; background-color: #3a3d41; border: 1px solid #55595e;"
            " border-radius: 4px; padding: 4px 12px; font-weight: 600; }"
            "QPushButton:hover:enabled { background-color: #4a4e53; }"
            "QPushButton:disabled { color: #6d7278; border-color: #3a3d41; }"));
        QObject::connect(skip_btn_, &QPushButton::clicked, this,
                         [this] { if (on_skip_) on_skip_(); });
        foot->addWidget(skip_btn_);
        root->addLayout(foot);
    }

    void set_skip_callback(SkipCallback cb) { on_skip_ = std::move(cb); }

    void update_view(const AffordanceExecution &v)
    {
        header_->setText(header_text(v));
        // Armed while an affordance is running AND through the dwell. Disabling it during the dwell was
        // wrong in the way that matters: the dwell is precisely where the robot can get stuck waiting
        // for a confirmation that is not coming, and that is the moment an operator most needs a way
        // out. Pressing it there cancels the wait.
        skip_btn_->setEnabled(v.active or v.dwell_left_s > 0.f);

        flow_->set_view(v);

        if (v.recent.empty())
            recent_->clear();
        else
        {
            QString s = QStringLiteral("<span style='color:#c4c8cc'>recent: ");
            for (std::size_t i = 0; i < v.recent.size(); ++i)
                s += (i ? QStringLiteral(" &nbsp;·&nbsp; ") : QString())
                   + QString::fromStdString(v.recent[i]).toHtmlEscaped();
            recent_->setText(s + QStringLiteral("</span>"));
        }

        render_transcript(v);
    }

    /// Repaint only when the conversation actually moved. Rebuilding every frame would also reset the
    /// scrollbar every frame, so an operator could never read back through it while a run continued.
    void render_transcript(const AffordanceExecution &v)
    {
        if (v.transcript.size() == transcript_shown_) return;
        transcript_shown_ = v.transcript.size();

        const bool at_end = transcript_->verticalScrollBar()->value()
                         >= transcript_->verticalScrollBar()->maximum() - 4;
        QString html;
        const std::uint64_t t0 = v.transcript.empty() ? 0 : v.transcript.front().t_ms;
        for (const auto &l : v.transcript)
        {
            // Producer and consumer get different colours and different indents, because the one
            // question the transcript answers is WHICH SIDE spoke.
            // ★NAME THE SPEAKER ON EVERY LINE. Colour and indent alone need a legend and a memory;
            // the one question a transcript answers is WHICH SIDE said this, so it is written out.
            // Fixed-width so the three names form a column the eye can skip down.
            const char *col  = "#8b9198";
            const char *who  = "selector";
            switch (l.side)
            {
                case AffordanceExecution::ProtocolLine::Side::Producer:
                    col = "#7fa9d6"; who = "producer"; break;
                case AffordanceExecution::ProtocolLine::Side::Consumer:
                    col = "#d68c74"; who = "consumer"; break;
                default: break;
            }
            html += QStringLiteral("<div style='color:%1;white-space:pre'>"
                                   "<span style='color:#5f666d'>%2</span> "
                                   "<b>%3</b>  %4</div>")
                        .arg(QString::fromLatin1(col))
                        .arg(QString::asprintf("%7.1f", (l.t_ms - t0) / 1000.0))
                        .arg(QString::asprintf("%-8s", who))
                        .arg(QString::fromStdString(l.text).toHtmlEscaped());
        }
        transcript_->setHtml(html);
        // Follow the tail only if the operator was already at the tail. Yanking the view back while
        // someone is reading history is the fastest way to make a live log useless.
        if (at_end)
            transcript_->verticalScrollBar()->setValue(transcript_->verticalScrollBar()->maximum());
    }

private:

    // "It was top of the list and it was passed over, on purpose, and here is which one." Without this
    // the no-two-in-a-row rule looks exactly like a selector ignoring its own scores.
    static QString suppressed_line(const AffordanceExecution &v)
    {
        if (v.suppressed.empty()) return {};
        return QStringLiteral("<br><span style='color:#e6a23c'>skipping <b>%1</b> — just completed, "
                              "not selected twice in a row</span>")
                   .arg(QString::fromStdString(v.suppressed).toHtmlEscaped());
    }

    static QString header_text(const AffordanceExecution &v)
    {
        // A DWELL IS NOT IDLE. The robot is deliberately standing still so the acquisition above can be
        // read, and a panel that says "idle" for three seconds of that trains you to distrust it.
        if (v.dwell_left_s > 0.f)
        {
            QString s = QStringLiteral("<span style='color:#4aa3e0'><b>DWELL %1 s</b></span>")
                            .arg(static_cast<double>(v.dwell_left_s), 0, 'f', 1);
            // The confirming-look count is the REASON the robot is still standing there once the clock
            // has run out, so it belongs in the same line as the countdown, not somewhere else.
            if (v.dwell_mask_needed > 0)
                s += QStringLiteral(" <span style='color:%1'><b>mask %2/%3</b></span>")
                         .arg(v.dwell_mask_hits >= v.dwell_mask_needed ? QStringLiteral("#2ecc71")
                                                                       : QStringLiteral("#e6a23c"))
                         .arg(v.dwell_mask_hits).arg(v.dwell_mask_needed);
            return s + QStringLiteral(" <span style='color:#c4c8cc'>— finished; holding still until the "
                                      "acquisition is confirmed. The rows below are that affordance.</span>");
        }
        if (not v.active)
            return QStringLiteral("<b>idle</b> <span style='color:#c4c8cc'>— no affordance is executing. "
                                  "The rows below are the last one.</span>") + suppressed_line(v);
        QString s = QStringLiteral("<b>%1</b>").arg(QString::fromStdString(v.affordance).toHtmlEscaped());
        if (not v.object.empty())
            s += QStringLiteral(" <span style='color:#c4c8cc'>on</span> %1")
                     .arg(QString::fromStdString(v.object).toHtmlEscaped());
        s += QStringLiteral(" &nbsp; <span style='color:#c4c8cc'>policy</span> %1")
                 .arg(QString::fromStdString(v.policy).toHtmlEscaped());
        if (not v.phase.empty())
            s += QStringLiteral(" &nbsp; <span style='color:#c4c8cc'>phase</span> %1")
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
            s += QStringLiteral(" &nbsp; <span style='color:#c4c8cc'>%1 s</span>")
                     .arg(static_cast<double>(v.elapsed_s), 0, 'f', 1);
        if (not v.contract_known)
            s += QStringLiteral("<br><span style='color:#c4c8cc'>contract not resolved yet — pipeline "
                                "steps only, no completion clauses</span>");
        s += suppressed_line(v);
        return s;
    }

public:
    // Scaled to the label, KeepAspectRatio, and only re-scaled when the source or the size changed —
    // this is repainted at the control rate and a full rescale per cycle is a visible cost for nothing.
    void set_camera_view(const CameraMasksView &c)
    {
        if (not c.image.isNull())
        {
            const QSize target = camera_->size();
            if (c.image.cacheKey() != last_image_key_ or target != last_scaled_to_)
            {
                last_image_key_ = c.image.cacheKey();
                last_scaled_to_ = target;
                camera_->setPixmap(QPixmap::fromImage(c.image).scaled(target, Qt::KeepAspectRatio,
                                                                      Qt::SmoothTransformation));
            }
        }

        // Nothing else to render: the acquisition verdict is IN the image (the affordance's object is
        // outlined green and tagged "←", everything else in its own colour), and a note about a missing
        // camera or a stalled producer is drawn onto the frame itself by the composer. One rendering of
        // one frame — a second one in text could only agree or contradict.
    }

private:
    SkipCallback on_skip_;
    QPushButton *skip_btn_ = nullptr;
    QLabel *header_ = nullptr;
    QTextEdit *transcript_ = nullptr;
    std::size_t transcript_shown_ = 0;
    QLabel *recent_ = nullptr;
    QLabel *camera_ = nullptr;
    qint64  last_image_key_ = 0;
    QSize   last_scaled_to_;
    AffordanceFlow *flow_ = nullptr;
};

}   // namespace rc
