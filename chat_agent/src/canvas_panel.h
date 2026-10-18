#pragma once
/*
 * canvas_panel.h — the side canvas of the chat window: what the robot draws for the person (2026-10-18).
 *
 *   ┌ room layout ● live                     [</>] [Save] [×] ┐
 *   │                                                          │
 *   │        the PNG a canvas script drew (canvas_runner.h),   │   page 0: a drawing, scaled to fit
 *   │        or the mission plan (plan_view.h)                 │   page 1: the plan, live
 *   │        or a node's 3-D model (gl_mesh_viewer.h)          │   page 2: drag to orbit, wheel to zoom
 *   │                                                          │
 *   ├ script (read-only, toggled by </>) ──────────────────────┤
 *   └──────────────────────────────────────────────────────────┘
 *
 * The chat window docks it to the right of the transcript (a QSplitter); it is open from the start with a hint,
 * [×] hides it and the next drawing brings it back. It does not run anything itself: it shows what it is given and reports [×], [Save] and its size upward
 * through callbacks (no Q_OBJECT, as chat_window.h). GUI thread only.
 */

#include <functional>

#include <QImage>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QStackedWidget>
#include <QString>
#include <QWidget>

#include <span>

#include <QVector3D>

#include "plan_view.h"

namespace rc::viewers { class GLMeshViewer; }

namespace rc::chat
{

class CanvasPanel : public QWidget
{
public:
    struct Callbacks
    {
        std::function<void()> on_close;                    ///< [×]: hide, stop the live re-runs
        std::function<void(const QString&)> on_save;       ///< [Save]: keep the script in the library under this name
    };

    explicit CanvasPanel(Callbacks cb, QWidget* parent = nullptr);

    /// A drawing: its title, the picture, whether it re-runs live, the script that drew it, and whether that script
    /// may be saved (a model-written one; the stock ones are already in the library).
    void show_drawing(const QString& title, const QImage& image, bool live, const QString& source, bool savable);
    void update_image(const QImage& image);   ///< a live re-run's new picture, nothing else changes
    void show_message(const QString& title, const QString& text);   ///< no picture: why (a failed script)
    void show_plan();
    /// A 3-D model (canvas_model.h): a triangle soup, shown in an orbit/zoom view; `caption` under the title.
    void show_model(const QString& title, std::span<const QVector3D> triangles, const QString& caption);
    [[nodiscard]] PlanView* plan() const { return plan_; }
    [[nodiscard]] bool showing_plan() const;
    void set_live(bool live);

    /// The pixel size a drawing should be made for (the picture area).
    [[nodiscard]] QSize drawing_size() const;

    void resizeEvent(QResizeEvent* e) override;

private:
    void set_title(const QString& title, bool live);
    void rescale();

    Callbacks cb_;
    QLabel* title_ = nullptr;
    QPushButton* code_btn_ = nullptr;
    QPushButton* save_btn_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    QLabel* picture_ = nullptr;
    PlanView* plan_ = nullptr;
    rc::viewers::GLMeshViewer* model_ = nullptr;
    QLabel* caption_ = nullptr;
    QPlainTextEdit* code_ = nullptr;
    QImage image_;
    QString title_text_;
    bool live_ = false;
};

}   // namespace rc::chat
