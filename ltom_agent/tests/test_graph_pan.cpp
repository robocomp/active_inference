// A left-drag on empty canvas pans the graph viewer when the window is shown again over a graph that no longer changes (a memory graph).
// Real viewer, offscreen Qt, private DDS domain 63.
#include "../../common/viewers/graph_pan.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QMouseEvent>
#include <QScrollBar>

#include <cmath>
#include <memory>
#include <print>

namespace
{
void run_events(int ms)
{
    QElapsedTimer t; t.start();
    while (t.elapsed() < ms) QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
}
void mouse(QWidget* vp, QEvent::Type type, QPointF at, Qt::MouseButtons held)
{
    QMouseEvent e(type, at, vp->mapToGlobal(at), type == QEvent::MouseMove ? Qt::NoButton : Qt::LeftButton, held, Qt::NoModifier);
    QCoreApplication::sendEvent(vp, &e);
}
}   // namespace

int main(int argc, char** argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    auto G = std::make_shared<DSR::DSRGraph>("ltom_pan_test", 63, std::string(LTOM_ROOT_JSON), true, static_cast<int8_t>(63), DSR::Queue);
    const auto root = G->get_node("root")->id();
    for (const auto& [name, x, y] : {std::tuple{"room_1", 0.f, 120.f}, {"room_2", 300.f, 120.f}, {"chair_1", 30.f, 300.f}})
    {
        auto n = DSR::Node::create<room_node_type>(name);
        G->add_or_modify_attrib_local<parent_att>(n, root);
        G->add_or_modify_attrib_local<level_att>(n, 1);
        G->add_or_modify_attrib_local<pos_x_att>(n, x);
        G->add_or_modify_attrib_local<pos_y_att>(n, y);
        G->insert_node(n);
    }
    {
        auto rn = *G->get_node("root");
        auto rt = G->get_rt_api();
        rt->insert_or_assign_edge_RT(rn, G->get_node("room_1")->id(), {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
        rt->insert_or_assign_edge_RT(rn, G->get_node("room_2")->id(), {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
        auto r1 = *G->get_node("room_1");
        rt->insert_or_assign_edge_RT(r1, G->get_node("chair_1")->id(), {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
    }
    DSR::GraphViewer gv(G);
    rc::viewers::keep_pannable(&gv);
    gv.resize(546, 720);
    gv.show();
    run_events(800);   // the show and its refits: the graph is quiet from here on
    gv.hide();         // minimised, or another desktop
    gv.show();
    run_events(400);

    auto* vp = gv.viewport();
    const QPointF from(4, 4);
    const QPointF to = from + QPointF(40, 30);
    if (gv.scene.itemAt(gv.mapToScene(from.toPoint()), QTransform()) != nullptr) { std::println(stderr, "FAIL: the drag does not start on empty canvas"); return 1; }
    const int h0 = gv.horizontalScrollBar()->value(), v0 = gv.verticalScrollBar()->value();
    mouse(vp, QEvent::MouseButtonPress, from, Qt::LeftButton);
    mouse(vp, QEvent::MouseMove, to, Qt::LeftButton);
    mouse(vp, QEvent::MouseButtonRelease, to, Qt::NoButton);
    const int dh = gv.horizontalScrollBar()->value() - h0, dv = gv.verticalScrollBar()->value() - v0;
    std::println("drag (+40,+30) moved the scrollbars by ({},{}), ranges h {} v {}", dh, dv,
                 gv.horizontalScrollBar()->maximum() - gv.horizontalScrollBar()->minimum(), gv.verticalScrollBar()->maximum() - gv.verticalScrollBar()->minimum());
    const bool ok = dh == -40 and dv == -30;
    std::println("{}", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
