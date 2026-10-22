"""stock: graph [node] — the robot's working memory (its DSR graph): RT edges as the tree, every other edge dashed
and labelled, nodes coloured by type. With a node name (e.g. room_1), only that node, the RT chain above it, its RT
subtree and its direct neighbours."""
import sys

from rcanvas import draw_graph, save_image

focus = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
save_image(draw_graph(focus))
