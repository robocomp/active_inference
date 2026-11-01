"""stock: layout — the current room seen from above: walls, doors (green open / red shut), objects, the robot and
where it is heading."""
from rcanvas import draw_layout, figure, layout_axes, save, world

w = world()
fig = figure()
ax = layout_axes(fig)
draw_layout(ax, w)
title = w.room.name if w.room is not None else "room unknown"
if w.robot is None:
    title += " — robot pose unknown"
ax.set_title(title)
fig.tight_layout()
save(fig)
