"""stock: imagined — the last imagine_view picture (chat_agent writes imagined.json + imagined_zed.png into the run
dir): the imagined ZED view on the left, the LiDAR returns seen from above on the right, and one side-view panel per
LiDAR below (its ring lines from its own mount, plus its points)."""
import sys

from rcanvas import *

im = imagined()
if im is None:
    sys.exit("nothing imagined yet: ask the robot to imagine something first")
names = ["helios", "bpearl"]
has_zed = int(getattr(im.image, "w", 0) or 0) > 0
lidars = [k for k, sensor in enumerate(im.lidar) if sensor]
if not has_zed and not lidars:
    sys.exit("the imagined view has neither a camera image nor a LiDAR scan")

fig = figure()
if has_zed and lidars:
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.2])
    zed, top, bottom = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), gs[1, :]
elif has_zed:
    zed, top, bottom = fig.add_subplot(111), None, None
else:
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.2])
    zed, top, bottom = None, fig.add_subplot(gs[0, 0]), gs[1, 0]
if zed is not None:
    draw_imagined_zed(zed, im)
if lidars:
    inner = bottom.subgridspec(1, len(lidars))
    side = {k: fig.add_subplot(inner[0, j]) for j, k in enumerate(lidars)}
    draw_imagined_lidar(top, [side.get(k) for k in range(len(im.lidar))], im)
    for k, ax in side.items():
        ax.set_title(f"{names[k] if k < len(names) else f'lidar {k}'}: side view", fontsize=8)
fig.tight_layout()
save(fig)
