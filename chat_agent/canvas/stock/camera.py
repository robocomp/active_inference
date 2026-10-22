"""stock: camera [zed|ricoh] — the newest frame of a camera, fitted to the canvas. PIL only (no matplotlib), so a
live view re-runs fast. image() is mentioned here so chat_agent writes the frames into the snapshot."""
import sys

from PIL import Image, ImageDraw

from rcanvas import image, save_image, world

name = sys.argv[1] if len(sys.argv) > 1 else "zed"
frame = image(name)
w = world()
cw, ch = max(200, int(w.canvas.w)), max(150, int(w.canvas.h))
if frame is None:
    img = Image.new("RGB", (cw, ch), (40, 40, 40))
    ImageDraw.Draw(img).text((20, ch // 2), f"no frame from the {name} camera", fill=(230, 230, 230))
else:
    img = Image.fromarray(frame)
    img.thumbnail((cw, ch))
save_image(img)
