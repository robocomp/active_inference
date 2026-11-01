"""stock: table <json> — a table of words from the model's table tool: {"title": str, "header": [str], "rows": [[…]]}
(chat_session.cpp passes the tool's arguments as one JSON argument). Cells may be numbers or null."""
import json
import sys

from rcanvas import draw_table, save

spec = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
rows = [[("" if c is None else c) for c in (r if isinstance(r, list) else [r])] for r in spec.get("rows") or []]
if not rows:
    sys.exit("no rows to draw: pass rows as a list of lists of cells")
save(draw_table(rows, header=spec.get("header") or None, title=spec.get("title") or None))
