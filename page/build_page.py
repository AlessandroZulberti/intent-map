"""Bake the map data into the page.
Usage (from the repository root): python3 page/build_page.py data/map.json [data/twin.json]
Writes index.html in the repository root, which GitHub Pages serves."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
d = json.load(open(sys.argv[1], encoding="utf-8"))
tw = sys.argv[2] if len(sys.argv) > 2 else None
h = open(os.path.join(here, "template.html"), encoding="utf-8").read()
h = h.replace("__NN__", str(len(d["needs"]))).replace("__NS__", str(len(d["steps"]))).replace("__THR__", str(d.get("threshold", "")))
h = h.replace("__TWIN__", json.dumps(json.load(open(tw, encoding="utf-8")), ensure_ascii=False).replace("</", "<\\/") if tw else "null")
h = h.replace("__DATA__", json.dumps(d, ensure_ascii=False).replace("</", "<\\/"))
open(os.path.join(root, "index.html"), "w", encoding="utf-8").write(h)
print("Wrote index.html")
