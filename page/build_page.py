"""Bake the map data into the page.

Usage (from the repository root):
  python3 page/build_page.py data/map.json [data/twin.json] [--framework FILE ...] [--out FILE]

--framework adds a framework's lens to the page; repeat it to add several. See frameworks/README.md.
--out writes the page somewhere other than index.html, for a version you do not publish here.
"""
import json
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)

args, fws, out = [], [], os.path.join(root, "index.html")
rest = sys.argv[1:]
while rest:
    x = rest.pop(0)
    if x == "--framework":
        fws.append(rest.pop(0))
    elif x == "--out":
        out = rest.pop(0)
    else:
        args.append(x)
d = json.load(open(args[0], encoding="utf-8"))
tw = args[1] if len(args) > 1 else None

for n in d["needs"]:
    n.setdefault("open", bool(n.get("gap") or n.get("partly")))
if "calib" not in d:
    print("Note: the map has no calib list, so the 'Check the judge' exercise will be empty")

BUILT_IN = {"acc", "cat", "shop", "ai", "tier", "dg", "src"}
reserved = set().union(*(n.keys() for n in d["needs"]))
texts = {n["t"]: n for n in d["needs"]}
d["frameworks"] = []
for path in fws:
    f = json.load(open(path, encoding="utf-8"))
    key = f.get("key", "")
    if (not re.fullmatch(r"[a-z][a-z0-9_]{1,20}", key) or key in reserved
            or key in {x["key"] for x in d["frameworks"]}):
        sys.exit("%s: key %r must be 2 to 21 lower-case letters, digits or _, "
                 "and not one the map or another framework already uses" % (path, key))
    codes = f.get("codes", {})
    if not codes or not all(isinstance(v, dict) and v.get("n") for v in codes.values()):
        sys.exit('%s: codes must map each code to {"n": name, "d": description}' % path)
    bad = [h for h in f.get("hide_builtin", []) if h not in BUILT_IN]
    if bad:
        sys.exit("%s: hide_builtin can only name %s" % (path, ", ".join(sorted(BUILT_IN))))
    bad = [c for c in f.get("cross", []) if c not in BUILT_IN]
    if bad:
        sys.exit("%s: cross can only name %s" % (path, ", ".join(sorted(BUILT_IN))))
    coding = f.get("needs", {})
    unknown = sorted({c for c in coding.values() if c not in codes})
    if unknown:
        sys.exit("%s: needs use codes that are not in codes: %s" % (path, ", ".join(unknown)))
    missing = [t for t in coding if t not in texts]
    for t, c in coding.items():
        if t in texts:
            texts[t][key] = c
    coded = sum(1 for n in d["needs"] if key in n)
    print("Framework %s: %d of %d needs coded%s" % (
        f.get("name", key), coded, len(d["needs"]),
        "; %d coded needs are not in this map (reworded?)" % len(missing) if missing else ""))
    d["frameworks"].append({k: f[k] for k in ("key", "name", "hint", "credit", "codes", "hide_builtin", "cross") if k in f})

h = open(os.path.join(here, "template.html"), encoding="utf-8").read()
h = h.replace("__NN__", str(len(d["needs"]))).replace("__NS__", str(len(d["steps"])))
h = h.replace("__THR__", str(d.get("threshold", "")))
h = h.replace("__TWIN__", json.dumps(json.load(open(tw, encoding="utf-8")), ensure_ascii=False)
              .replace("</", "<\\/") if tw else "null")
h = h.replace("__DATA__", json.dumps(d, ensure_ascii=False).replace("</", "<\\/"))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, "w", encoding="utf-8").write(h)
print("Wrote %s" % os.path.relpath(out, root))
