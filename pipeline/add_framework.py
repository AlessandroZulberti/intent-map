"""Draft a new framework for Intent map from its source: a link or a text file.

The AI reads the source, drafts the framework's levels, then codes every customer need and
journey step against them. The result is a DRAFT for a researcher to review before it is used.

Usage (from the repository root, same judge options as build_map.py):
  python3 pipeline/add_framework.py --source https://example.com/article --key care --name "CARE"
  python3 pipeline/add_framework.py --source notes/framework.txt --key mine --name "My framework"
      [--judge openrouter --model nvidia/nemotron-3.5-super-vl:free] [--out private/frameworks/mine.json]

--source   a web address, or a text file with the framework pasted in
--key      short lower-case name for the framework (letters, digits, _)
--name     the name shown on the page
--out      where to write it; default private/frameworks/<key>.json (git ignores private/)

About 10 requests: one to draft the levels, the rest to code needs and steps in batches.
Then build with:  python3 page/build_page.py data/map.json data/twin.json --framework <out> [--dashboard] --out private/index.html
"""
import html
import json
import os
import re
import sys
import urllib.request

import build_map as B
from larkhob_data import NEEDS, STEPS

BATCH = 15


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else default


def read_source(src):
    if re.match(r"https?://", src):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0 (Intent map framework drafting)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", "replace")
        raw = re.sub(r"(?is)<(script|style|nav|header|footer|noscript)[^>]*>.*?</\1>", " ", raw)
        text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    else:
        text = open(src, encoding="utf-8").read()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 200:
        sys.exit("The source has too little text to draft a framework from (%d characters)." % len(text))
    return text[:15000]


def ask(prompt, schema, name):
    for attempt in range(3):
        try:
            return json.loads(B.chat(prompt, schema, name))
        except Exception as e:
            err = e
            print("  retrying after: %s" % str(e)[:200])
    sys.exit("The model did not answer (%s). Nothing was written." % err)


LEVELS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"levels": {"type": "array", "minItems": 3, "maxItems": 10, "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"code": {"type": "string"}, "name": {"type": "string"}, "question": {"type": "string"}},
        "required": ["code", "name", "question"]}}},
    "required": ["levels"]}

CODES_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"codes": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"id": {"type": "integer"}, "code": {"type": "string"}, "second": {"type": "string"}},
        "required": ["id", "code", "second"]}}},
    "required": ["codes"]}


def code_batch(kind, items, levels):
    lv = "\n".join("%s = %s: %s" % (l["code"], l["name"], l["question"]) for l in levels)
    if kind == "needs":
        task = ("Each item is a question a customer asks a kitchen brand. For each, choose the level this need "
                "puts at risk if the brand leaves it unanswered.")
    else:
        task = "Each item is a step in a kitchen brand's customer journey. For each, choose the level the step is there to serve."
    prompt = (
        "You are coding items against a framework for a researcher, who will review every code.\n\n"
        "The framework's levels:\n%s\n\n%s\n"
        "Rules: code is one of the level codes above, or U when no level fits; do not force a fit. "
        "second is a second level only when the item clearly belongs to two; otherwise an empty string. "
        "Return one entry per item, using its id.\n\nItems:\n%s"
        % (lv, task, "\n".join("%d. %s" % (i, t) for i, t in items)))
    out = ask(prompt, CODES_SCHEMA, "codes")
    valid = {l["code"] for l in levels} | {"U"}
    got = {}
    for c in out.get("codes", []):
        code, second = c.get("code", "").strip(), c.get("second", "").strip()
        if c.get("id") in dict(items) and code in valid:
            got[c["id"]] = [code] + ([second] if second in valid and second not in (code, "U") else [])
    return got


def main():
    src, key, name = arg("--source"), arg("--key"), arg("--name")
    if not (src and key and name):
        sys.exit(__doc__)
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,20}", key):
        sys.exit("--key must be 2 to 21 lower-case letters, digits or _")
    out_path = arg("--out", os.path.join(B.HERE, "..", "private", "frameworks", "%s.json" % key)
                   if os.path.basename(B.HERE) == "pipeline" else os.path.join(B.HERE, "frameworks", "%s.json" % key))

    print("Reading the source...")
    text = read_source(src)
    print("Drafting the levels with %s..." % B.JUDGE_MODEL)
    levels = ask(
        "Below is a description of a framework. List its levels or components, in order, as the framework defines them. "
        "For each give: code (one or two capital letters, unique, never U), name (the framework's own name for it), and "
        "question (the question a customer would ask at that level, in your own words, under 12 words). "
        "Do not copy sentences from the text.\n\nFramework: %s\n\nText:\n%s" % (name, text),
        LEVELS_SCHEMA, "levels")["levels"]
    seen, clean = set(), []
    for l in levels:
        c = re.sub(r"[^A-Z]", "", l["code"].upper())[:2]
        if c and c != "U" and c not in seen:
            seen.add(c)
            clean.append({"code": c, "name": l["name"].strip(), "question": l["question"].strip()})
    if len(clean) < 3:
        sys.exit("The model found fewer than three levels; check the source.")
    for l in clean:
        print("  %-3s %s: %s" % (l["code"], l["name"], l["question"]))

    need_texts = [t for _, t in NEEDS]
    needs, steps = {}, {}
    print("Coding %d needs and %d steps..." % (len(need_texts), len(STEPS)))
    for i in range(0, len(need_texts), BATCH):
        got = code_batch("needs", [(k, need_texts[k]) for k in range(i, min(i + BATCH, len(need_texts)))], clean)
        needs.update({need_texts[k]: v for k, v in got.items()})
        print("  needs coded: %d of %d" % (len(needs), len(need_texts)))
    titles = [s[1] for s in STEPS]
    for i in range(0, len(titles), BATCH):
        got = code_batch("steps", [(k, "%s: %s" % (STEPS[k][1], STEPS[k][2])) for k in range(i, min(i + BATCH, len(titles)))], clean)
        steps.update({titles[k]: v[0] for k, v in got.items()})
        print("  steps coded: %d of %d" % (len(steps), len(titles)))

    # one more pass for anything the model skipped or coded with an unknown level
    left = [k for k, t in enumerate(need_texts) if t not in needs]
    for i in range(0, len(left), BATCH):
        got = code_batch("needs", [(k, need_texts[k]) for k in left[i:i + BATCH]], clean)
        needs.update({need_texts[k]: v for k, v in got.items()})
    left = [k for k, t in enumerate(titles) if t not in steps]
    for i in range(0, len(left), BATCH):
        got = code_batch("steps", [(k, "%s: %s" % (STEPS[k][1], STEPS[k][2])) for k in left[i:i + BATCH]], clean)
        steps.update({titles[k]: v[0] for k, v in got.items()})

    fw = {
        "key": key, "name": name, "draft": True,
        "hint": "Levels and codes drafted by %s from the source below; review before relying on them." % B.JUDGE_MODEL,
        "source": {"url": src} if re.match(r"https?://", src) else {"text": "Pasted text (%s)" % os.path.basename(src)},
        "codes": dict([(l["code"], {"n": l["name"], "d": l["question"]}) for l in clean]
                      + [("U", {"n": "Does not fit", "d": "no level describes what this need is about"})]),
        "cross": ["shop", "ai"],
        "needs": {t: (v if len(v) > 1 else v[0]) for t, v in needs.items()},
        "steps": steps,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(fw, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    missing = len(need_texts) - len(needs), len(titles) - len(steps)
    print("Wrote %s%s" % (out_path, "; not coded: %d needs, %d steps (run again or code by hand)" % missing if any(missing) else ""))


if __name__ == "__main__":
    main()
