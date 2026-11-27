"""Check whether pages deliver the levels of a framework.

For each page, the AI reads the page content and judges every level of the framework: does the page
deliver it, partly deliver it, or not deliver it, quoting the words on the page that do. A quote that is
not on the page cannot count as delivering. Results go into the framework file, where the Frameworks
applied panel shows them next to the levels each page is meant to carry ("pages" in the framework file:
[{"id", "name", "url", "stage", "meant": [codes]}]). The AI is not told which levels a page is meant to carry.

Usage (from the repository root, same judge options as build_map.py):
  python3 pipeline/check_pages.py --pages private/pages/larkhob_pages.json --framework private/frameworks/levels.json
      [--judge openrouter --model nvidia/nemotron-3.5-super-vl:free] [--force]

One request per page. Pages already checked with the same model are skipped unless --force.
"""
import json
import re
import sys

import build_map as B


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else default


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"levels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"code": {"type": "string"}, "verdict": {"type": "string", "enum": ["delivers", "partly", "no"]},
                       "evidence": {"type": "string"}},
        "required": ["code", "verdict", "evidence"]}}},
    "required": ["levels"]}


def norm(s):
    return re.sub(r"[^a-z0-9£%]+", " ", s.lower()).strip()


def main():
    pages_path, fw_path = arg("--pages"), arg("--framework")
    if not (pages_path and fw_path):
        sys.exit(__doc__)
    pages = json.load(open(pages_path, encoding="utf-8"))["pages"]
    fw = json.load(open(fw_path, encoding="utf-8"))
    levels = {c: v for c, v in fw["codes"].items() if c != "U"}
    lv = "\n".join("%s = %s: %s" % (c, v["n"], v.get("d", "")) for c, v in levels.items())
    check = fw.get("page_check", {})
    force = "--force" in sys.argv
    for p in pages:
        done = check.get(p["id"])
        if done and done.get("_model") == B.JUDGE_MODEL and not force:
            print("  %s: already checked" % p["name"])
            continue
        prompt = (
            "You are reviewing one page of a kitchen brand's website against a framework, for a researcher.\n\n"
            "The framework's levels:\n%s\n\n"
            "For EVERY level, judge the page as a shopper would read it: delivers (the page gives a shopper what that "
            "level asks for), partly (it touches it but leaves an important part open), or no. Lines starting [Image: "
            "describe pictures on the page and count as content.\n"
            "evidence: copy the exact words from the page that deliver the level, under 15 words; an empty string if none. "
            "Be strict: a word in passing is not delivery.\n\nPage: %s (%s)\n\n%s"
            % (lv, p["name"], p.get("url", ""), p["text"]))
        out = None
        for attempt in range(3):
            try:
                out = json.loads(B.chat(prompt, SCHEMA, "page_check"))
                break
            except Exception as e:
                print("  retrying %s after: %s" % (p["name"], str(e)[:200]))
        if out is None:
            json.dump(fw, open(fw_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            sys.exit("Stopped at %s. Progress is saved; run the same command again." % p["name"])
        body = norm(p["text"])
        res = {"_model": B.JUDGE_MODEL}
        for r in out.get("levels", []):
            c = r.get("code", "").strip().upper()
            if c not in levels:
                continue
            v, e = r.get("verdict", "no"), r.get("evidence", "").strip().strip('"')
            if v == "delivers" and (not e or norm(e) not in body):
                v = "partly"      # a quote that is not on the page cannot prove delivery
            res[c] = {"v": v, "e": e if e and norm(e) in body else ""}
        for c in levels:
            res.setdefault(c, {"v": "no", "e": ""})
        check[p["id"]] = res
        print("  %s: %s" % (p["name"], ", ".join("%s %s" % (levels[c]["n"], res[c]["v"]) for c in levels)))
        fw["page_check"] = check
        json.dump(fw, open(fw_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Wrote the page check into %s" % fw_path)


if __name__ == "__main__":
    main()
