"""Build the Intent map data file locally with Ollama.

Two models, both free and local:
  embeddinggemma  places every need and step by meaning, and shortlists the five steps nearest each need
  gemma3:4b       reads each need with its shortlist and judges which step answers it, if any

Usage:
  python3 pipeline/build_map.py              # full run, writes data/map.json (judge answers cached in pipeline/judge_cache.json)
  python3 pipeline/build_map.py --dry-run    # random vectors and a fake judge, to test without Ollama
  python3 pipeline/build_map.py --judge openrouter --model <id>   # judge with a larger hosted model (OPENROUTER_API_KEY)
Then: python3 page/build_page.py data/map.json data/twin.json   # bakes the data into index.html

Setup: Ollama running, `ollama pull embeddinggemma`, `ollama pull gemma3:4b`, `pip install numpy scikit-learn`.
"""
import json
import re
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from larkhob_data import STEPS, NEEDS, NEED_LENS, STEP_TYPE, CATS, ACC, STEP_QUESTIONS, OVERRIDES

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
JUDGE_MODEL = "gemma3:4b"
SHORTLIST = 5
OUT = os.path.join(HERE, "..", "data", "map.json")
CACHE = os.path.join(HERE, "judge_cache.json")


def post(path, body, timeout=600):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        sys.exit("Ollama call to %s failed (%s).\nIs the Ollama app running, and have you pulled %s and %s?"
                 % (path, e, EMBED_MODEL, JUDGE_MODEL))


def arg(name, default=None):
    """Value after a command-line flag, e.g. --model x."""
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else default


JUDGE_VIA = arg("--judge", "ollama")          # ollama | openrouter
if JUDGE_VIA == "openrouter":
    JUDGE_MODEL = arg("--model") or sys.exit("With --judge openrouter, give a model id: --model vendor/model:free")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"


def chat(prompt, schema, name):
    """Send one prompt to the judge and return its JSON text. Same prompt and schema for both routes."""
    if JUDGE_VIA == "ollama":
        r = post("/api/chat", {"model": JUDGE_MODEL, "stream": False, "format": schema, "options": {"temperature": 0},
                               "messages": [{"role": "user", "content": prompt}]})
        return r["message"]["content"]
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY first.")
    body = {"model": JUDGE_MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            "provider": {"require_parameters": True}}
    for wait in (0, 20, 60):
        time.sleep(wait)
        req = urllib.request.Request(OPENROUTER, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                msg = json.load(r)["choices"][0]["message"]
                if not msg.get("content"):
                    raise RuntimeError("empty content; message keys: %s" % list(msg))
                return msg["content"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code == 429:
                print("  rate limited (429), waiting... %s" % msg)
                continue
            raise RuntimeError("OpenRouter %s: %s" % (e.code, msg))
    raise RuntimeError("OpenRouter: still rate limited. The free plan allows about 50 requests a day.")


def embed(texts, label):
    vecs = []
    for i in range(0, len(texts), 16):
        vecs.extend(post("/api/embed", {"model": EMBED_MODEL, "input": texts[i:i + 16]})["embeddings"])
        print("  %s: embedded %d of %d" % (label, min(i + 16, len(texts)), len(texts)))
    v = np.array(vecs, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


JUDGE_VERSION = "strict-4-evidence"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_needs_to_know": {"type": "string"},
        "best_step": {"type": "integer", "minimum": 0, "maximum": SHORTLIST},
        "what_that_step_gives": {"type": "string"},
        "evidence": {"type": "string"},
        "verdict": {"type": "string", "enum": ["answers", "partly", "none"]},
        "reason": {"type": "string"},
        "missing_step": {"type": "string"},
    },
    "required": ["customer_needs_to_know", "best_step", "what_that_step_gives", "evidence", "verdict", "reason", "missing_step"],
}

EXAMPLES = """Examples from other shops (not Larkhob):

Need: "Will this rug's colour run if I wash it at 40 degrees?"
Steps: 1. Clean and care: Clean it and look after it so it lasts. Questions it answers: How do I wash it? 2. Check materials and safety: Check what it is made of.
customer_needs_to_know: whether the dye is colourfast at 40 degrees.
best_step: 1. what_that_step_gives: washing and care instructions, which normally state the safe temperature.
evidence: "How do I wash it?". verdict: answers. missing_step: ""

Need: "I'm 1.95 m tall. Will this bike frame fit me, and can I test ride it?"
Steps: 1. Check size and weight: Check the size and dimensions. 2. Ask before buying: Ask the shop a question.
customer_needs_to_know: which frame size suits their height, and whether a test ride is possible.
best_step: 1. what_that_step_gives: frame dimensions, but not a size guide by height and nothing about test rides.
evidence: "Check the size and dimensions". verdict: partly. missing_step: "Find my size and try it"

Need: "I use a wheelchair. Can I reach the controls on this sofa's recliner from a seated transfer?"
Steps: 1. Check size and weight: Check the size and dimensions. 2. Learn to use it: Learn how to use it from instructions.
customer_needs_to_know: whether the controls are reachable and usable for someone transferring from a wheelchair.
best_step: 0. what_that_step_gives: dimensions and operating instructions, which say nothing about reach or use from a wheelchair.
evidence: "". verdict: none. missing_step: "Check it works for my body"
"""


CONTACT_STEPS = {"Ask before buying", "Get help"}


def evidence_ok(evidence, step):
    """The quoted evidence must come from the step's own text: most of its content words must appear there."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in evidence.lower()).split() if len(w) > 3]
    if not words:
        return False
    text = (step[1] + " " + step[2] + " " + STEP_QUESTIONS[step[1]]).lower()
    return sum(w in text for w in words) / len(words) >= 0.6


PROMPT = arg("--prompt", "strict-2")      # strict-2 (default: tested best, 26 Nov 2026) | strict-4 (extra rules; tested worse)
if PROMPT not in ("strict-2", "strict-4"):
    sys.exit("--prompt must be strict-2 or strict-4")
if PROMPT == "strict-2":
    JUDGE_VERSION = "strict-2"
STRICT4_RULES = (
    "Being able to ask the shop or contact customer service is not an answer: \"Ask before buying\" and \"Get help\" "
    "can only partly answer a need.\n"
    "If the need names a specific detail (a body or health condition, a number, a person it is for, a comparison over "
    "time), the step answers only if it covers that detail, not just the topic.\n"
    "evidence: copy the exact words from the chosen step's text or questions that give the customer what they need. "
    "If no words do, the verdict cannot be \"answers\".\n\n")


def examples():
    """strict-2 had no evidence lines in its examples."""
    if PROMPT == "strict-4":
        return EXAMPLES
    return re.sub(r"evidence: .*?\. (?=verdict:)", "", EXAMPLES)


def judge(need, options):
    listing = "\n".join("%d. %s: %s Questions it answers: %s" % (k + 1, s[1], s[2], STEP_QUESTIONS[s[1]])
                        for k, s in enumerate(options))
    prompt = (
        "You are a strict reviewer of a shop's customer journey. The shop is Larkhob, a kitchen brand.\n\n"
        "Rule: a step ANSWERS a need only if the specific information or action the customer needs is a normal, "
        "explicit part of that step. Sharing a topic or a word is not enough. If a typical version of the step would "
        "leave the customer still asking their question, it does not answer it. When unsure, choose \"none\".\n"
        "- \"answers\": a step clearly gives them what they need.\n"
        "- \"partly\": a step gives some of it, but an important part of the question is left open.\n"
        "- \"none\": no step gives them what they need. best_step is 0.\n"
        "Needs about disability, access, a specific body or health condition, rules and laws, or safety are rarely "
        "answered by a general step: check carefully.\n"
        + (STRICT4_RULES if PROMPT == "strict-4" else "\n") + (
        "%s\n"
        "Now the real case.\nNeed: \"%s\"\nSteps:\n%s\n\n"
        "Fill every field in order. customer_needs_to_know: the specific thing they must find out or do. "
        "what_that_step_gives: what the best step would actually tell or let them do. reason: one plain sentence "
        "under 25 words. missing_step: if the verdict is not \"answers\", the step the journey is missing in 2 to 6 "
        "words, written as a customer action; otherwise an empty string."
        % (examples(), need, listing)))
    err = "no answer"
    for attempt in range(2):
        try:
            out = json.loads(chat(prompt, JUDGE_SCHEMA, "verdict"))
            v, k = out["verdict"], int(out["best_step"])
            if v in ("answers", "partly", "none") and 0 <= k <= len(options):
                if v != "none" and k == 0:
                    v = "none"
                if v == "none":
                    k = 0
                flag = ""
                if v == "answers" and PROMPT == "strict-4":
                    s = options[k - 1]
                    if s[1] in CONTACT_STEPS:
                        v, flag = "partly", "contact step"
                    elif not evidence_ok(out.get("evidence", ""), s):
                        v, flag = "partly", "no evidence in step"
                return {"flag": flag, "evidence": out.get("evidence", ""), "verdict": v, "step": k, "reason": out["reason"], "missing_step": out["missing_step"],
                        "needs_to_know": out["customer_needs_to_know"], "step_gives": out["what_that_step_gives"]}
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, str(e)[:300])
    print("  judge failed: %s" % err)
    return {"verdict": "none", "step": 0, "reason": "The judge could not decide.", "missing_step": "",
            "needs_to_know": "", "step_gives": "", "error": err}


FRAG_VERSION = "frag-1"
FRAG_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "integer", "minimum": 1, "maximum": SHORTLIST}, "part": {"type": "string"}},
            "required": ["step", "part"]}},
        "together": {"type": "string", "enum": ["answers", "partly", "none"]},
    },
    "required": ["parts", "together"],
}


def fragments(need, main, others):
    """Second pass, only for needs a step partly answers: do other steps supply the missing parts?"""
    listing = "\n".join("%d. %s: %s" % (k + 1, s[1], s[2]) for k, s in enumerate(others))
    prompt = (
        "You are a strict reviewer of a shop's customer journey.\n"
        "Need: \"%s\"\nThe main step \"%s\" (%s) answers only part of it.\n\nOther steps:\n%s\n\n"
        "parts: list each other step that clearly supplies a specific missing part of the answer, with that part "
        "in under 12 words. A step that only shares a topic does not count. Most needs have no such step: then "
        "return an empty list.\ntogether: if the customer used the main step and the listed steps together, would "
        "the need be answered (\"answers\"), still partly (\"partly\"), or not (\"none\")? With an empty list, "
        "together is \"partly\"." % (need, main[1], main[2], listing))
    content = chat(prompt, FRAG_SCHEMA, "fragments")
    try:
        out = json.loads(content)
        idx = sorted({int(p["step"]) - 1 for p in out.get("parts", [])
                      if 1 <= int(p["step"]) <= len(others) and p.get("part", "").strip()})
        tog = out.get("together") if idx else "partly"
        return {"also": idx, "together": tog if tog in ("answers", "partly", "none") else "partly"}
    except Exception:
        return {"also": [], "together": "partly"}


def main():
    dry = "--dry-run" in sys.argv
    n_needs, n_steps = len(NEEDS), len(STEPS)
    need_texts = [t for _, t in NEEDS]

    if dry:
        print("Dry run: random vectors and a fake judge. Results are meaningless.")
        rng = np.random.default_rng(0)
        def rnd(k):
            v = rng.normal(size=(k, 64)); return v / np.linalg.norm(v, axis=1, keepdims=True)
        Nq, Sd, Ns, Ss = rnd(n_needs), rnd(n_steps), rnd(n_needs), rnd(n_steps)
    else:
        print("Embedding with %s..." % EMBED_MODEL)
        # Retrieval mode: the need as a question, each step as the document that should answer it.
        Nq = embed(["task: question answering | query: " + t for t in need_texts], "needs as questions")
        Sd = embed(["title: %s | text: %s %s" % (a, b, STEP_QUESTIONS[a]) for _, a, b in STEPS], "steps as answers")
        # Similarity mode, used only for the 'by meaning' layout.
        Ns = embed(["task: clustering | query: " + t for t in need_texts], "needs for layout")
        Ss = embed(["task: clustering | query: %s. %s" % (a, b) for _, a, b in STEPS], "steps for layout")

    sim = Nq @ Sd.T
    shortlist = np.argsort(-sim, axis=1)[:, :SHORTLIST]

    cache = {}
    if os.path.exists(CACHE) and not dry:
        cache = json.load(open(CACHE))
    only = None
    if "--only-holdout" in sys.argv:
        gold = json.load(open(os.path.join(HERE, "gold_labels.json")))
        only = {g["need"] for g in gold["holdout"]}
        missing = only - set(need_texts)
        if missing:
            sys.exit("%d held-out needs are not in larkhob_data.py; was a need reworded?" % len(missing))
    verdicts = []
    fails = 0
    print("Judging %d needs with %s..." % (n_needs, JUDGE_MODEL))
    for i, text in enumerate(need_texts):
        if only is not None and text not in only:
            verdicts.append(None)
            continue
        if only is None and i in OVERRIDES:
            verdicts.append({"verdict": "none", "step": 0, "reason": "", "missing_step": "",
                             "needs_to_know": "", "step_gives": ""})
            continue
        opts = [STEPS[j] for j in shortlist[i]]
        key = JUDGE_VERSION + "|" + JUDGE_MODEL + "||" + text + "||" + "|".join(s[1] for s in opts)
        if dry:
            v = {"verdict": ["answers", "partly", "none"][i % 3], "step": 0 if i % 3 == 2 else 1,
                 "reason": "Fake judge.", "missing_step": "" if i % 3 == 0 else "Check it works for me"}
        elif key in cache and not cache[key].get("error") and cache[key].get("reason") != "The judge could not decide.":
            v = cache[key]
        else:
            v = judge(text, opts)
            if v.get("error"):
                fails += 1
                if fails >= 3:
                    json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
                    sys.exit("Stopped: the judge failed 3 times in a row (errors above). Failures were not saved.")
            else:
                fails = 0
                cache[key] = v
            if (i + 1) % 10 == 0:
                json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
        verdicts.append(v)
        if (i + 1) % 10 == 0 or i + 1 == n_needs:
            print("  judged %d of %d" % (i + 1, n_needs))
    if not dry:
        json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
    failed = sum(1 for v in verdicts if v and v.get("error"))
    if failed:
        sys.exit("%d needs could not be judged (errors above). Their answers were not saved; run the same command "
                 "again (tomorrow, if the daily limit was reached) and it will continue where it stopped." % failed)
    if only is not None:
        res = {"model": "%s + %s (%s)" % (EMBED_MODEL, JUDGE_MODEL, JUDGE_VIA), "judge_version": JUDGE_VERSION,
               "steps": [{"t": s[1]} for s in STEPS], "needs": []}
        for i, v in enumerate(verdicts):
            if v is None:
                continue
            step = int(shortlist[i][int(v["step"]) - 1]) if v["verdict"] != "none" else 0
            res["needs"].append({"t": need_texts[i], "verdict": v["verdict"], "step": step, "by": "model",
                                 "reason": v.get("reason", ""), "evidence": v.get("evidence", ""),
                                 "flag": v.get("flag", ""), "missing": v.get("missing_step", "")})
        path = os.path.join(HERE, "holdout_run.json")
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
        sys.exit("Held-out needs judged: %d. Wrote %s\nNext: python3 pipeline/eval_judge.py pipeline/holdout_run.json"
                 % (len(res["needs"]), path))

    # Second pass: for needs a step only partly answers, check whether other shortlisted steps fill the gap
    frag = {}
    if not dry:
        for i, text in enumerate(need_texts):
            v = verdicts[i]
            if v["verdict"] != "partly" or int(v["step"]) == 0:
                continue
            main_j = int(shortlist[i][int(v["step"]) - 1])
            other_j = [int(j) for j in shortlist[i] if int(j) != main_j]
            key = (FRAG_VERSION + "|" + JUDGE_MODEL + "||" + text + "||" + STEPS[main_j][1] + "||"
                   + "|".join(STEPS[j][1] for j in other_j))
            if key not in cache:
                try:
                    cache[key] = fragments(text, STEPS[main_j], [STEPS[j] for j in other_j])
                except RuntimeError as e:
                    json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
                    sys.exit("Second pass stopped: %s\nProgress is saved; run the same command again to continue." % e)
                json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
            f = cache[key]
            frag[i] = {"also": [other_j[k] for k in f["also"]], "fragmented": bool(f["also"]) and f["together"] == "answers"}
        json.dump(cache, open(CACHE, "w"), ensure_ascii=False)

    # Layout for the 'by meaning' view
    xyz = TSNE(n_components=3, perplexity=14, metric="cosine", init="pca",
               random_state=7).fit_transform(np.vstack([Ns, Ss]))
    xyz = xyz - np.median(xyz, 0)
    xyz = np.clip(xyz / np.percentile(np.abs(xyz), 97), -1.1, 1.1)

    titles = [s[1] for s in STEPS]
    out = {"model": "dry-run" if dry else "%s + %s (%s)" % (EMBED_MODEL, JUDGE_MODEL, JUDGE_VIA),
           "needs": [], "steps": [], "missing": []}
    for i, (stage, text) in enumerate(NEEDS):
        shop, ai, tier, dg, src = NEED_LENS[i].split()
        v = verdicts[i]
        if v["verdict"] != "none":
            step = int(shortlist[i][int(v["step"]) - 1])
        else:
            step = int(shortlist[i][0])
        rec = {"t": text, "stage": stage, "p": [round(float(c), 3) for c in xyz[i]],
               "step": step, "sim": round(float(sim[i, step]), 3),
               "verdict": v["verdict"], "gap": v["verdict"] == "none", "partly": v["verdict"] == "partly",
               "reason": v["reason"].strip(), "missing": v["missing_step"].strip() if v["verdict"] != "answers" else "",
               "needs_to_know": v.get("needs_to_know", ""), "step_gives": v.get("step_gives", ""),
               "evidence": v.get("evidence", ""), "flag": v.get("flag", ""),
               "also": frag.get(i, {}).get("also", []), "fragmented": frag.get(i, {}).get("fragmented", False),
               "by": "model",
               "shop": shop, "ai": ai, "tier": tier, "dg": dg, "src": src, "cat": CATS[i], "acc": ACC[i]}
        if i in OVERRIDES:
            verdict, title, note = OVERRIDES[i][:3]
            if title:
                rec["step"] = titles.index(title)
            rec.update(verdict=verdict, gap=verdict == "none", partly=verdict == "partly",
                       reason=note, by="researcher")
            if verdict == "answers":
                rec["missing"] = ""
            elif len(OVERRIDES[i]) > 3:
                rec["missing"] = OVERRIDES[i][3]
            rec["also"], rec["fragmented"] = [], False
        out["needs"].append(rec)
    for j, (stage, title, desc) in enumerate(STEPS):
        out["steps"].append({"t": title, "d": desc, "q": STEP_QUESTIONS[title], "stage": stage,
                             "p": [round(float(c), 3) for c in xyz[n_needs + j]],
                             "n": sum(1 for n in out["needs"] if n["step"] == j and not n["gap"]),
                             "shop": STEP_TYPE.get(title, "")})

    # Group the judge's suggested missing steps into proposals for the blueprint
    miss = [(i, n["missing"]) for i, n in enumerate(out["needs"]) if n["missing"]]
    if len(miss) >= 2:
        if dry:
            M = rnd(len(miss))
        else:
            M = embed(["task: clustering | query: " + m for _, m in miss], "missing steps")
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=0.10).fit_predict(M)  # 0.35 merged nearly all of a large model's specific names into one group
        for g in sorted(set(labels)):
            idx = [k for k in range(len(miss)) if labels[k] == g]
            centre = M[idx].mean(0)
            name = miss[idx[int(np.argmax(M[idx] @ centre))]][1]
            out["missing"].append({"name": name, "needs": [miss[k][0] for k in idx],
                                   "variants": sorted(set(miss[k][1] for k in idx))})
        out["missing"].sort(key=lambda g: -len(g["needs"]))
    elif miss:
        out["missing"].append({"name": miss[0][1], "needs": [miss[0][0]], "variants": [miss[0][1]]})

    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    c = Counter(n["verdict"] for n in out["needs"])
    print("\nWrote %s: %d needs. Answered %d, partly %d, none %d."
          % (OUT, n_needs, c["answers"], c["partly"], c["none"]))
    gaps = [n for n in out["needs"] if n["gap"]]
    for key in ("acc", "cat", "shop", "dg", "src"):
        cc = Counter(n[key] for n in out["needs"]); g = Counter(n[key] for n in gaps)
        print("  %-5s" % key, ", ".join("%s %d/%d" % (k, g[k], cc[k]) for k in sorted(cc)))
    fr = [n for n in out["needs"] if n["fragmented"]]
    print("  Answered only by joining steps: %d" % len(fr))
    for n in fr[:8]:
        print("    - %s -> %s" % (n["t"][:70], " + ".join(titles[j] for j in [n["step"]] + n["also"])))
    print("  Missing steps proposed:", "; ".join("%s (%d)" % (g["name"], len(g["needs"])) for g in out["missing"][:8]))
    print("\nNext: python3 page/build_page.py data/map.json data/twin.json")


if __name__ == "__main__":
    main()
