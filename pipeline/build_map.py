"""Build the Intent map data file locally with Ollama.

Two models, both free and local:
  embeddinggemma  places every need and step by meaning, and shortlists the five steps nearest each need
  gemma3:4b       reads each need with its shortlist and judges which step answers it, if any

Usage:
  python3 pipeline/build_map.py              # full run, writes data/map.json (judge answers cached in pipeline/judge_cache.json)
  python3 pipeline/build_map.py --dry-run    # random vectors and a fake judge, to test without Ollama
Then: python3 page/build_page.py data/map.json data/twin.json   # bakes the data into index.html

To use another model, change EMBED_MODEL and JUDGE_MODEL to any Ollama model. To use a hosted API instead,
replace the two functions embed() and judge(): everything else stays the same.

Setup: Ollama running, `ollama pull embeddinggemma`, `ollama pull gemma3:4b`, `pip install numpy scikit-learn`.
"""
import json
import os
import sys
import urllib.request
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from larkhob_data import STEPS, NEEDS, NEED_LENS, STEP_TYPE, CATS, ACC, STEP_QUESTIONS, OVERRIDES

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
JUDGE_MODEL = "gemma3:4b"
SHORTLIST = 5
HERE = os.path.dirname(os.path.abspath(__file__))
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


def embed(texts, label):
    vecs = []
    for i in range(0, len(texts), 16):
        vecs.extend(post("/api/embed", {"model": EMBED_MODEL, "input": texts[i:i + 16]})["embeddings"])
        print("  %s: embedded %d of %d" % (label, min(i + 16, len(texts)), len(texts)))
    v = np.array(vecs, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


JUDGE_VERSION = "strict-2"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_needs_to_know": {"type": "string"},
        "best_step": {"type": "integer", "minimum": 0, "maximum": SHORTLIST},
        "what_that_step_gives": {"type": "string"},
        "verdict": {"type": "string", "enum": ["answers", "partly", "none"]},
        "reason": {"type": "string"},
        "missing_step": {"type": "string"},
    },
    "required": ["customer_needs_to_know", "best_step", "what_that_step_gives", "verdict", "reason", "missing_step"],
}

EXAMPLES = """Examples from other shops (not Larkhob):

Need: "Will this rug's colour run if I wash it at 40 degrees?"
Steps: 1. Clean and care: Clean it and look after it so it lasts. 2. Check size and weight: Check the size, weight and what it is made of.
customer_needs_to_know: whether the dye is colourfast at 40 degrees.
best_step: 1. what_that_step_gives: washing and care instructions, which normally state the safe temperature.
verdict: answers. missing_step: ""

Need: "I'm 1.95 m tall. Will this bike frame fit me, and can I test ride it?"
Steps: 1. Check size and weight: Check the size and dimensions. 2. Ask before buying: Ask the shop a question.
customer_needs_to_know: which frame size suits their height, and whether a test ride is possible.
best_step: 1. what_that_step_gives: frame dimensions, but not a size guide by height and nothing about test rides.
verdict: partly. missing_step: "Find my size and try it"

Need: "I use a wheelchair. Can I reach the controls on this sofa's recliner from a seated transfer?"
Steps: 1. Check size and weight: Check the size and dimensions. 2. Learn to use it: Learn how to use it from instructions.
customer_needs_to_know: whether the controls are reachable and usable for someone transferring from a wheelchair.
best_step: 0. what_that_step_gives: dimensions and operating instructions, which say nothing about reach or use from a wheelchair.
verdict: none. missing_step: "Check it works for my body"
"""


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
        "answered by a general step: check carefully.\n\n"
        "%s\n"
        "Now the real case.\nNeed: \"%s\"\nSteps:\n%s\n\n"
        "Fill every field in order. customer_needs_to_know: the specific thing they must find out or do. "
        "what_that_step_gives: what the best step would actually tell or let them do. reason: one plain sentence "
        "under 25 words. missing_step: if the verdict is not \"answers\", the step the journey is missing in 2 to 6 "
        "words, written as a customer action; otherwise an empty string."
        % (EXAMPLES, need, listing))
    for attempt in range(2):
        r = post("/api/chat", {"model": JUDGE_MODEL, "stream": False, "format": JUDGE_SCHEMA,
                               "options": {"temperature": 0},
                               "messages": [{"role": "user", "content": prompt}]})
        try:
            out = json.loads(r["message"]["content"])
            v, k = out["verdict"], int(out["best_step"])
            if v in ("answers", "partly", "none") and 0 <= k <= len(options):
                if v != "none" and k == 0:
                    v = "none"
                if v == "none":
                    k = 0
                return {"verdict": v, "step": k, "reason": out["reason"], "missing_step": out["missing_step"],
                        "needs_to_know": out["customer_needs_to_know"], "step_gives": out["what_that_step_gives"]}
        except Exception:
            pass
    return {"verdict": "none", "step": 0, "reason": "The judge could not decide.", "missing_step": "",
            "needs_to_know": "", "step_gives": ""}


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
    verdicts = []
    print("Judging %d needs with %s..." % (n_needs, JUDGE_MODEL))
    for i, text in enumerate(need_texts):
        opts = [STEPS[j] for j in shortlist[i]]
        key = JUDGE_VERSION + "||" + text + "||" + "|".join(s[1] for s in opts)
        if dry:
            v = {"verdict": ["answers", "partly", "none"][i % 3], "step": 0 if i % 3 == 2 else 1,
                 "reason": "Fake judge.", "missing_step": "" if i % 3 == 0 else "Check it works for me"}
        elif key in cache:
            v = cache[key]
        else:
            v = judge(text, opts)
            cache[key] = v
            if (i + 1) % 10 == 0:
                json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
        verdicts.append(v)
        if (i + 1) % 10 == 0 or i + 1 == n_needs:
            print("  judged %d of %d" % (i + 1, n_needs))
    if not dry:
        json.dump(cache, open(CACHE, "w"), ensure_ascii=False)

    # Layout for the 'by meaning' view
    xyz = TSNE(n_components=3, perplexity=14, metric="cosine", init="pca",
               random_state=7).fit_transform(np.vstack([Ns, Ss]))
    xyz = xyz - np.median(xyz, 0)
    xyz = np.clip(xyz / np.percentile(np.abs(xyz), 97), -1.1, 1.1)

    titles = [s[1] for s in STEPS]
    out = {"model": "dry-run" if dry else "%s + %s" % (EMBED_MODEL, JUDGE_MODEL),
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
               "by": "model",
               "shop": shop, "ai": ai, "tier": tier, "dg": dg, "src": src, "cat": CATS[i], "acc": ACC[i]}
        if i in OVERRIDES:
            verdict, title, note = OVERRIDES[i]
            if title:
                rec["step"] = titles.index(title)
            rec.update(verdict=verdict, gap=verdict == "none", partly=verdict == "partly",
                       reason=note, by="researcher")
            if verdict == "answers":
                rec["missing"] = ""
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
                                         distance_threshold=0.35).fit_predict(M)
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
    print("  Missing steps proposed:", "; ".join("%s (%d)" % (g["name"], len(g["needs"])) for g in out["missing"][:8]))
    print("\nNext: python3 page/build_page.py data/map.json data/twin.json")


if __name__ == "__main__":
    main()
