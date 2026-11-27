"""Step-deletion test: does the map notice when a step is taken out of the journey?

For each removed step, the needs that the current map says it answers are judged again with that step gone.
If the method works, those needs should stop being "answers", and the missing step the judge proposes
should point back to the step that was removed.

Usage (same judge options as build_map.py):
  python3 pipeline/ablation.py --judge openrouter --model nvidia/nemotron-3.5-super-vl:free
  python3 pipeline/ablation.py ... --steps "Buy spares" "Return for a refund"     # choose the steps yourself
  python3 pipeline/ablation.py ... --dry-run                                     # check it runs, no API calls

Default: the 5 steps that answer the most needs in map.json (researcher calls excluded).
Writes ablation_results.json next to this script. Uses judge_cache.json, so an interrupted run continues where it stopped.

Pass marks, set on 26 November 2026 before any ablation run:
  - detection: at least 80% of affected needs stop being "answers" once their step is removed
  - recovery: for at least 3 of the 5 removed steps, the removed step is among the 3 steps closest in meaning
    to the judge's proposed missing steps for its needs
Report the result whether it passes or not.
"""
import json
import os
import sys

import numpy as np

import build_map as B
from larkhob_data import NEEDS, STEPS, STEP_QUESTIONS

ABL_VERSION = "ablation-1"


def steps_to_remove(m):
    if "--steps" in sys.argv:
        names = sys.argv[sys.argv.index("--steps") + 1:]
        names = [n for n in names if not n.startswith("--")]
        titles = [s[1] for s in STEPS]
        bad = [n for n in names if n not in titles]
        if bad:
            sys.exit("Not a step: %s" % ", ".join(bad))
        return names
    count = {}
    for n in m["needs"]:
        if n["by"] == "model" and n["verdict"] == "answers":
            t = m["steps"][n["step"]]["t"]
            count[t] = count.get(t, 0) + 1
    return [t for t, _ in sorted(count.items(), key=lambda x: -x[1])[:5]]


def main():
    dry = "--dry-run" in sys.argv
    m = json.load(open(B.OUT))
    if not dry and B.JUDGE_MODEL not in m.get("model", ""):
        print("Note: map.json was built with %s; this test judges with %s." % (m.get("model"), B.JUDGE_MODEL))
    need_texts = [t for _, t in NEEDS]
    titles = [s[1] for s in STEPS]
    removed = steps_to_remove(m)
    print("Steps removed one at a time: %s" % "; ".join(removed))

    if dry:
        rng = np.random.default_rng(0)
        def rnd(k):
            v = rng.normal(size=(k, 64)); return v / np.linalg.norm(v, axis=1, keepdims=True)
        Nq, Sd = rnd(len(need_texts)), rnd(len(STEPS))
    else:
        Nq = B.embed(["task: question answering | query: " + t for t in need_texts], "needs as questions")
        Sd = B.embed(["title: %s | text: %s %s" % (a, b, STEP_QUESTIONS[a]) for _, a, b in STEPS], "steps as answers")
    sim = Nq @ Sd.T

    cache = json.load(open(B.CACHE)) if os.path.exists(B.CACHE) and not dry else {}
    results = []
    for title in removed:
        j_out = titles.index(title)
        affected = [i for i, n in enumerate(m["needs"])
                    if n["by"] == "model" and n["verdict"] == "answers" and m["steps"][n["step"]]["t"] == title]
        print("\n%s: %d needs to judge again" % (title, len(affected)))
        rows = []
        for i in affected:
            order = [j for j in np.argsort(-sim[i]) if j != j_out][:B.SHORTLIST]
            opts = [STEPS[j] for j in order]
            key = "%s|%s|%s||%s||without %s" % (ABL_VERSION, B.JUDGE_VERSION, B.JUDGE_MODEL, need_texts[i], title)
            if dry:
                v = {"verdict": ["answers", "partly", "none"][i % 3], "step": 1, "missing_step": "Check something"}
            elif key in cache and not cache[key].get("error"):
                v = cache[key]
            else:
                v = B.judge(need_texts[i], opts)
                if v.get("error"):
                    json.dump(cache, open(B.CACHE, "w"), ensure_ascii=False)
                    sys.exit("Stopped on a judge error (above). Progress is saved; run the same command again.")
                cache[key] = v
                json.dump(cache, open(B.CACHE, "w"), ensure_ascii=False)
            now = STEPS[order[int(v["step"]) - 1]][1] if v["verdict"] != "none" else ""
            rows.append({"need": need_texts[i], "verdict": v["verdict"], "step_now": now,
                         "missing": v.get("missing_step", "").strip() if v["verdict"] != "answers" else ""})
            print("  %-7s %-30s | %s" % (v["verdict"], (now or "-")[:30], need_texts[i][:70]))

        # Recovery: is the removed step close in meaning to what the judge says is missing?
        missing = [r["missing"] for r in rows if r["missing"]]
        rank = None
        if missing and not dry:
            Mv = B.embed(["title: %s | text: %s" % (t, t) for t in missing], "proposed missing steps")
            score = (Mv @ Sd.T).mean(axis=0)
            rank = int(np.where(np.argsort(-score) == j_out)[0][0]) + 1
        flipped = sum(1 for r in rows if r["verdict"] != "answers")
        results.append({"removed": title, "affected": len(rows), "no_longer_answered": flipped,
                        "recovery_rank": rank, "proposed_missing": missing, "needs": rows})
        print("  no longer answered: %d of %d; removed step ranks %s of %d against the proposed missing steps"
              % (flipped, len(rows), rank if rank else "-", len(STEPS)))

    json.dump({"judge": B.JUDGE_MODEL, "prompt": B.PROMPT, "results": results},
              open(os.path.join(B.HERE, "ablation_results.json"), "w"), ensure_ascii=False, indent=1)
    tot = sum(r["affected"] for r in results)
    flip = sum(r["no_longer_answered"] for r in results)
    rec = sum(1 for r in results if r["recovery_rank"] and r["recovery_rank"] <= 3)
    print("\nDetection: %d of %d affected needs no longer answered (%.0f%%). Pass mark 80%%: %s"
          % (flip, tot, 100.0 * flip / tot if tot else 0, "pass" if tot and flip / tot >= 0.8 else "FAIL"))
    print("Recovery: removed step in the top 3 for %d of %d steps. Pass mark 3 of 5: %s"
          % (rec, len(results), "pass" if rec >= 3 else "FAIL"))
    print("Wrote ablation_results.json")


if __name__ == "__main__":
    main()
