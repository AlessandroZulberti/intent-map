"""Score the judge in map.json against gold_labels.json.

Usage: python3 eval_judge.py [map.json or holdout_run.json]

Pass marks, set on 26 November 2026 BEFORE the strict-4-evidence run (renamed from strict-3, which collided with an old cache entry), apply to the holdout set only:
  - gap recall >= 80%   (gold partly/none that the judge does not call "answers")
  - false gaps <= 20%   (gold answers that the judge calls partly/none)
  - kappa >= 0.50       (verdict agreement beyond chance)
Baseline, strict-2 judge: holdout gap recall 54% (14 of 26), false gaps 7% (1 of 14), kappa 0.39.
Report the result whether it passes or not. Do not tune on the holdout set.
"""
import json
import os
import sys
from collections import Counter

V = ["answers", "partly", "none"]
MARKS = {"gap recall": 0.80, "false gaps": 0.20, "kappa": 0.50}


def score(gold, model):
    cm = Counter()
    step_hits = step_n = 0
    for g in gold:
        n = model.get(g["need"])
        if n is None:
            continue
        cm[(g["verdict"], n["verdict"])] += 1
        if g["verdict"] != "none" and n["verdict"] != "none" and g["step"]:
            step_n += 1
            step_hits += g["step"] == n["step_title"]
    t = sum(cm.values())
    po = sum(cm[(a, a)] for a in V) / t
    pe = sum(sum(cm[(a, b)] for b in V) * sum(cm[(b, a)] for b in V) for a in V) / t ** 2
    gaps = sum(cm[(a, b)] for a in ("partly", "none") for b in V)
    caught = gaps - sum(cm[(a, "answers")] for a in ("partly", "none"))
    ans = sum(cm[("answers", b)] for b in V)
    false = cm[("answers", "partly")] + cm[("answers", "none")]
    nones = sum(cm[("none", b)] for b in V)
    return {"n": t, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else 0,
            "gap recall": caught / gaps if gaps else 0, "gaps": (caught, gaps),
            "false gaps": false / ans if ans else 0, "false": (false, ans),
            "none found": (cm[("none", "none")], nones), "step match": (step_hits, step_n), "cm": cm}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = next((p for p in (os.path.join(here, "..", "data", "map.json"), os.path.join(here, "map.json")) if os.path.exists(p)), "map.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    m = json.load(open(path))
    titles = [s["t"] for s in m["steps"]]
    model = {n["t"]: dict(n, step_title=titles[n["step"]] if n["verdict"] != "none" else "") for n in m["needs"]}
    gold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_labels.json")))
    print("Judge: %s" % m.get("model", "?"))
    for name in [k for k in gold if k != "note"]:
        if not any(g["need"] in model for g in gold[name]):
            print("\n%s: not in this run" % name)
            continue
        r = score(gold[name], model)
        print("\n%s (%d needs)" % (name, r["n"]))
        print("  agreement %.0f%%  kappa %.2f" % (r["agreement"] * 100, r["kappa"]))
        print("  gap recall %.0f%% (%d of %d)  false gaps %.0f%% (%d of %d)" % (
            r["gap recall"] * 100, *r["gaps"], r["false gaps"] * 100, *r["false"]))
        print("  'none' found %d of %d   same step %d of %d" % (*r["none found"], *r["step match"]))
        print("  gold \\ judge   " + "  ".join("%7s" % v for v in V))
        for a in V:
            print("  %-13s" % a + "  ".join("%7d" % r["cm"][(a, b)] for b in V))
        if name.startswith("holdout"):
            ok = [r["gap recall"] >= MARKS["gap recall"], r["false gaps"] <= MARKS["false gaps"], r["kappa"] >= MARKS["kappa"]]
            print("  pass marks: gap recall %s, false gaps %s, kappa %s -> %s" % (
                *["pass" if o else "FAIL" for o in ok], "PASS" if all(ok) else "FAIL"))


if __name__ == "__main__":
    main()
