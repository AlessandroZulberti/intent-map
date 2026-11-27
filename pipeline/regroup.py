"""Regroup the judge's proposed missing steps in map.json, without judging again.

The grouping joins proposed steps whose names mean much the same. With short names that all start
"Check ...", a loose threshold merges nearly everything into one group. This tries several thresholds
and shows the result, so the grouping can be set to one that reads sensibly.

Uses Ollama for embeddings only: no OpenRouter requests.

  python3 pipeline/regroup.py                    # show the groups at several thresholds, change nothing
  python3 pipeline/regroup.py --threshold 0.2    # rewrite the "missing" groups in map.json at 0.2
"""
import json
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import build_map as B


def groups(M, miss, t):
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=t).fit_predict(M)
    out = []
    for g in sorted(set(labels)):
        idx = [k for k in range(len(miss)) if labels[k] == g]
        centre = M[idx].mean(0)
        name = miss[idx[int(np.argmax(M[idx] @ centre))]][1]
        out.append({"name": name, "needs": [miss[k][0] for k in idx],
                    "variants": sorted(set(miss[k][1] for k in idx))})
    out.sort(key=lambda g: -len(g["needs"]))
    return out


def main():
    m = json.load(open(B.OUT))
    miss = [(i, n["missing"]) for i, n in enumerate(m["needs"]) if n["missing"]]
    print("%d needs with a proposed missing step" % len(miss))
    M = B.embed(["task: clustering | query: " + t for _, t in miss], "missing steps")
    if "--threshold" in sys.argv:
        t = float(sys.argv[sys.argv.index("--threshold") + 1])
        m["missing"] = groups(M, miss, t)
        json.dump(m, open(B.OUT, "w"), ensure_ascii=False)
        print("Rewrote the missing-step groups in %s at threshold %.2f: %d groups" % (B.OUT, t, len(m["missing"])))
        for g in m["missing"][:10]:
            print("  %3d  %s" % (len(g["needs"]), g["name"]))
        return
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        gs = groups(M, miss, t)
        print("\nthreshold %.2f: %d groups; largest: %s" % (
            t, len(gs), ", ".join("%s (%d)" % (g["name"], len(g["needs"])) for g in gs[:6])))


if __name__ == "__main__":
    main()
