"""Turn an ExploraTwin export into data/twin.json for the page.
Usage: python3 twins/compare_twins.py path/to/long.csv
long.csv is in the export bundle ExploraTwin gives you after a run of twins/larkhob_survey.json."""
import csv, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = {"source": {"Larkhob's own website or customer service": "B", "Reviews, forums or independent experts": "X",
                  "Both, about equally": "M"},
       "delegate": {"Let it handle it for me": "A", "Let it help me look into it, then decide myself": "S",
                    "Deal with it myself, without the AI": "K"}}

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
cnt = {q: defaultdict(Counter) for q in LAB}
for r in rows:
    q, rid = r["question_id"].split("_", 1)
    k = LAB.get(q, {}).get(r["answer_label"])
    if k:
        cnt[q][rid][k] += 1
key = list(csv.DictReader(open(os.path.join(HERE, "larkhob_answer_key.csv"), encoding="utf-8")))
out = [{"row": k["row"], "need": int(k["need_index"]), "t": k["text_uk"],
        "src_r": k["researcher_source"], "dg_r": k["researcher_delegation"],
        "src": dict(cnt["source"][k["row"]]), "dg": dict(cnt["delegate"][k["row"]])} for k in key]
twins = len({r["persona_id"] for r in rows})
json.dump({"twins": twins, "panel": "Open e-commerce 1.0", "rows": out},
          open(os.path.join(HERE, "..", "data", "twin.json"), "w"), ensure_ascii=False)
top = lambda o: max(o, key=o.get) if o else None
print("%d twins. Source agreement %d of %d; delegation agreement %d of %d." % (
    twins, sum(top(x["src"]) == x["src_r"] for x in out), len(out), sum(top(x["dg"]) == x["dg_r"] for x in out), len(out)))
