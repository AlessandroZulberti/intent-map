# Intent map

A service blueprint shows what a company does. It does not show whether any of it answers what customers ask.

Intent map adds that measure as a third direction. Customer needs are set against the steps of a pre-, during- and post-purchase blueprint. Open-source models place each need beside its nearest steps and judge whether any step actually answers it, with a reason. The result is a map of where the journey leaves customers without an answer, who owns each gap, and the steps that are missing.

**Live page:** https://alessandrozulberti.github.io/intent-map/
**Written up at:** https://alessandrozulberti.com

The example is **Larkhob, a fictional kitchen brand** made up to show the method, with a made-to-order and limited-edition line, Larkhob Atelier. Its products, customers, journey and every customer need in this repository are invented, not collected from real people or taken from any company. Any resemblance to an existing brand is unintended; if you think the name matches one, please [get in touch](https://alessandrozulberti.com/contact/) and it will be changed.

## What is in here

| Folder | What it holds |
|---|---|
| `pipeline/` | `build_map.py` runs the models; `larkhob_data.py` holds the steps, customer needs, coding and researcher overrides; `eval_judge.py`, `gold_labels.json` and `ablation.py` test the judge; `regroup.py` regroups the proposed missing steps without judging again |
| `frameworks/` | How to add your own framework as a lens, and a worked example |
| `page/` | `template.html` (the interactive page) and `build_page.py`, which bakes the data into `index.html` |
| `data/` | `map.json` (model results), `twin.json` (digital twin results) and `validation/` (the judge test runs) |
| `twins/` | The digital twin survey, its answer key, and `compare_twins.py` to read an ExploraTwin export |
| `index.html` | The built page, served by GitHub Pages |

## Run it

You need [Ollama](https://ollama.com) and Python 3.

```
ollama pull embeddinggemma
ollama pull gemma3:4b
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn
python3 pipeline/build_map.py
python3 page/build_page.py data/map.json data/twin.json
```

Open `index.html` in a browser. The judging step takes 10 to 15 minutes on an 8 GB Apple M1; answers are cached, so an interrupted run carries on.

To judge with a larger model, which the test below favours, keep the embeddings local and send the judging to [OpenRouter](https://openrouter.ai):

```
export OPENROUTER_API_KEY=your-key
python3 pipeline/build_map.py --judge openrouter --model nvidia/nemotron-3.5-super-vl:free
```

The free plan allows about 50 requests a day and a full run needs about 140, so it stops when the limit is reached; run the same command the next day and it continues. `python3 pipeline/list_free_models.py` lists the free models that can return the judge's format.

## Use your own model

- **Another local model:** change `EMBED_MODEL` and `JUDGE_MODEL` at the top of `pipeline/build_map.py` to any model Ollama serves.
- **A hosted model:** `--judge openrouter --model <id>` sends the judging to any OpenRouter model that supports structured outputs. For another provider, replace `chat()`. Check your provider's terms before sending customer data.
- **The prompt:** `--prompt strict-2` (default) or `--prompt strict-4`, which adds three rules and tested worse.

## Use your own data

Everything about the example lives in `pipeline/larkhob_data.py`:

- `STEPS`: your blueprint, as (stage, step, description)
- `STEP_QUESTIONS`: the questions each step is there to answer, in customers' words
- `LARKHOB`: your customer needs with their coding (shopper type, AI conversation type, price tier, agent delegation, where the answer lives), plus `CATS` and `ACC` for category and accessibility layer
- `OVERRIDES`: your own calls where you disagree with the model

Real needs make the map far more useful than written ones: site search, contact reasons, chat transcripts, survey answers. Remove personal data first, and weight the action list by how often each need is asked.

## Add your own framework

The lenses on the page are not fixed. Any framework your team uses (a conversion model, a service-quality scale, your own codes) can be added as a lens on the same needs, crossed with shopper type or AI conversation type, and switched on and off on the page. A framework you cannot publish stays in `private/`, which git ignores. See [`frameworks/README.md`](frameworks/README.md).

## How the method works

1. **Shortlist.** EmbeddingGemma embeds each need as a question and each step as an answer, and takes the five nearest steps.
2. **Judge.** A language model (Gemma 3 locally, or a larger model through OpenRouter) reads the need and the shortlist and decides: a step answers it, partly answers it, or none does. It writes what the customer needs to know, a one-line reason and, where nothing fits, the missing step. Its prompt follows NN/g's CARE structure (context, ask, rules, examples).
3. **Code.** Each need carries researcher coding: NN/g's ecommerce shopper types, NN/g's types of conversation with generative AI, price tier, agent delegation, where the answer lives, product category and accessibility layer (digital, product, service and place).
4. **Act.** Every need not fully answered gets a level, an owner and a first action by fixed rules shown on the page.
5. **Check.** The page includes a coding exercise for checking the judge, and a comparison with 500 digital twins from ExploraTwin.

## How reliable is the judge?

The map is only as good as its judge, so the judge was tested before anything was claimed for the method.

**The test.** 40 of the 101 judged needs were held back from all tuning. Claude labelled them blind before any change to the judge, and the pass marks were written into `pipeline/eval_judge.py` before the test ran: at least 80% of gaps found, no more than 20% false gaps, and agreement (Cohen's kappa, three verdicts) of at least 0.50. The 20 needs where Claude and the large model then disagreed were settled by the author, without knowing which label was whose. Both reference sets are kept in `pipeline/gold_labels.json` and reported side by side.

| Judge | Gaps found | False gaps | Kappa | Kappa, gap or not |
|---|---|---|---|---|
| Gemma 3 4B, strict-2 (the first live map) | 50% (54%) | 8% (7%) | 0.22 (0.39) | 0.32 (0.40) |
| Gemma 3 4B, strict-4 | 32% (35%) | 0% (0%) | 0.16 (0.25) | 0.22 (0.27) |
| Nemotron 3.5 Super VL, strict-4 | 93% (92%) | 42% (50%) | 0.36 (0.25) | 0.55 (0.46) |
| Nemotron 3.5 Super VL, strict-2 (the live map) | 86% (85%) | 25% (36%) | 0.41 (0.29) | 0.59 (0.50) |
| Pass mark | 80% | 20% | 0.50 | |

Scores against the reviewed labels, with Claude's original labels in brackets. 26 November 2026.

**What it shows.**

- No judge passed. The first live map was built with Gemma 3, which finds about half the gaps: it undercounted what the journey misses. The live map is now built with Nemotron 3.5 Super VL on the strict-2 prompt, the set-up that came closest; it leans the other way, calling some answered needs gaps.
- A large open-weight model finds the gaps but calls a quarter of answered needs gaps. Three rules added to rein in the small model (strict-4) made both models worse.
- Many disagreements came from steps the blueprint lacked. Five were added to Larkhob as a result: check what it can do, get told when it's back, change or cancel an order, learn from guides and articles, and get safety advice. They are findings of the method, not a fix to the score: the 40 test needs helped find them, so they cannot test them.

**Limits of the test.** 40 needs, so one need moves a rate by about four points; one run per model; and reference labels from the author and an AI, not independent coders. A second coder from outside the project is the next test. **Does the map notice a missing step?** `pipeline/ablation.py` takes one step at a time out of the journey and judges again the needs it answered, with pass marks set in advance. With Nemotron on the live map, five steps removed in turn: 10 of the 12 needs they answered became gaps (83%, pass mark 80%), and for all five the proposed missing step was among the three steps closest in meaning to the one removed (pass mark three of five). Twelve needs is a small test, and three of the five steps were themselves added after the review above. Results: `data/validation/ablation_results.json`.

**Grouping the missing steps.** A large model names a specific missing step for most open needs, so few are shared. The grouping now joins only names that mean almost the same (cosine distance 0.10); at the earlier 0.35 it merged 82 different steps into one.

To score a run yourself: `python3 pipeline/build_map.py --only-holdout ...` then `python3 pipeline/eval_judge.py pipeline/holdout_run.json`.

## Limits

- Larkhob's needs are written, not collected. The numbers show the method working, not facts about shoppers.
- The judge did not meet its pass marks (see above). Its reasons are shown and any call can be overridden.
- The coding is a researcher's judgement. A second coder would test it.
- Digital twins agree with real people only weakly at the level of the individual and lean towards technology. They test the coding; they are not customers.

## Credits

- NN/g: [ecommerce shopper types](https://www.nngroup.com/articles/ecommerce-shoppers/), [types of conversations with generative AI](https://www.nngroup.com/articles/AI-conversation-types/), [CARE prompt structure](https://www.nngroup.com/articles/careful-prompts/)
- [ExploraTwin](https://exploratwin.org), Columbia Business School Digital Twins Lab
- Twin panel: Berke, A. et al. Open e-commerce 1.0: Five years of crowdsourced U.S. Amazon purchase histories with user demographics. Harvard Dataverse, doi:10.7910/DVN/YGLYDY. Used for research. Only aggregated twin answers are published here.
- Models: Google EmbeddingGemma and Gemma 3, run through Ollama, under the Gemma Terms of Use; NVIDIA Nemotron 3.5 Super VL through OpenRouter, for the judge test

## Licence

Code (`pipeline/`, `page/`, `twins/*.py`) is under the MIT licence. Text, coding and data (this README, `larkhob_data.py` content, `data/`, the survey) are under CC BY 4.0. See `LICENSE`.

Alessandro Zulberti
