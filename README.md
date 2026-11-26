# Intent map

A service blueprint shows what a company does. It does not show whether any of it answers what customers ask.

Intent map adds that measure as a third direction. Customer needs are set against the steps of a pre-, during- and post-purchase blueprint. Open-source models place each need beside its nearest steps and judge whether any step actually answers it, with a reason. The result is a map of where the journey leaves customers without an answer, who owns each gap, and the steps that are missing.

**Live page:** https://alessandrozulberti.github.io/intent-map/
**Written up at:** https://alessandrozulberti.com

The example is **Larkhob, a fictional kitchen brand** made up to show the method, with a made-to-order and limited-edition line, Larkhob Atelier. Its products, customers, journey and every customer need in this repository are invented, not collected from real people or taken from any company. Any resemblance to an existing brand is unintended; if you think the name matches one, please [get in touch](https://alessandrozulberti.com/contact/) and it will be changed.

## What is in here

| Folder | What it holds |
|---|---|
| `pipeline/` | `build_map.py` runs the models; `larkhob_data.py` holds the steps, customer needs, coding and researcher overrides |
| `page/` | `template.html` (the interactive page) and `build_page.py`, which bakes the data into `index.html` |
| `data/` | `map.json` (model results) and `twin.json` (digital twin results) |
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

## Use your own model

- **Another local model:** change `EMBED_MODEL` and `JUDGE_MODEL` at the top of `pipeline/build_map.py` to any model Ollama serves.
- **A hosted API:** replace the two functions `embed()` and `judge()`. Nothing else depends on Ollama. Check your provider's terms before sending customer data.

## Use your own data

Everything about the example lives in `pipeline/larkhob_data.py`:

- `STEPS`: your blueprint, as (stage, step, description)
- `STEP_QUESTIONS`: the questions each step is there to answer, in customers' words
- `LARKHOB`: your customer needs with their coding (shopper type, AI conversation type, price tier, agent delegation, where the answer lives), plus `CATS` and `ACC` for category and accessibility layer
- `OVERRIDES`: your own calls where you disagree with the model

Real needs make the map far more useful than written ones: site search, contact reasons, chat transcripts, survey answers. Remove personal data first, and weight the action list by how often each need is asked.

## How the method works

1. **Shortlist.** EmbeddingGemma embeds each need as a question and each step as an answer, and takes the five nearest steps.
2. **Judge.** Gemma 3 reads the need and the shortlist and decides: a step answers it, partly answers it, or none does. It writes what the customer needs to know, a one-line reason and, where nothing fits, the missing step. Its prompt follows NN/g's CARE structure (context, ask, rules, examples).
3. **Code.** Each need carries researcher coding: NN/g's ecommerce shopper types, NN/g's types of conversation with generative AI, price tier, agent delegation, where the answer lives, product category and accessibility layer (digital, product, service and place).
4. **Act.** Every need not fully answered gets a level, an owner and a first action by fixed rules shown on the page.
5. **Check.** The page includes a coding exercise for checking the judge, and a comparison with 500 digital twins from ExploraTwin.

## Limits

- Larkhob's needs are written, not collected. The numbers show the method working, not facts about shoppers.
- The judge is a small model. It is fallible; its reasons are shown and any call can be overridden.
- The coding is a researcher's judgement. A second coder would test it.
- Digital twins agree with real people only weakly at the level of the individual and lean towards technology. They test the coding; they are not customers.

## Credits

- NN/g: [ecommerce shopper types](https://www.nngroup.com/articles/ecommerce-shoppers/), [types of conversations with generative AI](https://www.nngroup.com/articles/AI-conversation-types/), [CARE prompt structure](https://www.nngroup.com/articles/careful-prompts/)
- [ExploraTwin](https://exploratwin.org), Columbia Business School Digital Twins Lab
- Twin panel: Berke, A. et al. Open e-commerce 1.0: Five years of crowdsourced U.S. Amazon purchase histories with user demographics. Harvard Dataverse, doi:10.7910/DVN/YGLYDY. Used for research. Only aggregated twin answers are published here.
- Models: Google EmbeddingGemma and Gemma 3, run through Ollama, under the Gemma Terms of Use

## Licence

Code (`pipeline/`, `page/`, `twins/*.py`) is under the MIT licence. Text, coding and data (this README, `larkhob_data.py` content, `data/`, the survey) are under CC BY 4.0. See `LICENSE`.

Alessandro Zulberti
