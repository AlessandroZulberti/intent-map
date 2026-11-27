# Add your own framework

Intent map's built-in lenses come from NN/g. You can add any other framework your team uses, as a lens on the same needs, without changing the code.

## The file

One JSON file per framework:

```json
{
  "key": "effort",
  "name": "Effort to answer",
  "hint": "One or two sentences on what the lens shows.",
  "credit": "Where the framework comes from.",
  "codes": {
    "F": {"n": "A fact", "d": "the brand can state it once"},
    "J": {"n": "A judgement", "d": "needs advice"}
  },
  "cross": ["shop", "ai"],
  "hide_builtin": [],
  "needs": {
    "Is the blender jug BPA-free?": "F"
  }
}
```

- `key`: a short lower-case name, unique on the page.
- `codes`: each code's name and a short description, shown on the page.
- `needs`: each need's exact wording (as in `pipeline/larkhob_data.py`) and its code. Needs you leave out are simply not coded.
- `cross` (optional): built-in lenses to cross the framework with, as tables of needs not fully answered: `shop`, `ai`, `tier`, `dg`, `src`, `cat`, `acc`.
- `hide_builtin` (optional): built-in lenses to hide, when your framework replaces them.

## Build with it

```
python3 page/build_page.py data/map.json data/twin.json --framework frameworks/example-effort.json
```

Repeat `--framework` to add several. On the page, a row of buttons lets the reader switch each framework on and off.

## Keeping a framework private

Put private framework files in `private/` at the root of this repository. Git ignores that folder, so they are never committed here. Build a private page with `--out`, so `index.html` stays as published:

```
python3 page/build_page.py data/map.json data/twin.json --framework private/frameworks/yours.json --out private/index.html
```
