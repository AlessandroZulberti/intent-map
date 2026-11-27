"""List OpenRouter's free models that support structured outputs (needed by the judge).

Usage: python3 pipeline/list_free_models.py
No key needed. The list changes often, so run it on the day you test.
"""
import json
import urllib.request

with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
    models = json.load(r)["data"]
free = [m for m in models if m["id"].endswith(":free") and "structured_outputs" in m.get("supported_parameters", [])]
free.sort(key=lambda m: -m.get("context_length", 0))
print("%d free models with structured outputs\n" % len(free))
for m in free:
    print("%-55s %s" % (m["id"], (m.get("name") or "")[:50]))
print("\nPrefer the largest general model you recognise (look for a size such as 70B or 120B in the name).")
print("Avoid tiny models (1B to 8B): they will be no better than gemma3:4b.")
