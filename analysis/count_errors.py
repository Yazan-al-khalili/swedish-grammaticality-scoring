import json
from collections import Counter
from huggingface_hub import hf_hub_download

# Get the train file
path = hf_hub_download(
    repo_id="sbx/superlim-2",
    repo_type="dataset",
    filename="data/dalaj-ged-superlim/dalaj-ged-superlim_train.jsonl",
)
rows = [json.loads(line) for line in open(path, encoding="utf-8")]

# Keeping only the incorrect sentences (correct ones have no error type)
incorrect = [r for r in rows if r["label"] == "incorrect"]

# Counts the main error types (the first letter: L, M, S, O, P)
main_types = Counter(r["meta"]["error_label"][:1] for r in incorrect)
print("Main error types:")
for kind, count in main_types.most_common():
    print(f"  {kind}: {count}")

# counts the detailed error types (example: M-Verb, S-WO)
detailed = Counter(r["meta"]["error_label"] for r in incorrect)
print("\n15 most common detailed error types:")
for kind, count in detailed.most_common(15):
    print(f"  {kind}: {count}")

# shows one example for each main type
print("\nOne example per main type:")
shown = set()
for r in incorrect:
    kind = r["meta"]["error_label"][:1]
    if kind not in shown:
        pair = r["meta"]["confusion_pair"]
        print(f"\n  [{r['meta']['error_label']}] {r['sentence']}")
        print(f"     wrong: '{pair['incorrect_span']}'  ->  correct: '{pair['correction']}'")
        shown.add(kind)