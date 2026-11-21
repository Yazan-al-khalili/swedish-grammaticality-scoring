import json
from collections import Counter
from huggingface_hub import hf_hub_download

# download three DaLAJ files from Hugging Face (only the first time)
files = {}
for split in ["train", "dev", "test"]:
    files[split] = hf_hub_download(
        repo_id="sbx/superlim-2",
        repo_type="dataset",
        filename=f"data/dalaj-ged-superlim/dalaj-ged-superlim_{split}.jsonl",
    )

# Reads each file and show how big it is and what the labels are
for split, path in files.items():
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    labels = Counter(str(r.get("label")) for r in rows)
    print(f"{split}: {len(rows)} sentences, labels: {dict(labels)}")

# Showing what the first few examples look like
print("\nFirst 5 examples from train:\n")
train = [json.loads(line) for line in open(files["train"], encoding="utf-8")]
for r in train[:5]:
    print(r)
