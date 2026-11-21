import json
import os
from collections import Counter
from huggingface_hub import hf_hub_download

ALLOWED_ERRORS = {"L", "M", "S", "O"}   # droppin punctuation
GOOD_SCORE, BAD_SCORE = 0.95, 0.40      


def load_dalaj(split):
    """Download one DaLAJ file and return a cleaned list of examples."""
    path = hf_hub_download(
        repo_id="sbx/superlim-2",
        repo_type="dataset",
        filename=f"data/dalaj-ged-superlim/dalaj-ged-superlim_{split}.jsonl",
    )
    examples, skipped = [], Counter()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["label"] == "correct":
            examples.append({"text": r["sentence"], "score": GOOD_SCORE,
                             "label": "good", "source": "dalaj"})
            continue
        # Incorrect sentence: check its error type(s), e.g. "M" or "L, S"
        types = [t.strip() for t in r["meta"]["error_label"].split(",")]
        if types and all(t in ALLOWED_ERRORS for t in types):
            examples.append({"text": r["sentence"], "score": BAD_SCORE,
                             "label": "poor", "source": "dalaj"})
        else:
            skipped[r["meta"]["error_label"] or "(empty)"] += 1
    return examples, skipped


def load_ai_sweden():
    """Load AI Sweden's 478 training examples, keeping their own scores."""
    examples = []
    for line in open("data/train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        examples.append({"text": r["text"], "score": r["score"],
                         "label": r["label"], "source": "ai_sweden"})
    return examples


def save(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.makedirs("my_data", exist_ok=True)

    dalaj_train, skipped_train = load_dalaj("train")
    dalaj_dev, skipped_dev = load_dalaj("dev")
    train = dalaj_train + load_ai_sweden()

    save(train, "my_data/train.jsonl")
    save(dalaj_dev, "my_data/dev.jsonl")

    print("Removed from DaLAJ train:", dict(skipped_train))
    print(f"\nmy_data/train.jsonl: {len(train)} examples",
          dict(Counter(r["label"] for r in train)))
    print(f"my_data/dev.jsonl:   {len(dalaj_dev)} examples",
          dict(Counter(r["label"] for r in dalaj_dev)))