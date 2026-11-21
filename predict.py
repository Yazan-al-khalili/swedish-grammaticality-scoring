"""Score Swedish texts for grammatical correctness (0 = poor, 1 = good).

Usage: python predict.py --input test.jsonl --output scores.jsonl
"""
import argparse
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "yazanalkhalili/swedish-grammaticality-modernbert-base"


def pick_device():
    """Use the fastest chip available: NVIDIA GPU, Apple GPU (M1/M2/...), or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def score_texts(texts, tok, model, device, batch_size=64, max_length=128):
    """Return one score between 0 and 1 for each text, in the original order."""
    # Sort by length so each batch holds texts of similar length: less padding, faster.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores = [0.0] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        probs = torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().tolist()
        for i, p in zip(idx, probs):
            scores[i] = p
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL with one {'id', 'text'} per line")
    parser.add_argument("--output", required=True, help="JSONL to write one {'id', 'score'} per line")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local folder or Hugging Face model ID")
    args = parser.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    scores = score_texts([r["text"] for r in rows], tok, model, device)

    with open(args.output, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"id": r["id"], "score": float(s)}) + "\n")


if __name__ == "__main__":
    main()