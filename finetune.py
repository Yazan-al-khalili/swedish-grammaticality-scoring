"""Fine-tune a Swedish encoder to score grammatical correctness (0 = poor, 1 = good)."""
import argparse
import json
import random

import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)


def pick_device():
    """Use the fastest chip available: NVIDIA GPU, Apple GPU (M1/M2/...), or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path, limit=None, seed=42):
    """Read a JSONL file. With a limit, take a random sample (the files are sorted,
    so the first N lines would not be a fair mix of good and poor texts)."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


@torch.no_grad()
def predict_scores(texts, tok, model, device, batch_size=64, max_length=128):
    """Return one score between 0 and 1 for each text."""
    model.eval()
    scores = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits.squeeze(-1)
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
    return scores


def evaluate(rows, tok, model, device):
    """Spearman (is the order right?) and ROC-AUC (are good and poor separated?)."""
    scores = predict_scores([r["text"] for r in rows], tok, model, device)
    spearman = spearmanr([r["score"] for r in rows], scores).correlation
    auc = roc_auc_score([r["label"] == "good" for r in rows], scores)
    return spearman, auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="AI-Sweden-Models/ModernBERT-base")
    parser.add_argument("--train", default="my_data/train.jsonl")
    parser.add_argument("--dev", default="my_data/dev.jsonl")
    parser.add_argument("--output", default="model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N training examples (for a quick test run)")
    parser.add_argument("--dev-limit", type=int, default=1000,
                        help="score a random sample of N dev examples after each epoch (saves time)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)  # makes the run reproducible (python, numpy and torch)
    device = pick_device()
    print(f"Training on: {device}")

    train_rows = load_jsonl(args.train, args.limit, args.seed)
    dev_rows = load_jsonl(args.dev, args.dev_limit, args.seed)
    print(f"{len(train_rows)} training examples, {len(dev_rows)} dev examples")
    print("  labels in train sample:", {l: sum(r['label'] == l for r in train_rows) for l in ('good', 'poor')})

    tok = AutoTokenizer.from_pretrained(args.model)
    # num_labels=1: the head outputs one number, which sigmoid turns into a 0-1 score
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)

    def collate(batch):
        enc = tok([r["text"] for r in batch], padding=True, truncation=True,
                  max_length=args.max_length, return_tensors="pt")
        enc["targets"] = torch.tensor([r["score"] for r in batch], dtype=torch.float)
        return enc

    generator = torch.Generator().manual_seed(args.seed)  # same shuffle order every run
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=generator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # compares sigmoid(output) with the 0-1 target

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(loader, start=1):
            targets = batch.pop("targets").to(device)
            batch = batch.to(device)
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0 or step == len(loader):
                print(f"epoch {epoch}  step {step}/{len(loader)}  loss {loss.item():.4f}")

        spearman, auc = evaluate(dev_rows, tok, model, device)
        print(f"epoch {epoch} dev:  Spearman {spearman:.3f}  ROC-AUC {auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            model.save_pretrained(args.output)
            tok.save_pretrained(args.output)
            print(f"  saved best model so far to '{args.output}/'")


if __name__ == "__main__":
    main()