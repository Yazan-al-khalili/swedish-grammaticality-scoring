import json

# read the three files: texts, correct answers, and the models scores
texts  = {r["id"]: r["text"] for r in map(json.loads, open("data/test.jsonl", encoding="utf-8"))}
labels = {r["id"]: r for r in map(json.loads, open("data/test_labels.jsonl", encoding="utf-8"))}
scores = {r["id"]: r["score"] for r in map(json.loads, open("scores.jsonl", encoding="utf-8"))}

# split into good and poor texts, sorted by the models score
good = sorted([i for i in labels if labels[i]["label"] == "good"], key=lambda i: scores[i])
poor = sorted([i for i in labels if labels[i]["label"] == "poor"], key=lambda i: scores[i], reverse=True)

# Shows the worst mistakes
print("GOOD texts the model scored LOWEST (it thought they were bad):\n")
for i in good[:5]:
    print(f"  model: {scores[i]:.2f}  correct: {labels[i]['score']:.2f}  |  {texts[i]}")

print("\nPOOR texts the model scored HIGHEST (it missed the errors):\n")
for i in poor[:5]:
    print(f"  model: {scores[i]:.2f}  correct: {labels[i]['score']:.2f}  |  {texts[i]}")