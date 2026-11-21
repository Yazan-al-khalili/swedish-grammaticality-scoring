"""Create synthetic grammar errors: copy correct Swedish sentences and add 1-2 errors.

Each rule below makes one typical mistake. Because the copy differs from the
original only by the error, the model has to learn grammar, not writing style.
"""
import json
import random
import re

SEED = 42
SCORE_ONE_ERROR, SCORE_TWO_ERRORS = 0.40, 0.25   # same scale as AI Sweden's data

SUBJECTS = r"(jag|du|han|hon|vi|ni|de|man|det)"
FRONTED = r"(Igår|I går|Idag|I dag|Nu|Sedan|Då|Därför|Ibland|Ofta|Förra året|I morgon|Imorgon|Senare|Efteråt|Till slut)"


def wrong_gender(s):
    """Swap en/ett-word forms: min->mitt, din->ditt, sin->sitt, en->ett (and back)."""
    pairs = {"min": "mitt", "mitt": "min", "din": "ditt", "ditt": "din",
             "sin": "sitt", "sitt": "sin", "en": "ett", "ett": "en"}
    words = [m for m in re.finditer(r"\b(\w+)\b", s) if m.group(1).lower() in pairs]
    if not words:
        return None
    m = random.choice(words)
    new = pairs[m.group(1).lower()]
    if m.group(1)[0].isupper():
        new = new.capitalize()
    return s[:m.start()] + new + s[m.end():]


def wrong_verb_form(s):
    """Present tense -> infinitive after a subject: 'jag förstår' -> 'jag förstå'."""
    hits = [m for m in re.finditer(rf"\b{SUBJECTS}\s+(\w{{3,}}(?:ar|er|år|ör))\b", s, re.IGNORECASE)]
    if not hits:
        return None
    m = random.choice(hits)
    return s[:m.end(2) - 1] + s[m.end(2):]  


def wrong_word_order(s):
    """Break verb-second: 'Förra året reste vi' -> 'Förra året vi reste'."""
    m = re.match(rf"{FRONTED}\s+(\w+)\s+{SUBJECTS}\b", s)
    if not m:
        return None
    return f"{m.group(1)} {m.group(3)} {m.group(2)}" + s[m.end():]


def missing_definite_ending(s):
    """'den gamla bron' -> 'den gamla bro', 'det nya huset' -> 'det nya hus'."""
    hits = [m for m in re.finditer(r"\b(den|det)\s+(\w+a)\s+(\w{2,}?)(en|et|n|t)\b", s, re.IGNORECASE)]
    if not hits:
        return None
    m = random.choice(hits)
    return s[:m.start(4)] + s[m.end(4):]


def wrong_possessive(s):
    """Reflexive possessive -> non-reflexive: 'de sålde sitt hus' -> 'de sålde deras hus'."""
    m = re.search(r"\b(sin|sitt|sina)\b", s)
    if not m:
        return None
    return s[:m.start()] + "deras" + s[m.end():]


RULES = [wrong_gender, wrong_verb_form, wrong_word_order, missing_definite_ending, wrong_possessive]


def corrupt(sentence):
    """Apply one rule (sometimes two). Return (new_sentence, rule_names) or None."""
    rules = RULES[:]
    random.shuffle(rules)
    n_errors = 2 if random.random() < 0.25 else 1
    applied, text = [], sentence
    for rule in rules:
        new = rule(text)
        if new and new != text:
            text, applied = new, applied + [rule.__name__]
            if len(applied) == n_errors:
                break
    return (text, applied) if applied else None


def make_pairs(correct_sentences):
    """Return synthetic poor examples for every sentence a rule could change."""
    out = []
    for s in correct_sentences:
        result = corrupt(s)
        if result:
            text, applied = result
            score = SCORE_ONE_ERROR if len(applied) == 1 else SCORE_TWO_ERRORS
            out.append({"text": text, "score": score, "label": "poor",
                        "source": "synthetic", "error": ", ".join(applied)})
    return out


if __name__ == "__main__":
    random.seed(SEED)
    for split in ["train", "dev"]:
        rows = [json.loads(line) for line in open(f"my_data/{split}.jsonl", encoding="utf-8")]
        good = [r["text"] for r in rows if r["label"] == "good"]
        synthetic = make_pairs(good)
        with open(f"my_data/{split}_plus_synthetic.jsonl", "w", encoding="utf-8") as f:
            for r in rows + synthetic:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(good)} correct sentences -> {len(synthetic)} synthetic errors "
              f"-> my_data/{split}_plus_synthetic.jsonl ({len(rows) + len(synthetic)} examples)")