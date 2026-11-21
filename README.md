# Swedish Grammaticality Scoring

A model that reads Swedish text and returns a score between 0 and 1, where grammatically correct text scores higher. Built as a selection task for AI Sweden's master's thesis positions.

- **Model:** [yazanalkhalili/swedish-grammaticality-modernbert-base](https://huggingface.co/yazanalkhalili/swedish-grammaticality-modernbert-base)
- **Training data:** [yazanalkhalili/swedish-grammaticality-data](https://huggingface.co/datasets/yazanalkhalili/swedish-grammaticality-data)

## Results

On the 120 practice examples in `data/test.jsonl`, measured with `evaluate.py`:

| Model | Spearman | ROC-AUC | Samples/min |
|---|---|---|---|
| Baseline (ModernBERT-large embeddings + SVR) | 0.528 | 0.783 | 101 (Apple M1) |
| **This model** (fine-tuned ModernBERT-base) | **0.822** | **0.957** | 901 (Colab T4 GPU) |

Speed includes model loading. The two speeds were measured on different hardware and are not directly comparable; on the same Apple M1, a model of the same size ran at about 1,194 samples/min, roughly 12 times faster than the baseline.

## Approach

1. **Fine-tune instead of freezing.** The baseline keeps a large model frozen and trains a small regressor on its sentence embeddings. Those embeddings mostly capture meaning, so the baseline missed obvious errors such as *"Jag förstå inte."* (scored 0.78). Here, [AI-Sweden-Models/ModernBERT-base](https://huggingface.co/AI-Sweden-Models/ModernBERT-base) is fine-tuned end to end with a one-output head, and a sigmoid turns the output into a score between 0 and 1.
2. **Real learner errors.** Training data comes mainly from DaLAJ-GED (Swedish learner sentences labelled correct or incorrect).
3. **Synthetic minimal pairs.** Training only on DaLAJ reached 0.79 ROC-AUC on DaLAJ but only 0.58 on the practice set, likely because the model learned to tell textbook text from learner text instead of grammar. `make_errors.py` therefore copies correct sentences and adds one or two rule-based errors, so the correct and incorrect versions differ only by the error.

## Repository structure

| File | Purpose |
|---|---|
| `predict.py` | Scores texts (the required submission interface) |
| `prepare_data.py` | Downloads and cleans DaLAJ, adds AI Sweden's 478 training examples |
| `make_errors.py` | Adds synthetic errors to correct sentences |
| `finetune.py` | Fine-tunes the model and saves the best checkpoint |
| `evaluate.py` | AI Sweden's evaluation script (unchanged) |
| `data/` | AI Sweden's starter data (training examples and practice test set) |
| `analysis/` | Small scripts used to inspect the data and the baseline's errors |

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python predict.py --input data/test.jsonl --output scores.jsonl
python evaluate.py
```

`predict.py` downloads the model from Hugging Face on the first run and uses an NVIDIA GPU, an Apple GPU or the CPU, whichever is available. On a Mac, prefix commands with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Reproduce training from scratch

```bash
python prepare_data.py   # -> my_data/train.jsonl, my_data/dev.jsonl
python make_errors.py    # -> my_data/train_plus_synthetic.jsonl, my_data/dev_plus_synthetic.jsonl
python finetune.py --train my_data/train_plus_synthetic.jsonl --dev my_data/dev_plus_synthetic.jsonl \
                   --epochs 1 --batch-size 32 --max-length 128
```

The published model was trained this way on a Google Colab Tesla T4 GPU (about 15 minutes) with the versions in `requirements.txt`. All random seeds are fixed (seed 42); small differences can still occur between GPU types.

## Limitations

- The synthetic errors come from only five simple rules (en/ett and min/mitt swaps, a missing present-tense -r, broken verb-second order, a missing definite ending, sin/sitt/sina → deras), so other error types are learned only from DaLAJ.
- Punctuation errors were removed from the training data, so the model is not trained to detect them.
- Each text is truncated to 128 tokens, so errors late in long texts may be missed.

## Credits

- **DaLAJ-GED-SuperLim 2.0** by Elena Volodina and Yousuf Ali Mohammed, Språkbanken Text, University of Gothenburg. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). https://doi.org/10.23695/kxvz-tx42
- **ModernBERT-base** by AI Sweden, Apache-2.0 licence.
- **Starter code, evaluation script and example data** by AI Sweden.
