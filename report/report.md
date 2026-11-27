# COMP 451 Project#1 — English Tweet Sentiment Analysis

MEF University · Fall 2026 · Instructor: Dr. Demir · Student: Alper Yusuf Yaman

**Task.** Three-class sentiment (negative / neutral / positive) on TweetEval-sentiment tweets. **Data.** All TweetEval sentiment splits were pooled and exact duplicate texts removed (29). 250 stratified samples form Dataset2 (test, never used for training or model selection); the remaining 59,620 form Dataset1, of which 85 % trains and 15 % validates (best epoch chosen by validation macro-F1). TweetEval was chosen deliberately: Model 3's checkpoint was fine-tuned on SST-2 movie reviews, so tweets avoid train/test leakage.

## Models

| | Tokenizer | Architecture / weights | Params |
|---|---|---|---|
| Model 1 | character level, 462 symbols, learned from train split | 4-layer, 192-d, 4-head BERT, random init; 15 epochs masked-LM pretraining, then 12 epochs classifier | 1.9 M |
| Model 2 | ByteLevel BPE, vocab 8000, learned from train split | same architecture; 15 epochs MLM, then classifier (early-stopped at 6 epochs) | 3.4 M |
| Model 3 | WordPiece (DistilBERT) | `distilbert-base-uncased-finetuned-sst-2-english`, head re-initialised to 3 classes, 3 epochs, lr 2e-5 | 67.0 M |

Models 1 and 2 are trained from scratch in plain PyTorch (AdamW, linear warm-up/decay, bf16 autocast). Average sequence length per tweet: 107 symbols (char) vs 30 (BPE).

## How to run

```
pip install -r requirements.txt
bash run.sh        # data -> src/train.py x2 -> src/fine_tune.py -> src/pipeline.py x3 -> tools/compare.py
```
Everything uses relative paths. Single steps and arguments (`--model_save_path --dataset_path --num_train_epoch --model_load_path --input_file --output_file`) are listed in `README.md`. `src/pipeline.py` uses the Hugging Face `pipeline("text-classification")` and writes predictions plus metrics to `outputs/<model>.json`; `tools/compare.py` builds `outputs/summary.md`. Optional: `python model_lab/serve.py` opens a local results UI (curves, tokenizer view, live prediction).

## Results on Dataset2 (250 tweets)

| Model | Accuracy | Macro-F1 | Weighted-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|
| Model 1 (char, scratch) | 0.636 | 0.618 | 0.638 | 0.520 | 0.655 | 0.678 |
| Model 2 (BPE, scratch) | 0.704 | 0.700 | 0.705 | 0.673 | 0.700 | 0.728 |
| Model 3 (DistilBERT, fine-tuned) | 0.708 | 0.695 | 0.709 | 0.608 | 0.697 | 0.780 |
| Baseline (Word2Vec Skip-Gram) | 0.644 | 0.611 | 0.640 | 0.482 | 0.678 | 0.674 |

## Discussion

- **Tokenizer matters a lot for from-scratch models.** Same architecture and data, BPE beats char by 8 macro-F1 points: 3.6× shorter sequences let attention see whole words, and the 8k vocabulary carries word-level units that a 1.9 M-parameter char model must assemble itself.
- **A tiny BPE model nearly matches a 67 M pretrained model** (macro-F1 0.700 vs 0.695; accuracy 0.704 vs 0.708). With 250 test tweets, a difference of 0.006 is within noise (one tweet is 0.4 accuracy points), so we do not claim a winner between them.
- **Model 3 is better on positive tweets** (F1 0.780 vs 0.728) but worse on negative ones (0.608 vs 0.673): its SST-2 prior is polarity-driven, and the neutral class does not exist in SST-2, so most errors are neutral confused with polarity (see confusion matrices in the output files).
- **Limitations.** One seed, one split, small test set; hyper-parameters were not tuned per model. Neutral is the hardest class for all three.

## Extra: static Word2Vec baseline

`python tools/word2vec_baseline.py` (needs `requirements-extra.txt`; not part of `run.sh`). CBOW and Skip-Gram vectors (100-d, window 5) are learned from the training split only; a tweet is the average of its word vectors, classified by a small network. Skip-Gram was chosen on validation macro-F1 (0.591 vs 0.584); CBOW would have scored 0.619 on the test set. The baseline is level with Model 1 and 9 macro-F1 points below Model 2: averaging static vectors loses word order and context. Its nearest neighbour of *good* is *bad* (cosine 0.67), because opposites occur in the same contexts. In two different sentences *sick* gets one identical Word2Vec vector (cosine 1.0), while all three transformers give it different vectors (cosine -0.18 to 0.66; not comparable across models).
