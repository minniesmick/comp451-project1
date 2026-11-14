# COMP 451 Project#1 — English tweet sentiment analysis

> Course project for COMP 451 (Fall 2026) by students of MEF University. It is a student project: it is **not** an official MEF University product, and MEF University is not responsible for it, its content, or any use of it. Released as open source under the [MIT License](LICENSE).

Task: sentiment (negative / neutral / positive) on TweetEval. Three models, three tokenizers:

| Model | Tokenizer | Weights |
|---|---|---|
| Model1 | character-level (learned from train split) | tiny BERT, trained from scratch (MLM pretrain, then classification) |
| Model2 | ByteLevel BPE, vocab 8000 (learned from train split) | same architecture, from scratch |
| Model3 | WordPiece (DistilBERT) | `distilbert-base-uncased-finetuned-sst-2-english`, fine-tuned (head re-initialised to 3 classes) |

Data: `prepare_data.py` pools all TweetEval sentiment splits, removes exact duplicates, holds out 250 stratified samples as Dataset2 (test) and keeps the rest as Dataset1. Training uses 85 % of Dataset1 for training, 15 % for validation (best epoch picked by validation macro-F1). TweetEval was chosen on purpose: the SST-2 checkpoint of Model3 never saw tweets, so there is no train/test leakage.

## Run everything

```bash
pip install -r requirements.txt
bash run.sh                                  # Linux / macOS / Git Bash
PYTHON=.venv/Scripts/python.exe bash run.sh  # Windows with the local venv
```

`run.sh` runs `prepare_data.py` → `train.py` (×2) → `fine_tune.py` → `pipeline.py` (×3) → `compare.py`. All paths are relative.

## Single steps

```bash
python train.py --tokenizer char --model_save_path models/model1_char --dataset_path data/dataset1.jsonl --num_train_epoch 12
python fine_tune.py --model_save_path models/model3_distilbert --dataset_path data/dataset1.jsonl --num_train_epoch 3
python pipeline.py --model_load_path models/model1_char --input_file data/dataset2_test.jsonl --output_file outputs/model1_char.json
```

`pipeline.py` uses `transformers.pipeline("text-classification")`; output json holds per-sample predictions and scores plus accuracy, macro/weighted F1, per-class P/R/F1 and the confusion matrix. `outputs/summary.{json,md}` compare the three models.

## Files

`common.py` shared utils + PyTorch training loop · `custom_tokenizers.py` char/BPE builders · `train.py` Model1/2 · `fine_tune.py` Model3 · `pipeline.py` inference + metrics · `compare.py` summary · `run.sh`.

## Quick checks

```bash
bash run_pipeline.sh                      # uses the saved models in models/ (included in the zip): reproduces outputs/*.json in ~1 min
EPOCHS=1 FT_EPOCHS=1 MLM_EPOCHS=1 MAX_TRAIN=2000 bash run.sh   # whole pipeline end to end in a few minutes (numbers are meaningless, it only proves the flow)
bash run.sh                               # full training (~25 min on a GPU; Model 3 needs internet once to download DistilBERT)
```
Set `PYTHON=python3` (or a venv path) if `python` is not the right interpreter. Training uses the GPU when available, otherwise the CPU.

## Cloning from GitHub

The repository holds code only. Trained weights (`models/`, ~270 MB), the data files and `outputs/*.json` are git-ignored. After cloning, run `bash run.sh`: it downloads TweetEval, builds the splits, trains Model1/Model2, fine-tunes Model3 and writes `outputs/`. The submission zip additionally contains the trained models and data, so `bash run_pipeline.sh` works there without training.

## Model Lab (optional UI)

`python serve.py` starts a local viewer at http://127.0.0.1:8780 (comparison, training curves, tokenizer visualizer, live prediction, confusion matrices). It reads `outputs/` and `models/`, so run the pipeline first. Local only, no external requests.

## License and third-party material

- Code: [MIT](LICENSE). Use, copy and modify freely.
- Data: [TweetEval](https://github.com/cardiffnlp/tweeteval) sentiment (SemEval-2017 Task 4 tweets), downloaded at run time and not redistributed in the repository. Follow the original dataset terms.
- Model3 starts from `distilbert-base-uncased-finetuned-sst-2-english` (Apache-2.0, Hugging Face), downloaded at run time.
- Course project, provided "as is" without warranty. MEF University has no responsibility for this repository.
