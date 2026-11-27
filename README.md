<a id="top"></a>

<div align="right">

[Türkçe](./README_Tr.md)

</div>

<div align="center">

# COMP 451 Project#1

**English tweet sentiment analysis with three transformer models and three tokenizers**

![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c?style=flat-square)
![Transformers](https://img.shields.io/badge/Transformers-5.23.0-ffbf00?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-555?style=flat-square)
[![License](https://img.shields.io/badge/license-MIT-e0aaff?style=flat-square)](./LICENSE)
[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/minniesmick/comp451-project1?utm_source=readme&utm_medium=badge)

<br/>

**COMP 451 · Fall 2026 · MEF University** &nbsp;|&nbsp; Instructor: Dr. Demir &nbsp;|&nbsp; Student: Alper Yusuf Yaman

> Course project for COMP 451 (Fall 2026) by students of MEF University. It is a student project: it is **not** an official MEF University product, and MEF University is not responsible for it, its content, or any use of it. Released as open source under the [MIT License](./LICENSE).

</div>

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Features](#features)
- [Screenshots](#screenshots)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Development](#development)
- [Build & Package](#build--package)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Author](#author)
- [Sources and credits](#sources-and-credits)
- [License](#license)

## Overview

The task is three-class sentiment classification (negative / neutral / positive) on TweetEval. The project compares two small BERT-style transformers trained from scratch, each with its own tokenizer, against a fine-tuned pretrained DistilBERT. All models are evaluated with the Hugging Face `pipeline` API on a held-out set of 250 tweets.

`tools/prepare_data.py` pools all TweetEval sentiment splits, removes exact duplicates, holds out 250 stratified samples as Dataset2 (test) and keeps the rest as Dataset1 (59,620 tweets). Training uses 85 % of Dataset1 for training and 15 % for validation; the best epoch is picked by validation macro-F1. TweetEval was chosen on purpose: the SST-2 checkpoint behind Model3 never saw tweets, so there is no train/test leakage.

| Model | Tokenizer | Weights | Parameters |
|---|---|---|---|
| Model1 | character-level, learned from the train split (462 symbols) | tiny BERT from scratch (MLM pretrain, then classification) | 1.9 M |
| Model2 | ByteLevel BPE, vocab 8000, learned from the train split | same architecture, from scratch | 3.4 M |
| Model3 | WordPiece (DistilBERT) | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), fine-tuned, head re-initialised to 3 classes | 67.0 M |

The from-scratch models have 4 layers, hidden size 192, 4 attention heads and FFN size 768.

Links: Model3 base checkpoint [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (Hugging Face Hub, downloaded on first run); Model1 and Model2 use the [BERT](https://arxiv.org/abs/1810.04805) architecture via [`BertForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification) and are trained here, so there is no hosted checkpoint; data from [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

## Results

Dataset2 (250 held-out tweets):

| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| Model1 (char, scratch) | 0.636 | 0.618 | 0.520 | 0.655 | 0.678 |
| Model2 (BPE, scratch) | 0.704 | 0.700 | 0.673 | 0.700 | 0.728 |
| Model3 (DistilBERT, fine-tuned) | 0.708 | 0.695 | 0.608 | 0.697 | 0.780 |

BPE beats character-level tokens by about 8 macro-F1 points. Model2 and Model3 are within noise of each other on 250 samples. Details are in [report/report.md](./report/report.md).

## Extra: Word2Vec baseline

Not part of the assignment flow (`run.sh` does not call it). It learns CBOW and Skip-Gram vectors from the training split, classifies the average word vector, picks the variant on validation and measures it on the same 250 tweets. It also prints nearest neighbours and an analogy, and compares one word in two sentences (static vs contextual vectors).

```bash
pip install -r requirements-extra.txt
python tools/word2vec_baseline.py      # -> outputs/word2vec.json, about 3 minutes on CPU
```

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Word2Vec Skip-Gram baseline | 0.644 | 0.611 |

## Features

### Training
- Custom PyTorch loop: AdamW, warmup and decay, bf16 autocast on CUDA, early stopping on validation macro-F1.
- MLM pretraining (15 % masking) before the classification head for Model1 and Model2.
- Tokenizers built from the training split only (`lib/custom_tokenizers.py`).

### Evaluation
- `src/pipeline.py` uses `transformers.pipeline("text-classification")` with `top_k=None`.
- JSON output holds per-sample predictions and scores, accuracy, macro and weighted F1, per-class precision/recall/F1 and the confusion matrix (rows = gold labels).
- `tools/compare.py` writes `outputs/summary.json` and `outputs/summary.md`.

### Model Lab (local viewer)
- Compare tab: animated ranking of the three models.
- Training tab: loss and F1 curves per epoch.
- Tokenizers tab: see how each tokenizer splits a tweet.
- Attention tab: the real attention weights of all three models for one tweet. Attention rollout shows which tokens feed the [CLS] vector the classifier reads; below it you pick a layer, a head and a token and see its attention row and the full matrix.
- Try it tab: type a tweet or roll a random test tweet and see all three predictions.
- Test set tab: three confusion matrices side by side and a per-tweet list with model votes.

## Screenshots

> Screenshots coming soon. Run `python model_lab/serve.py` to see the Model Lab locally.

## Tech Stack

| Layer | Technology |
|---|---|
| Models | PyTorch 2.6.0, Hugging Face Transformers 5.23.0 |
| Tokenizers | Hugging Face `tokenizers` 0.23.3 (WordLevel, ByteLevel BPE, WordPiece) |
| Data | `datasets` 5.2.0, TweetEval sentiment |
| Metrics | scikit-learn 1.9.0, NumPy 2.4.4 |
| Model Lab | Python standard-library HTTP server + vanilla ES-module frontend (no build step) |

Tested on Python 3.11, Windows 11.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Other 3.x versions may work, pinned libraries were tested on 3.11. |
| bash | any | Linux, macOS, or Git Bash on Windows, for the `.sh` scripts. |
| GPU | optional | CUDA is used when available. CPU training works but takes hours. |
| Internet | first run | Downloads TweetEval and DistilBERT once. |

## Quick Start

1. Clone the repository.

   ```bash
   git clone https://github.com/minniesmick/comp451-project1.git
   cd comp451-project1
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Run the full pipeline (data, training, fine-tuning, evaluation, summary).

   ```bash
   bash run.sh                                  # Linux / macOS / Git Bash
   PYTHON=.venv/Scripts/python.exe bash run.sh  # Windows with a local venv
   ```

   About 25 minutes on a GPU. A short smoke run proves the flow end to end in a few minutes:

   ```bash
   EPOCHS=1 FT_EPOCHS=1 MLM_EPOCHS=1 MAX_TRAIN=2000 bash run.sh
   ```

4. Evaluate saved models without training (needs `models/`, which is not in the git repository).

   ```bash
   bash run_pipeline.sh
   ```

5. Open the Model Lab (optional).

   ```bash
   python model_lab/serve.py
   ```

The repository holds code only. Trained weights (`models/`, about 270 MB), the data files and `outputs/*.json` are git-ignored. The submission zip additionally contains models and data, so `run_pipeline.sh` works there without training.

## Configuration

There is no config file. Every script takes command-line arguments, and `run.sh` takes environment variables.

`run.sh` environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | auto (`python3`, else `python`) | Interpreter to use, e.g. `PYTHON=.venv/bin/python` (macOS / Linux) or `PYTHON=.venv/Scripts/python.exe` (Windows). |
| `EPOCHS` | `12` | Classification epochs for Model1 and Model2. |
| `MLM_EPOCHS` | `15` | MLM pretraining epochs for Model1 and Model2. |
| `FT_EPOCHS` | `3` | Fine-tuning epochs for Model3. |
| `MAX_TRAIN` | `0` | Limit on training samples, `0` means all. |

Script arguments:

```bash
python src/train.py --tokenizer char --model_save_path models/model1_char \
  --dataset_path data/dataset1.jsonl --num_train_epoch 12
python src/fine_tune.py --model_save_path models/model3_distilbert \
  --dataset_path data/dataset1.jsonl --num_train_epoch 3
python src/pipeline.py --model_load_path models/model1_char \
  --input_file data/dataset2_test.jsonl --output_file outputs/model1_char.json
```

Extra `src/train.py` options: `--mlm_epochs 15 --batch_size 64 --lr 5e-4 --hidden_size 192 --num_layers 4 --num_heads 4 --max_len --vocab_size --max_train_samples --seed`. All paths are relative.

## Development

| Port | Service |
|---|---|
| 8780 | Model Lab (`model_lab/serve.py`), local only |

- The Model Lab reads `outputs/*.json` and `models/*/train_log.json`, so run the pipeline first.
- API routes: `GET /api/models`, `POST /api/predict`, `POST /api/tokenize`, `POST /api/attention`.
- The frontend is plain ES modules in `model_lab/ui/`. Edit and reload, no build step.
- Line endings: `.sh` files are forced to LF through `.gitattributes`, so they work on Linux and macOS.

## Build & Package

Build the one-page report PDF and the submission zip:

```bash
python report/build_pdf.py          # report.md -> report.pdf (needs Edge or Chrome)
python tools/make_zip.py                  # code, data, outputs, report; about 3.5 MB
python tools/make_zip.py --with_models    # also include trained weights; about 270 MB
```

## Project Structure

```text
.
├── run.sh                # train, fine-tune, evaluate
├── run_pipeline.sh       # evaluate saved models only
├── requirements.txt      # pinned dependencies
├── src/                  # the three assignment scripts
│   ├── train.py              # Model1 / Model2 (from scratch)
│   ├── fine_tune.py          # Model3 (DistilBERT fine-tuning)
│   └── pipeline.py           # HF pipeline inference + metrics
├── lib/                  # shared code
│   ├── common.py             # utils + PyTorch training loop
│   └── custom_tokenizers.py  # char and BPE tokenizer builders
├── tools/                # helper scripts
│   ├── prepare_data.py       # download TweetEval, build Dataset1 / Dataset2
│   ├── word2vec_baseline.py  # static-vector baseline (extra)
│   ├── compare.py            # summary of the three models
│   ├── export_app_data.py    # export results for the course study app
│   └── make_zip.py           # build the submission zip
├── model_lab/            # local results viewer
│   ├── serve.py              # stdlib HTTP server
│   ├── ui/                   # index.html, style.css, app.js, fonts, logo
│   └── PRODUCT.md            # design notes (git-ignored)
├── report/               # one-page report (md, css, PDF builder, PDF)
├── data/                 # data_stats.json in git; jsonl files generated
├── models/               # trained weights, git-ignored
└── outputs/              # predictions and metrics, git-ignored
```

## Roadmap

- [ ] Demo video link in the report
- [ ] Multiple seeds and confidence intervals for the 250-tweet test set
- [ ] Larger scratch model or longer MLM pretraining
- [ ] Class-weighted loss for the negative class
- [ ] Publish trained weights as a GitHub Release
- [ ] Error analysis by tweet length and topic
- [ ] Screenshots of the Model Lab

## Contributing

1. Fork the repository.
2. Create a branch: `git checkout -b my-change`.
3. Commit your changes.
4. Push the branch and open a pull request.

Keep paths relative and keep `run.sh` working end to end. Check it with the smoke run from Quick Start.

## Author

[@minniesmick](https://github.com/minniesmick) on GitHub

## Sources and credits

This project stands on open work. Thank you to the authors and maintainers.

**Data**
- [TweetEval](https://github.com/cardiffnlp/tweeteval) by Cardiff NLP, sentiment task. Loaded from the Hugging Face Hub as [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (config `sentiment`). Tweets originate from SemEval-2017 Task 4. The dataset card lists the license as unknown, so check the original terms of use; the data is downloaded at run time and not redistributed in this repository.
- Barbieri et al., 2020. [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/). Findings of EMNLP.
- Rosenthal et al., 2017. [SemEval-2017 Task 4: Sentiment Analysis in Twitter](https://aclanthology.org/S17-2088/).

**Models**
- Model3 base: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (Apache-2.0), a DistilBERT checkpoint fine-tuned on SST-2.
- Sanh et al., 2019. [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108).
- Devlin et al., 2018. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805). Architecture and MLM objective of Model1 and Model2.
- Socher et al., 2013. [Stanford Sentiment Treebank (SST-2)](https://aclanthology.org/D13-1170/), the data behind the Model3 base checkpoint.

**Libraries**
[PyTorch](https://pytorch.org/) (BSD-3), [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [Datasets](https://github.com/huggingface/datasets) (Apache-2.0), [scikit-learn](https://scikit-learn.org/) (BSD-3), [NumPy](https://numpy.org/) (BSD-3).

## License

Licensed under the MIT License, see [LICENSE](./LICENSE).

- Data: [TweetEval](https://github.com/cardiffnlp/tweeteval) sentiment (SemEval-2017 Task 4 tweets), downloaded at run time and not redistributed here. Follow the original dataset terms.
- Model3 starts from `distilbert-base-uncased-finetuned-sst-2-english` (Apache-2.0, Hugging Face), downloaded at run time.
- Model Lab fonts: [Geist](https://github.com/vercel/geist-font) and Geist Mono by Vercel, SIL Open Font License 1.1, self-hosted in `model_lab/ui/fonts/`.
- The MEF University name and logo (shown in the Model Lab) are trademarks of MEF University. They are not covered by the MIT license and are used here only to identify the course context.
- Provided "as is" without warranty. MEF University has no responsibility for this repository.

<div align="center">
  <br/>
  <sub>Small models, honest numbers.</sub>
  <br/><br/>
  <a href="#top">↑ back to top</a>
</div>
