#!/usr/bin/env bash
# Önce eğitim / ince ayar, sonra pipeline. Tüm yollar göreli; proje kökünden çalıştırın:  bash run.sh
# Windows'ta Python yolu için:  PYTHON=.venv/Scripts/python.exe bash run.sh
# Hızlı deneme (birkaç dakika):  EPOCHS=1 FT_EPOCHS=1 MLM_EPOCHS=1 MAX_TRAIN=2000 bash run.sh
# Eğitimi atlayıp yalnızca kayıtlı modellerle pipeline çalıştırmak için:  bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-12}; FT_EPOCHS=${FT_EPOCHS:-3}; MLM_EPOCHS=${MLM_EPOCHS:-15}; MAX_TRAIN=${MAX_TRAIN:-0}   # 0 = tüm veri

# 0) veri (yoksa indir, Dataset1 + Dataset2 üret)
if [ ! -f data/dataset1.jsonl ]; then
  $PYTHON prepare_data.py --output_dir "data" --test_size 250
fi

# 1) Model1: karakter düzeyi tokenizer, sıfırdan transformer
$PYTHON train.py \
  --tokenizer "char" \
  --model_save_path "models/model1_char" \
  --dataset_path "data/dataset1.jsonl" \
  --num_train_epoch "$EPOCHS" --mlm_epochs "$MLM_EPOCHS" --max_train_samples "$MAX_TRAIN"

# 2) Model2: BPE (alt sözcük) tokenizer, sıfırdan transformer
$PYTHON train.py \
  --tokenizer "bpe" \
  --model_save_path "models/model2_bpe" \
  --dataset_path "data/dataset1.jsonl" \
  --num_train_epoch "$EPOCHS" --mlm_epochs "$MLM_EPOCHS" --max_train_samples "$MAX_TRAIN"

# 3) Model3: hazır DistilBERT SST-2 modeline ince ayar
$PYTHON fine_tune.py \
  --model_save_path "models/model3_distilbert" \
  --dataset_path "data/dataset1.jsonl" \
  --num_train_epoch "$FT_EPOCHS" --max_train_samples "$MAX_TRAIN"

# 4) Pipeline + özet
bash run_pipeline.sh
