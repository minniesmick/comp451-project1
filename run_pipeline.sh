#!/usr/bin/env bash
# Eğitilmiş / kayıtlı üç modeli Dataset2 (test) üzerinde çalıştırır (eğitim yok).  bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}

for m in model1_char model2_bpe model3_distilbert; do
  $PYTHON pipeline.py \
    --model_load_path "models/$m" \
    --input_file "data/dataset2_test.jsonl" \
    --output_file "outputs/$m.json"
done

$PYTHON compare.py --outputs_dir "outputs" --models_dir "models"
