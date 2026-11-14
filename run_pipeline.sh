#!/usr/bin/env bash
# Eğitilmiş / kayıtlı üç modeli Dataset2 (test) üzerinde çalıştırır (eğitim yok).  bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
# Python yorumlayıcısı: PYTHON verilmediyse çalışan ilk python3 / python seçilir (macOS'ta yalnızca python3 olur)
if [ -z "${PYTHON:-}" ]; then
  for c in python3 python; do
    if "$c" -c "import sys" >/dev/null 2>&1; then PYTHON=$c; break; fi
  done
fi
: "${PYTHON:?python bulunamadi, PYTHON=/yol/python bash run.sh ile verin}"

for m in model1_char model2_bpe model3_distilbert; do
  $PYTHON src/pipeline.py \
    --model_load_path "models/$m" \
    --input_file "data/dataset2_test.jsonl" \
    --output_file "outputs/$m.json"
done

$PYTHON tools/compare.py --outputs_dir "outputs" --models_dir "models"
