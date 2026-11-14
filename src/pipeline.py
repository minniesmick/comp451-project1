"""Hugging Face Pipeline API ile kayıtlı bir modeli Dataset2 (test) üzerinde çalıştırır; tahminleri ve ölçütleri json'a yazar.

Kullanım:
  python src/pipeline.py --model_load_path models/model1_char --input_file data/dataset2_test.jsonl --output_file outputs/model1_char.json
"""
import argparse
import os
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # proje kökü: lib/ paketi
from lib.common import LABEL2ID, LABELS, compute_metrics, get_device, read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_load_path", required=True)
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=0, help="0: model yapılandırmasındaki en uzun konum")
    args = ap.parse_args()

    rows = read_jsonl(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_load_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_load_path)
    max_len = args.max_len or model.config.max_position_embeddings
    device = get_device()
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

    t0 = time.time()
    outs = clf([r["text"] for r in rows], batch_size=args.batch_size, truncation=True, max_length=min(max_len, 512), top_k=None)
    seconds = time.time() - t0

    predictions, gold, pred = [], [], []
    for r, scores in zip(rows, outs):
        best = max(scores, key=lambda s: s["score"])
        label = best["label"]
        if label not in LABEL2ID:                 # LABEL_0 gibi genel adlar gelirse sıraya göre eşle
            label = LABELS[int(label.split("_")[-1])]
        predictions.append({"id": r["id"], "text": r["text"], "gold": r["label_name"], "predicted": label, "confidence": round(best["score"], 4),
                            "scores": {(s["label"] if s["label"] in LABEL2ID else LABELS[int(s["label"].split("_")[-1])]): round(s["score"], 4) for s in scores},
                            "correct": label == r["label_name"]})
        gold.append(r["label"])
        pred.append(LABEL2ID[label])

    result = {
        "model": os.path.basename(os.path.normpath(args.model_load_path)),
        "task": "sentiment-analysis (negative / neutral / positive)",
        "input_file": args.input_file,
        "num_samples": len(rows),
        "inference_seconds": round(seconds, 2),
        "metrics": compute_metrics(gold, pred),
        "predictions": predictions,
    }
    write_json(args.output_file, result)
    m = result["metrics"]
    print(f"{result['model']}: accuracy {m['accuracy']}  macro-F1 {m['macro_f1']}  weighted-F1 {m['weighted_f1']}  ->  {args.output_file}")


if __name__ == "__main__":
    main()
