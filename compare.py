"""Üç modelin pipeline çıktılarını (ve eğitim günlüklerini) tek bir özet dosyada toplar: outputs/summary.json ve outputs/summary.md.

Kullanım:
  python compare.py --outputs_dir outputs --models_dir models
"""
import argparse
import os

from common import read_jsonl, write_json

MODELS = [("model1_char", "Model1 (char tokenizer, sıfırdan)"), ("model2_bpe", "Model2 (BPE tokenizer, sıfırdan)"), ("model3_distilbert", "Model3 (DistilBERT, ince ayar)")]


def load(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()

    summary = {"models": []}
    lines = ["| Model | Parametre | Doğruluk | Macro-F1 | Weighted-F1 | neg F1 | neu F1 | pos F1 |", "|---|---|---|---|---|---|---|---|"]
    for key, title in MODELS:
        out_path = os.path.join(args.outputs_dir, f"{key}.json")
        if not os.path.exists(out_path):
            continue
        out = load(out_path)
        log_path = os.path.join(args.models_dir, key, "train_log.json")
        log = load(log_path) if os.path.exists(log_path) else {}
        m = out["metrics"]
        entry = {"key": key, "title": title, "parameters": log.get("parameters"), "tokenizer": log.get("tokenizer"), "epochs_run": log.get("epochs_run"),
                 "best_val_macro_f1": log.get("best_val_macro_f1"), "test": {k: m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "per_class")},
                 "confusion_matrix": m["confusion_matrix"], "inference_seconds": out["inference_seconds"]}
        summary["models"].append(entry)
        pc = m["per_class"]
        params = f"{log['parameters'] / 1e6:.1f} M" if log.get("parameters") else "-"
        lines.append(f"| {title} | {params} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} | {pc['negative']['f1']} | {pc['neutral']['f1']} | {pc['positive']['f1']} |")
    write_json(os.path.join(args.outputs_dir, "summary.json"), summary)
    with open(os.path.join(args.outputs_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
