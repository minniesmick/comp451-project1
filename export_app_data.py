"""Uygulama (turkish-wordnet-ui) "Proje 1" sekmesi için küçük bir json üretir: ölçütler, eğitim eğrileri, mimari bilgisi,
dört örnek tweet'in üç tokenizer ile parçalanışı. Kullanım: python export_app_data.py --out ../turkish-wordnet-ui/static/proj1/data.json
"""
import argparse
import json
import os

from transformers import AutoTokenizer

from common import read_jsonl, write_json

KEYS = ["model1_char", "model2_bpe", "model3_distilbert"]


def pieces(tok, text, max_len):
    ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
    out = []
    for t in tok.convert_ids_to_tokens(ids):
        ws = t.startswith("Ġ")
        out.append({"t": t[1:] if ws else t, "ws": ws, "sp": t in ("[CLS]", "[SEP]")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(args.data_dir, "data_stats.json"), encoding="utf-8"))
    models = []
    for k in KEYS:
        out = json.load(open(os.path.join(args.outputs_dir, f"{k}.json"), encoding="utf-8"))
        log = json.load(open(os.path.join(args.models_dir, k, "train_log.json"), encoding="utf-8"))
        cfg = json.load(open(os.path.join(args.models_dir, k, "config.json"), encoding="utf-8"))
        a = log["args"]
        models.append({
            "key": k, "tokenizer": log["tokenizer"], "parameters": log["parameters"], "epochs_run": log["epochs_run"],
            "best_val_macro_f1": log["best_val_macro_f1"], "history": log["history"], "mlm_history": log.get("mlm_history", []),
            "config": {x: cfg.get(x) for x in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "max_position_embeddings")},
            "lr": a.get("lr"), "batch_size": a.get("batch_size"), "mlm_epochs": a.get("mlm_epochs"),
            "metrics": out["metrics"], "inference_seconds": out["inference_seconds"],
        })

    toks = {k: AutoTokenizer.from_pretrained(os.path.join(args.models_dir, k)) for k in KEYS}
    cfgs = {m["key"]: m["config"]["max_position_embeddings"] for m in models}
    rows = [r for r in read_jsonl(os.path.join(args.data_dir, "dataset2_test.jsonl")) if 30 < len(r["text"]) < 95 and "http" not in r["text"]]
    picked, seen = [], set()
    for r in rows:                                    # her sınıftan bir tane, sonra bir tane daha
        if r["label_name"] not in seen:
            seen.add(r["label_name"])
            picked.append(r)
    picked.append(next(r for r in rows if r not in picked))
    examples = [{"text": r["text"], "label": r["label_name"], "tokens": {k: pieces(toks[k], r["text"], cfgs[k]) for k in KEYS}} for r in picked]

    write_json(args.out, {"task": "tweet sentiment (TweetEval)", "data": stats, "models": models, "examples": examples})
    print(f"yazıldı: {args.out}")


if __name__ == "__main__":
    main()
