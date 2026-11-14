"""Model3: hazır, göreve özel bir modeli (DistilBERT SST-2) Dataset1 ile ince ayar yapar.

SST-2 başlığı 2 sınıflıdır (negatif/pozitif); veri 3 sınıflı (negative/neutral/positive) olduğu için
sınıflandırma başlığı 3 çıkışla yeniden başlatılır (ignore_mismatched_sizes). Gövde ve WordPiece tokenizer hazır gelir.

Kullanım:
  python fine_tune.py --model_save_path models/model3_distilbert --dataset_path data/dataset1.jsonl --num_train_epoch 3
"""
import argparse
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ID2LABEL, LABEL2ID, LABELS, read_jsonl, seed_everything, split_train_val, train_classifier, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="distilbert-base-uncased-finetuned-sst-2-english")
    ap.add_argument("--model_save_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--num_train_epoch", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    rows = read_jsonl(args.dataset_path)
    train_rows, val_rows = split_train_val(rows, 0.85, args.seed)
    if args.max_train_samples:
        train_rows, val_rows = train_rows[:args.max_train_samples], val_rows[:max(200, args.max_train_samples // 5)]
    print(f"eğitim {len(train_rows)}, doğrulama {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)
    n_params = sum(p.numel() for p in model.parameters())
    os.makedirs(args.model_save_path, exist_ok=True)
    result = train_classifier(model, tokenizer, train_rows, val_rows, args.max_len, args.num_train_epoch, args.batch_size, args.lr,
                              args.model_save_path, seed=args.seed, patience=0)
    result.update({"tokenizer": "wordpiece (distilbert)", "parameters": n_params, "pretrained": args.pretrained, "args": vars(args)})
    write_json(os.path.join(args.model_save_path, "train_log.json"), result)
    print(f"en iyi doğrulama macro-F1: {result['best_val_macro_f1']}  ->  {args.model_save_path}")


if __name__ == "__main__":
    main()
