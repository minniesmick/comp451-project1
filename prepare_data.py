"""Veri hazırlama: TweetEval (sentiment) tüm bölümleri birleştirilir, 250 örnek Dataset2 (test) olarak ayrılır,
geri kalanı Dataset1 (eğitim; train.py bunun %85'ini eğitim, %15'ini doğrulama için kullanır) olur.

Kullanım:
  python prepare_data.py --output_dir data --test_size 250 --seed 42
"""
import argparse
import random
from collections import Counter

from datasets import load_dataset

from common import ID2LABEL, write_json, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="cardiffnlp/tweet_eval")
    ap.add_argument("--dataset_config", default="sentiment")
    ap.add_argument("--output_dir", default="data")
    ap.add_argument("--test_size", type=int, default=250, help="Dataset2 (test) örnek sayısı, 200-250 arası")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, args.dataset_config)
    pool = []
    for split in ds:                      # train + validation + test hepsi tek havuza
        for ex in ds[split]:
            pool.append({"text": ex["text"].strip(), "label": int(ex["label"])})
    # yinelenen metinleri at (eğitim ve test arasında sızıntı olmasın)
    seen, unique = set(), []
    for r in pool:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    removed = len(pool) - len(unique)

    rng = random.Random(args.seed)
    by_label = {}
    for r in unique:
        by_label.setdefault(r["label"], []).append(r)
    total = len(unique)
    test = []
    for label in sorted(by_label):        # test kümesi sınıf oranlarını korur
        items = by_label[label]
        rng.shuffle(items)
        k = round(args.test_size * len(items) / total)
        test += items[:k]
        by_label[label] = items[k:]
    # yuvarlama farkını kapat
    rest = [r for label in sorted(by_label) for r in by_label[label]]
    rng.shuffle(rest)
    while len(test) < args.test_size:
        test.append(rest.pop())
    while len(test) > args.test_size:
        rest.append(test.pop())
    rng.shuffle(rest)
    rng.shuffle(test)

    def finish(rows, prefix):
        return [{"id": f"{prefix}{i:05d}", "text": r["text"], "label": r["label"], "label_name": ID2LABEL[r["label"]]} for i, r in enumerate(rows)]

    dataset1, dataset2 = finish(rest, "d1-"), finish(test, "d2-")
    write_jsonl(f"{args.output_dir}/dataset1.jsonl", dataset1)
    write_jsonl(f"{args.output_dir}/dataset2_test.jsonl", dataset2)
    stats = {
        "source": f"{args.dataset_name}/{args.dataset_config}",
        "seed": args.seed,
        "pooled_examples": len(pool),
        "duplicates_removed": removed,
        "dataset1_train_pool": {"size": len(dataset1), "labels": dict(Counter(r["label_name"] for r in dataset1))},
        "dataset2_test": {"size": len(dataset2), "labels": dict(Counter(r["label_name"] for r in dataset2))},
    }
    write_json(f"{args.output_dir}/data_stats.json", stats)
    print(f"Dataset1: {len(dataset1)} örnek, Dataset2 (test): {len(dataset2)} örnek, kaldırılan yineleme: {removed}")
    print("Dataset1 etiketleri:", stats["dataset1_train_pool"]["labels"])
    print("Dataset2 etiketleri:", stats["dataset2_test"]["labels"])


if __name__ == "__main__":
    main()
