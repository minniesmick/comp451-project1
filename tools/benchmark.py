"""Model3 için aday hazır modelleri Dataset2 (test) üzerinde fine-tune ÖNCESİ (zero-shot) karşılaştırır.

Ödev, Model3'ün "hazır, göreve özel bir model"in Dataset1 ile ince ayarıyla üretilmesini ister. Bu betik hangi tabanın
iyi bir başlangıç olduğunu eğitim yapmadan gösterir: her aday, pipeline.py ile aynı 250 tweet ve aynı ölçütlerle denenir.
Adayların çıkış etiketleri (NEG/NEU/POS, 1-5 yıldız, yalnız NEGATIVE/POSITIVE) üç sınıfa olasılık toplanarak eşlenir.

Kullanım:
  python tools/benchmark.py                          # tüm adaylar
  python tools/benchmark.py --only sst2_distilbert twitter_roberta
Çıktı: outputs/bench/<aday>.json (pipeline.py ile aynı şema) ve report/benchmark.md (özet tablo).
"""
import argparse
import os
import sys
import time

from transformers import pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import LABEL2ID, LABELS, compute_metrics, get_device, read_jsonl, write_json

# leak: eğitim verisi Dataset2'yi (TweetEval'in birleşik bölümlerinden ayrıldı) görmüş olabilir mi
CANDIDATES = {
    "sst2_distilbert": dict(id="distilbert/distilbert-base-uncased-finetuned-sst-2-english", lang="EN", leak="yok (film yorumları, SST-2)",
                            labels={"NEGATIVE": "negative", "POSITIVE": "positive"},
                            note="Mevcut Model3 tabanı. İki sınıflı: neutral hiç tahmin edemez, bu yüzden fine-tune şart."),
    "twitter_roberta": dict(id="cardiffnlp/twitter-roberta-base-sentiment-latest", lang="EN", leak="VAR olabilir (TweetEval duygu eğitim verisi)",
                            labels={"negative": "negative", "neutral": "neutral", "positive": "positive"},
                            note="Tweet için en güçlü hazır taban; üç sınıfı zaten biliyor. Sızıntı riski yüzünden sonuçlar şişkin olabilir."),
    "bertweet": dict(id="finiteautomata/bertweet-base-sentiment-analysis", lang="EN", leak="VAR olabilir (SemEval-2017, TweetEval'in kaynağı)",
                     labels={"NEG": "negative", "NEU": "neutral", "POS": "positive"},
                     note="BERTweet: tweet'lerle ön eğitilmiş RoBERTa. Üç sınıflı, SemEval-2017 üzerinde ince ayarlı."),
    "nlptown_multilingual": dict(id="nlptown/bert-base-multilingual-uncased-sentiment", lang="çok dilli (Türkçe dahil)", leak="yok (ürün yorumları)",
                                 labels={"1 star": "negative", "2 stars": "negative", "3 stars": "neutral", "4 stars": "positive", "5 stars": "positive"},
                                 note="Sızıntısız ve çok dilli; 1-5 yıldızı 3 sınıfa topluyoruz. Alan farkı (yorum -> tweet) nedeniyle zero-shot zayıf, fine-tune ile toparlanması beklenir."),
}


def evaluate(key, spec, rows, args):
    t_load = time.time()
    clf = pipeline("text-classification", model=spec["id"], device=get_device())
    max_len = min(getattr(clf.tokenizer, "model_max_length", 512) or 512, 512)
    params = sum(p.numel() for p in clf.model.parameters())
    t0 = time.time()
    outs = clf([r["text"] for r in rows], batch_size=args.batch_size, truncation=True, max_length=min(max_len, 128), top_k=None)
    seconds = time.time() - t0
    predictions, gold, pred = [], [], []
    for r, scores in zip(rows, outs):
        agg = {c: 0.0 for c in LABELS}
        for s in scores:
            agg[spec["labels"][s["label"]]] += s["score"]
        label = max(agg, key=agg.get)
        predictions.append({"id": r["id"], "text": r["text"], "gold": r["label_name"], "predicted": label, "confidence": round(agg[label], 4),
                            "scores": {c: round(v, 4) for c, v in agg.items()}, "correct": label == r["label_name"]})
        gold.append(r["label"])
        pred.append(LABEL2ID[label])
    result = {"model": key, "hf_id": spec["id"], "task": "zero-shot sentiment (negative / neutral / positive)", "input_file": args.input_file, "num_samples": len(rows),
              "parameters": params, "load_seconds": round(t0 - t_load, 2), "inference_seconds": round(seconds, 2), "label_map": spec["labels"],
              "metrics": compute_metrics(gold, pred), "predictions": predictions}
    write_json(os.path.join(args.outputs_dir, f"{key}.json"), result)
    m = result["metrics"]
    print(f"{key}: accuracy {m['accuracy']}  macro-F1 {m['macro_f1']}  ({params / 1e6:.0f} M, {seconds:.1f} s)")
    return result


def write_md(results, rows, path):
    n = len(rows)
    majority = max(LABELS, key=lambda c: sum(r["label_name"] == c for r in rows))
    maj_acc = sum(r["label_name"] == majority for r in rows) / n
    maj_f1 = compute_metrics([r["label"] for r in rows], [LABEL2ID[majority]] * n)["macro_f1"]
    lines = ["# Model3 aday benchmark'ı (fine-tune öncesi, zero-shot)", "",
             f"Aynı {n} tweet (Dataset2), aynı ölçütler (`lib/common.py`). Hiçbir modele eğitim yapılmadı. Taban çizgisi: hep `{majority}` demek doğruluk {maj_acc:.3f}, macro-F1 {maj_f1:.3f}.", "",
             "| Aday | Dil | Parametre | Doğruluk | Macro-F1 | neg F1 | neu F1 | pos F1 | Sızıntı riski |", "|---|---|---|---|---|---|---|---|---|"]
    for key, r in results.items():
        m, pc = r["metrics"], r["metrics"]["per_class"]
        lines.append(f"| [`{r['hf_id']}`](https://huggingface.co/{r['hf_id']}) | {CANDIDATES[key]['lang']} | {r['parameters'] / 1e6:.0f} M | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                     f"{pc['negative']['f1']:.3f} | {pc['neutral']['f1']:.3f} | {pc['positive']['f1']:.3f} | {CANDIDATES[key]['leak']} |")
    lines += ["", "## Adaylar", ""] + [f"- **`{CANDIDATES[k]['id']}`**: {CANDIDATES[k]['note']}" for k in results]
    skipped = [k for k in CANDIDATES if k not in results]
    if skipped:
        lines += ["", "Bu koşuda çalıştırılmadı (indirme tamamlanamadı; `python tools/benchmark.py --only " + " ".join(skipped) + "` ile çalıştırılabilir):", ""]
        lines += [f"- **`{CANDIDATES[k]['id']}`**: {CANDIDATES[k]['note']}" for k in skipped]
    lines += ["", "Sızıntı notu: Dataset2, TweetEval duygu bölümlerinin birleşiminden ayrıldı. Tweet'lerde eğitilmiş adayların eğitim verisi bu tweet'leri içerebilir; "
              "bu satırlar bir üst sınır olarak okunmalı, adil karşılaştırma için sızıntısız adaylarla ayrı tutulmalı.", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", default="data/dataset2_test.jsonl")
    ap.add_argument("--outputs_dir", default="outputs/bench")
    ap.add_argument("--report_file", default="report/benchmark.md")
    ap.add_argument("--only", nargs="*", choices=list(CANDIDATES), help="yalnız bu adaylar")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    rows = read_jsonl(args.input_file)
    results = {}
    for key, spec in CANDIDATES.items():
        if args.only and key not in args.only:
            continue
        try:
            results[key] = evaluate(key, spec, rows, args)
        except Exception as e:                              # noqa: BLE001  tek aday inmezse diğerleri sürsün
            print(f"{key}: ATLANDI ({type(e).__name__}: {e})")
    if results:
        write_md(results, rows, args.report_file)
        print(f"özet -> {args.report_file}")


if __name__ == "__main__":
    main()
