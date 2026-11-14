"""Model1 / Model2: küçük bir BERT tipi transformer'ı SIFIRDAN (rastgele başlangıç ağırlıklarıyla) eğitir.

İki aşama (ikincisi isteğe bağlı değil, varsayılan açık):
  1) Dil modeli ön eğitimi (maskeli dil modelleme, MLM): metnin rastgele %15 simgesi maskelenir, model tahmin eder.
  2) Duygu sınıflandırması: aynı gövde üzerinde sınıflandırma başlığı eğitilir.
Tokenizer seçeneği: --tokenizer char (Model1) ya da bpe (Model2). Tokenizer yalnızca eğitim bölümünden öğrenilir.

Kullanım:
  python src/train.py --tokenizer char --model_save_path models/model1_char --dataset_path data/dataset1.jsonl --num_train_epoch 12
"""
import argparse
import json
import os

import torch
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, DataCollatorForLanguageModeling

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # proje kökü: lib/ paketi
from lib.common import ID2LABEL, LABEL2ID, LABELS, get_device, read_jsonl, seed_everything, split_train_val, train_classifier, write_json
from lib.custom_tokenizers import BUILDERS


def pretrain_mlm(tokenizer, texts, config_kwargs, max_len, epochs, batch_size, lr, save_path, seed):
    """Küçük BERT'i yalnızca eğitim metinleriyle MLM görevinde eğitir; ağırlıkları save_path'e yazar."""
    device = get_device()
    seed_everything(seed)
    model = BertForMaskedLM(BertConfig(**config_kwargs)).to(device)
    enc = tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]
    examples = [{"input_ids": ids} for ids in enc]
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collator, generator=g)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / max(1, int(total * 0.06)), max(0.0, (total - s) / total)))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        run, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            run += out.loss.item()
            n += 1
        history.append({"epoch": epoch, "mlm_loss": round(run / n, 4)})
        print(f"MLM epoch {epoch}/{epochs}  kayıp {history[-1]['mlm_loss']}", flush=True)
    model.save_pretrained(save_path)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=sorted(BUILDERS), required=True, help="char: Model1, bpe: Model2")
    ap.add_argument("--model_save_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--num_train_epoch", type=int, default=12, help="sınıflandırma epoch sayısı")
    ap.add_argument("--mlm_epochs", type=int, default=15, help="MLM ön eğitim epoch sayısı (0 = ön eğitim yok)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden_size", type=int, default=192)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=0, help="0: char için 224, bpe için 96")
    ap.add_argument("--vocab_size", type=int, default=8000, help="yalnızca bpe")
    ap.add_argument("--max_train_samples", type=int, default=0, help="0: hepsi; hızlı deneme için küçük bir sayı")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    rows = read_jsonl(args.dataset_path)
    train_rows, val_rows = split_train_val(rows, 0.85, args.seed)
    if args.max_train_samples:
        train_rows, val_rows = train_rows[:args.max_train_samples], val_rows[:max(200, args.max_train_samples // 5)]
    print(f"eğitim {len(train_rows)}, doğrulama {len(val_rows)} (Dataset1 %85 / %15)")

    texts = [r["text"] for r in train_rows]
    tokenizer = BUILDERS[args.tokenizer](texts, args.vocab_size) if args.tokenizer == "bpe" else BUILDERS[args.tokenizer](texts)
    max_len = args.max_len or (224 if args.tokenizer == "char" else 96)
    lens = [len(x) for x in tokenizer(texts[:2000])["input_ids"]]
    print(f"tokenizer: {args.tokenizer}, sözlük {len(tokenizer)}, örnek başına ortalama {sum(lens) / len(lens):.1f} simge, max_len {max_len}")

    config_kwargs = dict(vocab_size=len(tokenizer), hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
                         intermediate_size=args.hidden_size * 4, max_position_embeddings=max_len, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                         pad_token_id=tokenizer.pad_token_id)
    os.makedirs(args.model_save_path, exist_ok=True)

    mlm_history, init_path = [], None
    if args.mlm_epochs > 0:
        init_path = os.path.join(args.model_save_path, "_mlm_tmp")
        mlm_history = pretrain_mlm(tokenizer, texts, config_kwargs, max_len, args.mlm_epochs, args.batch_size, args.lr, init_path, args.seed)

    cls_kwargs = dict(num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    if init_path:
        model = BertForSequenceClassification.from_pretrained(init_path, **cls_kwargs)     # gövde MLM ağırlıklarıyla, başlık rastgele
    else:
        model = BertForSequenceClassification(BertConfig(**config_kwargs, **cls_kwargs))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametre sayısı: {n_params / 1e6:.2f} M")

    result = train_classifier(model, tokenizer, train_rows, val_rows, max_len, args.num_train_epoch, args.batch_size, args.lr / 2, args.model_save_path, seed=args.seed)
    result.update({"tokenizer": args.tokenizer, "parameters": n_params, "mlm_history": mlm_history, "args": vars(args)})
    write_json(os.path.join(args.model_save_path, "train_log.json"), result)
    if init_path:
        import shutil
        shutil.rmtree(init_path, ignore_errors=True)
    print(f"en iyi doğrulama macro-F1: {result['best_val_macro_f1']}  ->  {args.model_save_path}")


if __name__ == "__main__":
    main()
