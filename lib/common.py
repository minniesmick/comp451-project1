"""Ortak yardımcılar: veri okuma/yazma, tohum, eğitim/doğrulama bölme, ölçütler, PyTorch eğitim döngüsü.

Görev: İngilizce tweet duygu analizi (negative / neutral / positive).
Tüm yollar göreli verilir; script'ler proje kökünden (run.sh ile) çalıştırılır.
"""
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ["negative", "neutral", "positive"]
ID2LABEL = {i: n for i, n in enumerate(LABELS)}
LABEL2ID = {n: i for i, n in enumerate(LABELS)}


# ───────────────────────── dosya ve tohum
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────── bölme: Dataset1'in %85'i eğitim, %15'i doğrulama
def split_train_val(rows, train_ratio=0.85, seed=42):
    """Sınıf oranlarını koruyarak (stratified) eğitim / doğrulama ayrımı."""
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train, val = [], []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        cut = int(round(len(items) * train_ratio))
        train += items[:cut]
        val += items[cut:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ───────────────────────── ölçütler
def compute_metrics(gold, pred):
    """accuracy, macro / weighted F1, sınıf başına precision-recall-F1 ve karışıklık matrisi."""
    labels = list(range(len(LABELS)))
    p, r, f1, sup = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(gold, pred, labels=labels, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(gold, pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels)  # satır: altın, sütun: tahmin
    return {
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "per_class": {
            LABELS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f1[i]), 4), "support": int(sup[i])}
            for i in labels
        },
        "confusion_matrix": {"rows": "gold", "cols": "pred", "labels": LABELS, "matrix": cm.tolist()},
    }


# ───────────────────────── toplu veri hazırlama
def encode_rows(tokenizer, rows, max_len):
    texts = [r["text"] for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    return [{"input_ids": ids, "attention_mask": m, "labels": r["label"]} for ids, m, r in zip(enc["input_ids"], enc["attention_mask"], rows)]


def make_collate(pad_id):
    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([b["labels"] for b in batch])}
    return collate


# ───────────────────────── PyTorch eğitim döngüsü (sınıflandırma)
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    gold, pred, total_loss = [], [], 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item() * len(batch["labels"])
        pred += out.logits.argmax(-1).tolist()
        gold += batch["labels"].tolist()
    m = compute_metrics(gold, pred)
    m["loss"] = round(total_loss / max(len(gold), 1), 4)
    return m


def train_classifier(model, tokenizer, train_rows, val_rows, max_len, epochs, batch_size, lr, save_path, seed=42, warmup_ratio=0.06,
                     weight_decay=0.01, patience=4, log_every=0):
    """Eğitir, her epoch doğrulama macro-F1'ine bakar, en iyi durumu save_path'e kaydeder. Eğitim günlüğünü döndürür."""
    seed_everything(seed)
    device = get_device()
    model.to(device)
    collate = make_collate(tokenizer.pad_token_id)
    train_ds = encode_rows(tokenizer, train_rows, max_len)
    val_ds = encode_rows(tokenizer, val_rows, max_len)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = epochs * len(train_loader)
    warmup = max(1, int(total_steps * warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    use_amp = device.type == "cuda"

    log, best, bad = [], -1.0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += out.loss.item() * len(batch["labels"])
            seen += len(batch["labels"])
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch} adım {step}/{len(train_loader)} kayıp {running / seen:.4f}", flush=True)
        val = evaluate(model, val_loader, device)
        entry = {"epoch": epoch, "train_loss": round(running / seen, 4), "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                 "val_macro_f1": val["macro_f1"], "seconds": round(time.time() - t0, 1)}
        log.append(entry)
        print(f"epoch {epoch}/{epochs}  eğitim kaybı {entry['train_loss']}  doğrulama kaybı {entry['val_loss']}  "
              f"accuracy {entry['val_accuracy']}  macro-F1 {entry['val_macro_f1']}", flush=True)
        if val["macro_f1"] > best:
            best, bad = val["macro_f1"], 0
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            bad += 1
            if patience and bad >= patience:
                print(f"doğrulama macro-F1 {patience} epoch iyileşmedi, eğitim durduruldu.")
                break
    return {"best_val_macro_f1": best, "epochs_run": len(log), "history": log}
