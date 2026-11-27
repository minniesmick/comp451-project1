"""Word2Vec baseline (extra, not part of run.sh): static word vectors next to the three transformers.

Pipeline as in the course slides: standardise, tokenise, index, encode.
  1. CBOW and Skip-Gram vectors are learned from the training split of Dataset1 only.
  2. A tweet is the average of its word vectors; a small PyTorch network classifies it.
  3. CBOW vs Skip-Gram is chosen on the validation split (never on the test set), then measured on Dataset2.
  4. Nearest neighbours and an analogy (cosine similarity), and static vs contextual vectors for one word.

    pip install -r requirements-extra.txt
    python tools/word2vec_baseline.py
"""
import argparse
import html
import os
import re
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from gensim.models import Word2Vec

from lib.common import LABELS, compute_metrics, read_jsonl, seed_everything, split_train_val, write_json

TOKEN = re.compile(r"#?\w+(?:'\w+)?|[!?]|[^\w\s.,;:\"'()\[\]\-/\\*+=<>|~^`%$&@#{}]")
PROBE_WORDS = ["good", "bad", "happy", "love", "worst"]
ANALOGY = ("man", "woman", "king")                      # man : woman :: king : ?
CONTEXT_WORD = "sick"
CONTEXTS = ["i feel sick today and i am staying in bed", "that new trailer is sick, i love it"]


def standardise(text):
    """Undo the dataset's literal escapes, drop links and user handles, lower-case. Stop words stay: 'not' matters."""
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = html.unescape(text).replace("’", "'").lower()
    return re.sub(r"https?://\S+|@\w+", " ", text)


def tokenise(text):
    return TOKEN.findall(standardise(text))


def stable_hash(word):                                   # gensim's default hash() changes between runs
    return zlib.crc32(word.encode("utf-8"))


def sentence_vectors(wv, sentences):
    out = np.zeros((len(sentences), wv.vector_size), dtype=np.float32)
    for i, words in enumerate(sentences):
        known = [w for w in words if w in wv]
        if known:
            out[i] = wv[known].mean(axis=0)
    return torch.from_numpy(out)


def train_head(x_train, y_train, x_val, y_val, seed, epochs=60, patience=8, batch_size=256, lr=1e-3):
    """Input vector -> hidden layer (ReLU) -> softmax over the three classes. Best epoch by validation macro-F1."""
    seed_everything(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, len(LABELS)))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, waited = -1.0, None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(x_train), generator=gen).split(batch_size):
            opt.zero_grad()
            loss_fn(net(x_train[idx]), y_train[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = compute_metrics(y_val.tolist(), net(x_val).argmax(1).tolist())["macro_f1"]
        if score > best:
            best, best_state, waited = score, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def contextual_cosines(models_dir):
    """Cosine between the vectors of the same word in two sentences, for each saved transformer (last hidden layer)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = {}
    for name in ("model1_char", "model2_bpe", "model3_distilbert"):
        path = os.path.join(models_dir, name)
        if not os.path.isdir(path):
            continue
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        vectors = []
        for text in CONTEXTS:
            start = text.index(CONTEXT_WORD)
            end = start + len(CONTEXT_WORD)
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            keep = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
            enc.pop("token_type_ids", None)
            with torch.no_grad():
                hidden = model(**enc, output_hidden_states=True).hidden_states[-1][0]
            vectors.append(hidden[keep].mean(0))
        out[name] = round(float(torch.nn.functional.cosine_similarity(vectors[0], vectors[1], dim=0)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="data/dataset1.jsonl")
    ap.add_argument("--input_file", default="data/dataset2_test.jsonl")
    ap.add_argument("--model_save_path", default="models/word2vec")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--output_file", default="outputs/word2vec.json")
    ap.add_argument("--vector_size", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--num_train_epoch", type=int, default=15, help="Word2Vec epochs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    train_rows, val_rows = split_train_val(read_jsonl(args.dataset_path), seed=args.seed)
    test_rows = read_jsonl(args.input_file)
    train_s, val_s, test_s = ([tokenise(r["text"]) for r in rows] for rows in (train_rows, val_rows, test_rows))
    y_train, y_val = (torch.tensor([r["label"] for r in rows]) for rows in (train_rows, val_rows))
    gold = [r["label"] for r in test_rows]
    print(f"train {len(train_s)}, validation {len(val_s)}, test {len(test_s)}")
    os.makedirs(args.model_save_path, exist_ok=True)

    variants, kept = {}, {}
    for name, sg in (("cbow", 0), ("skipgram", 1)):
        w2v = Word2Vec(sentences=train_s, vector_size=args.vector_size, window=args.window, min_count=args.min_count, sg=sg,
                       epochs=args.num_train_epoch, seed=args.seed, workers=1, hashfxn=stable_hash)
        net, val_f1 = train_head(sentence_vectors(w2v.wv, train_s), y_train, sentence_vectors(w2v.wv, val_s), y_val, args.seed)
        with torch.no_grad():
            probs = net(sentence_vectors(w2v.wv, test_s)).softmax(1)
        pred = probs.argmax(1).tolist()
        total = sum(len(s) for s in test_s)
        variants[name] = {"val_macro_f1": val_f1, "vocabulary": len(w2v.wv),
                          "test_oov_rate": round(sum(w not in w2v.wv for s in test_s for w in s) / max(total, 1), 4),
                          "metrics": compute_metrics(gold, pred)}
        kept[name] = (w2v, net, pred, probs)
        print(f"{name}: vocabulary {len(w2v.wv)}, validation macro-F1 {val_f1:.4f}, test accuracy {variants[name]['metrics']['accuracy']}, "
              f"test macro-F1 {variants[name]['metrics']['macro_f1']}")

    chosen = max(variants, key=lambda n: variants[n]["val_macro_f1"])
    w2v, net, pred, probs = kept[chosen]
    w2v.wv.save_word2vec_format(os.path.join(args.model_save_path, "vectors.txt"))
    torch.save(net.state_dict(), os.path.join(args.model_save_path, "classifier.pt"))

    neighbours = {w: [[n, round(float(s), 3)] for n, s in w2v.wv.most_similar(w, topn=6)] for w in PROBE_WORDS if w in w2v.wv}
    a, b, c = ANALOGY
    analogy = [[n, round(float(s), 3)] for n, s in w2v.wv.most_similar(positive=[b, c], negative=[a], topn=5)] if all(w in w2v.wv for w in ANALOGY) else []
    context = {"word": CONTEXT_WORD, "sentences": CONTEXTS, "word2vec": 1.0 if CONTEXT_WORD in w2v.wv else None,
               "transformers": contextual_cosines(args.models_dir)}

    write_json(args.output_file, {
        "model": "word2vec baseline", "chosen_by_validation": chosen, "settings": {k: v for k, v in vars(args).items()},
        "variants": variants, "nearest_neighbours": neighbours, "analogy": {"query": f"{a} : {b} :: {c} : ?", "answers": analogy},
        "same_word_two_sentences": context,
        "predictions": [{"id": r["id"], "text": r["text"], "gold": LABELS[g], "predicted": LABELS[p], "confidence": round(float(probs[i, p]), 4),
                         "correct": g == p} for i, (r, g, p) in enumerate(zip(test_rows, gold, pred))],
    })
    print(f"chosen by validation: {chosen}  ->  {args.output_file}")
    for w, ns in neighbours.items():
        print(f"  {w}: " + ", ".join(f"{n} {s}" for n, s in ns))
    print(f"  {a}:{b}::{c}:?  " + ", ".join(f"{n} {s}" for n, s in analogy))
    print(f"  '{CONTEXT_WORD}' in two sentences, cosine: word2vec {context['word2vec']}, " + ", ".join(f"{k} {v}" for k, v in context["transformers"].items()))


if __name__ == "__main__":
    main()
