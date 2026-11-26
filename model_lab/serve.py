"""Model Lab sunucusu (isteğe bağlı): sonuç arayüzünü ve canlı tokenize / tahmin uçlarını yalnızca localhost'ta sunar.

Kullanım (proje kökünden, run.sh bittikten sonra):
  python model_lab/serve.py            ->  http://127.0.0.1:8780
"""
import argparse
import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))        # model_lab/ (ui)
ROOT = os.path.dirname(HERE)                               # project root (models, outputs)
MODELS = [("model1_char", "Model 1 · char"), ("model2_bpe", "Model 2 · BPE"), ("model3_distilbert", "Model 3 · DistilBERT")]
STATIC_OK = re.compile(r"^(ui/[\w./-]+|outputs/[\w-]+\.json|models/[\w-]+/train_log\.json)$")
_cache = {}
_lock = threading.Lock()      # tek seferde yükle: eşzamanlı ilk istekler transformers içe aktarımında yarışır


def _load():
    with _lock:
        return _load_locked()


def _load_locked():
    if _cache:
        return _cache
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    import torch
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)   # mps: yerel ekleme
    for key, title in MODELS:
        path = os.path.join(ROOT, "models", key)
        if not os.path.isdir(path):
            continue
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="eager")   # eager: dikkat ağırlıkları döner
        _cache[key] = {"title": title, "tok": tok, "model": model, "max_len": min(model.config.max_position_embeddings, 512),
                       "pipe": pipeline("text-classification", model=model, tokenizer=tok, device=device)}
    return _cache


def _pieces(tok, ids):
    toks = []
    for t in tok.convert_ids_to_tokens(ids):
        special = t in ("[CLS]", "[SEP]", "[PAD]")
        ws = t.startswith("Ġ")                       # ByteLevel BPE: sözcük başı
        piece = t[1:] if ws else t
        toks.append({"t": piece, "ws": ws, "special": special, "cont": piece.startswith("##")})
    return toks


def tokenize(text):
    out = {}
    for key, m in _load().items():
        ids = m["tok"](text, truncation=True, max_length=m["max_len"])["input_ids"]
        toks = _pieces(m["tok"], ids)
        out[key] = {"title": m["title"], "count": len(toks), "tokens": toks}
    return out


def attention(text):
    """Her model için belirteçler, tahmin olasılıkları ve ham dikkat ağırlıkları [katman][kafa][sorgu][anahtar]."""
    import torch
    out = {}
    for key, m in _load().items():
        tok, model = m["tok"], m["model"]
        enc = tok(text, truncation=True, max_length=m["max_len"], return_tensors="pt")
        if model.config.model_type == "distilbert":
            enc.pop("token_type_ids", None)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            res = model(**enc, output_attentions=True)
        probs = torch.softmax(res.logits[0].float(), -1).cpu().tolist()
        id2 = model.config.id2label
        scores = {id2[i]: round(p, 4) for i, p in enumerate(probs)}
        att = torch.stack([a[0] for a in res.attentions]).float().cpu().round(decimals=3).tolist()   # L x H x n x n
        out[key] = {"title": m["title"], "tokens": _pieces(tok, enc["input_ids"][0].cpu().tolist()), "scores": scores,
                    "label": max(scores, key=scores.get), "layers": len(att), "heads": len(att[0]), "attn": att}
    return out


def predict(text):
    out = {}
    for key, m in _load().items():
        scores = m["pipe"](text, truncation=True, max_length=m["max_len"], top_k=None)
        scores = {s["label"]: round(s["score"], 4) for s in scores}
        out[key] = {"title": m["title"], "scores": scores, "label": max(scores, key=scores.get)}
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].lstrip("/")
        if path in ("", "ui"):                      # göreli css/js yolları için /ui/ altına yönlendir
            self.send_response(302)
            self.send_header("Location", "/ui/")
            self.end_headers()
            return
        if path == "ui/":
            path = "ui/index.html"
        if path == "api/models":
            return self._send(200, json.dumps([{"key": k, "title": t, "ready": os.path.isdir(os.path.join(ROOT, "models", k))} for k, t in MODELS]).encode(), "application/json")
        if not STATIC_OK.match(path) or ".." in path:
            return self._send(404, b"not found", "text/plain")
        full = os.path.join(HERE if path.startswith("ui/") else ROOT, path)
        if not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype.endswith("json") or ctype.endswith("javascript") else ""))

    def do_POST(self):
        if self.path not in ("/api/predict", "/api/tokenize", "/api/attention"):
            return self._send(404, b"not found", "text/plain")
        try:
            text = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))["text"])[:600]
            if not text.strip():
                raise ValueError("empty")
            data = {"/api/predict": predict, "/api/tokenize": tokenize, "/api/attention": attention}[self.path](text)
            self._send(200, json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(), "application/json; charset=utf-8")
        except Exception as e:                                    # noqa: BLE001
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")


def _safe_warm():
    try:
        _load()
    except Exception as e:                       # modeller yoksa arayüz yine açılır
        print(f"warm-up atlandı: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=lambda: _safe_warm(), daemon=True).start()       # modelleri arka planda ısıt
    print(f"Model Lab: http://127.0.0.1:{args.port}  (yalnızca localhost)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
