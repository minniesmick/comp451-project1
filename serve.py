"""Model Lab sunucusu (isteğe bağlı): sonuç arayüzünü ve canlı tokenize / tahmin uçlarını yalnızca localhost'ta sunar.

Kullanım (proje kökünden, run.sh bittikten sonra):
  python serve.py            ->  http://127.0.0.1:8780
"""
import argparse
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = [("model1_char", "Model 1 · char"), ("model2_bpe", "Model 2 · BPE"), ("model3_distilbert", "Model 3 · DistilBERT")]
STATIC_OK = re.compile(r"^(ui/[\w./-]+|outputs/[\w-]+\.json|models/[\w-]+/train_log\.json)$")
_cache = {}


def _load():
    if _cache:
        return _cache
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    for key, title in MODELS:
        path = os.path.join(ROOT, "models", key)
        if not os.path.isdir(path):
            continue
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        _cache[key] = {"title": title, "tok": tok, "max_len": min(model.config.max_position_embeddings, 512),
                       "pipe": pipeline("text-classification", model=model, tokenizer=tok, device=device)}
    return _cache


def tokenize(text):
    out = {}
    for key, m in _load().items():
        ids = m["tok"](text, truncation=True, max_length=m["max_len"])["input_ids"]
        toks = []
        for t in m["tok"].convert_ids_to_tokens(ids):
            special = t in ("[CLS]", "[SEP]", "[PAD]")
            ws = t.startswith("Ġ")                       # ByteLevel BPE: sözcük başı
            piece = t[1:] if ws else t
            toks.append({"t": piece, "ws": ws, "special": special, "cont": piece.startswith("##")})
        out[key] = {"title": m["title"], "count": len(toks), "tokens": toks}
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
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype.endswith("json") or ctype.endswith("javascript") else ""))

    def do_POST(self):
        if self.path not in ("/api/predict", "/api/tokenize"):
            return self._send(404, b"not found", "text/plain")
        try:
            text = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))["text"])[:600]
            if not text.strip():
                raise ValueError("empty")
            data = predict(text) if self.path == "/api/predict" else tokenize(text)
            self._send(200, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except Exception as e:                                    # noqa: BLE001
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Model Lab: http://127.0.0.1:{args.port}  (yalnızca localhost)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
