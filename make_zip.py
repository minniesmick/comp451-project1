"""Teslim zip'i: kod + script + veri + sonuçlar + rapor (model ağırlıkları ve .venv hariç). Kullanım: python make_zip.py"""
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
WITH_MODELS = "--with_models" in sys.argv          # kayıtlı model ağırlıklarını da koy (~270 MB)
SKIP_DIRS = {".venv", "__pycache__", ".git", ".cache", ".claude"} | (set() if WITH_MODELS else {"models"})
SKIP_FILES = {"run.log", "PRODUCT.md"}
OUT = os.path.join(ROOT, "comp451_project1.zip")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f in SKIP_FILES or f.endswith(".zip"):
                continue
            full = os.path.join(base, f)
            z.write(full, os.path.join("comp451_project1", os.path.relpath(full, ROOT)))
print(f"{OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
