"""report.md -> report.pdf (markdown kütüphanesi + Edge/Chrome headless). Kullanım: python report/build_pdf.py"""
import os
import re
import subprocess
import markdown

here = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(here, "report.md"), encoding="utf-8").read()
css = open(os.path.join(here, "style.css"), encoding="utf-8").read()
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
html = f'<!doctype html><meta charset="utf-8"><style>{css}</style>{body}'
hp = os.path.join(here, "report.html")
open(hp, "w", encoding="utf-8").write(html)
pdf = os.path.join(here, "report.pdf")
import shutil
cands = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe",
         "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
cands += [shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")]
for b in cands:
    if b and os.path.exists(b):
        subprocess.run([b, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file://" + (hp.replace("\\", "/") if hp[1:2] != ":" else "/" + hp.replace("\\", "/"))], check=True, capture_output=True)
        break
else:
    raise SystemExit("Edge/Chrome bulunamadı")
os.remove(hp)
print(pdf, "sayfa:", len(re.findall(rb"/Type\s*/Page[^s]", open(pdf, "rb").read())))
