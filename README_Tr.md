<a id="top"></a>

<div align="right">

[English](./README.md)

</div>

<div align="center">

# COMP 451 Project#1

**Üç transformer modeli ve üç tokenizer ile İngilizce tweet duygu analizi**

![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c?style=flat-square)
![Transformers](https://img.shields.io/badge/Transformers-5.23.0-ffbf00?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-555?style=flat-square)
[![License](https://img.shields.io/badge/license-MIT-e0aaff?style=flat-square)](./LICENSE)

<br/>

**COMP 451 · Güz 2026 · MEF Üniversitesi** &nbsp;|&nbsp; Öğretim üyesi: Dr. Demir &nbsp;|&nbsp; Öğrenci: Alper Yusuf Yaman

> MEF Üniversitesi öğrencilerinin COMP 451 (Güz 2026) ders projesidir. Öğrenci projesidir: resmi bir MEF Üniversitesi ürünü **değildir**; MEF Üniversitesi projeden, içeriğinden veya kullanımından sorumlu değildir. [MIT Lisansı](./LICENSE) ile açık kaynaktır.

</div>

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Sonuçlar](#sonuçlar)
- [Özellikler](#özellikler)
- [Ekran Görüntüleri](#ekran-görüntüleri)
- [Teknoloji](#teknoloji)
- [Gereksinimler](#gereksinimler)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Yapılandırma](#yapılandırma)
- [Geliştirme](#geliştirme)
- [Paketleme](#paketleme)
- [Proje Yapısı](#proje-yapısı)
- [Yol Haritası](#yol-haritası)
- [Katkı](#katkı)
- [Yazar](#yazar)
- [Kaynaklar ve teşekkür](#kaynaklar-ve-teşekkür)
- [Lisans](#lisans)

## Genel Bakış

Görev, TweetEval üzerinde üç sınıflı duygu sınıflandırmadır (negatif / nötr / pozitif). Proje, her biri kendi tokenizer'ına sahip, sıfırdan eğitilmiş iki küçük BERT tarzı transformer'ı, ince ayar yapılmış hazır bir DistilBERT ile karşılaştırır. Tüm modeller, ayrılmış 250 tweetlik bir kümede Hugging Face `pipeline` API'siyle değerlendirilir.

`tools/prepare_data.py` tüm TweetEval duygu bölümlerini birleştirir, birebir kopyaları atar, 250 katmanlı örneği Dataset2 (test) olarak ayırır, kalanı Dataset1 (59.620 tweet) yapar. Eğitimde Dataset1'in %85'i eğitim, %15'i doğrulama olur; en iyi epoch doğrulama macro-F1'e göre seçilir. TweetEval bilerek seçildi: Model3'ün dayandığı SST-2 modeli hiç tweet görmedi, yani eğitim/test sızıntısı yok.

| Model | Tokenizer | Ağırlıklar | Parametre |
|---|---|---|---|
| Model1 | karakter düzeyi, eğitim bölümünden öğrenilir (462 sembol) | sıfırdan küçük BERT (MLM ön eğitim, sonra sınıflandırma) | 1,9 M |
| Model2 | ByteLevel BPE, sözlük 8000, eğitim bölümünden öğrenilir | aynı mimari, sıfırdan | 3,4 M |
| Model3 | WordPiece (DistilBERT) | `distilbert-base-uncased-finetuned-sst-2-english`, ince ayar, çıkış katmanı 3 sınıfa yeniden kuruldu | 67,0 M |

Sıfırdan modeller 4 katman, gizli boyut 192, 4 dikkat başı ve FFN boyutu 768 kullanır.

## Sonuçlar

Dataset2 (ayrılmış 250 tweet):

| Model | Doğruluk | Macro-F1 | F1 negatif | F1 nötr | F1 pozitif |
|---|---|---|---|---|---|
| Model1 (karakter, sıfırdan) | 0,636 | 0,618 | 0,520 | 0,655 | 0,678 |
| Model2 (BPE, sıfırdan) | 0,704 | 0,700 | 0,673 | 0,700 | 0,728 |
| Model3 (DistilBERT, ince ayar) | 0,708 | 0,695 | 0,608 | 0,697 | 0,780 |

BPE, karakter düzeyine göre yaklaşık 8 macro-F1 puan öndedir. 250 örnekte Model2 ile Model3 arasındaki fark gürültü sınırındadır. Ayrıntılar [report/report.md](./report/report.md) içinde.

## Özellikler

### Eğitim
- Özel PyTorch döngüsü: AdamW, ısınma ve azalma, CUDA'da bf16 autocast, doğrulama macro-F1'e göre erken durdurma.
- Model1 ve Model2 için sınıflandırma öncesi MLM ön eğitimi (%15 maskeleme).
- Tokenizer'lar yalnızca eğitim bölümünden kurulur (`lib/custom_tokenizers.py`).

### Değerlendirme
- `src/pipeline.py`, `top_k=None` ile `transformers.pipeline("text-classification")` kullanır.
- JSON çıktısı örnek başına tahmin ve skorları, doğruluk, macro ve ağırlıklı F1'i, sınıf başına precision/recall/F1'i ve karışıklık matrisini (satırlar = gerçek etiket) içerir.
- `tools/compare.py`, `outputs/summary.json` ve `outputs/summary.md` yazar.

### Model Lab (yerel görüntüleyici)
- Compare sekmesi: üç modelin animasyonlu sıralaması.
- Training sekmesi: epoch başına kayıp ve F1 eğrileri.
- Tokenizers sekmesi: her tokenizer'ın bir tweeti nasıl böldüğü.
- Try it sekmesi: tweet yaz ya da rastgele test tweeti at, üç tahmini gör.
- Test set sekmesi: yan yana üç karışıklık matrisi ve model oylarıyla tweet listesi.

## Ekran Görüntüleri

> Ekran görüntüleri yakında. Model Lab'ı görmek için `python model_lab/serve.py` çalıştırın.

## Teknoloji

| Katman | Teknoloji |
|---|---|
| Modeller | PyTorch 2.6.0, Hugging Face Transformers 5.23.0 |
| Tokenizer | Hugging Face `tokenizers` 0.23.3 (WordLevel, ByteLevel BPE, WordPiece) |
| Veri | `datasets` 5.2.0, TweetEval duygu |
| Metrik | scikit-learn 1.9.0, NumPy 2.4.4 |
| Model Lab | Python standart kütüphane HTTP sunucusu + sade ES modül ön yüzü (derleme adımı yok) |

Python 3.11 ve Windows 11'de test edildi.

## Gereksinimler

| Gereksinim | Sürüm | Not |
|---|---|---|
| Python | 3.11 | Sabitlenen kütüphaneler 3.11'de denendi, diğer 3.x sürümleri çalışabilir. |
| bash | herhangi | `.sh` betikleri için Linux, macOS veya Windows'ta Git Bash. |
| GPU | isteğe bağlı | Varsa CUDA kullanılır. CPU'da eğitim saatler sürer. |
| İnternet | ilk çalıştırma | TweetEval ve DistilBERT bir kez indirilir. |

## Hızlı Başlangıç

1. Depoyu klonlayın.

   ```bash
   git clone https://github.com/minniesmick/comp451-project1.git
   cd comp451-project1
   ```

2. Bağımlılıkları kurun.

   ```bash
   pip install -r requirements.txt
   ```

3. Tüm akışı çalıştırın (veri, eğitim, ince ayar, değerlendirme, özet).

   ```bash
   bash run.sh                                  # Linux / macOS / Git Bash
   PYTHON=.venv/Scripts/python.exe bash run.sh  # Windows, yerel venv ile
   ```

   GPU'da yaklaşık 25 dakika. Kısa bir duman testi akışı birkaç dakikada uçtan uca doğrular:

   ```bash
   EPOCHS=1 FT_EPOCHS=1 MLM_EPOCHS=1 MAX_TRAIN=2000 bash run.sh
   ```

4. Kayıtlı modelleri eğitmeden değerlendirin (git deposunda olmayan `models/` gerekir).

   ```bash
   bash run_pipeline.sh
   ```

5. Model Lab'ı açın (isteğe bağlı).

   ```bash
   python model_lab/serve.py
   ```

Depo yalnızca kod içerir. Eğitilmiş ağırlıklar (`models/`, yaklaşık 270 MB), veri dosyaları ve `outputs/*.json` git'te yoktur. Teslim zip'inde modeller ve veri de var, bu yüzden orada `run_pipeline.sh` eğitimsiz çalışır.

## Yapılandırma

Yapılandırma dosyası yok. Her betik komut satırı argümanı alır, `run.sh` ortam değişkeni alır.

`run.sh` ortam değişkenleri:

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `PYTHON` | otomatik (`python3`, yoksa `python`) | Kullanılacak yorumlayıcı, örn. `PYTHON=.venv/bin/python` (macOS / Linux) veya `PYTHON=.venv/Scripts/python.exe` (Windows). |
| `EPOCHS` | `12` | Model1 ve Model2 sınıflandırma epoch sayısı. |
| `MLM_EPOCHS` | `15` | Model1 ve Model2 MLM ön eğitim epoch sayısı. |
| `FT_EPOCHS` | `3` | Model3 ince ayar epoch sayısı. |
| `MAX_TRAIN` | `0` | Eğitim örneği sınırı, `0` hepsi demek. |

Betik argümanları:

```bash
python src/train.py --tokenizer char --model_save_path models/model1_char \
  --dataset_path data/dataset1.jsonl --num_train_epoch 12
python src/fine_tune.py --model_save_path models/model3_distilbert \
  --dataset_path data/dataset1.jsonl --num_train_epoch 3
python src/pipeline.py --model_load_path models/model1_char \
  --input_file data/dataset2_test.jsonl --output_file outputs/model1_char.json
```

`src/train.py` ek seçenekleri: `--mlm_epochs 15 --batch_size 64 --lr 5e-4 --hidden_size 192 --num_layers 4 --num_heads 4 --max_len --vocab_size --max_train_samples --seed`. Tüm yollar göreli.

## Geliştirme

| Port | Servis |
|---|---|
| 8780 | Model Lab (`model_lab/serve.py`), yalnızca yerel |

- Model Lab `outputs/*.json` ve `models/*/train_log.json` okur, önce akışı çalıştırın.
- API rotaları: `GET /api/models`, `POST /api/predict`, `POST /api/tokenize`.
- Ön yüz `model_lab/ui/` içinde sade ES modülleridir. Düzenleyip yenileyin, derleme yok.
- Satır sonu: `.sh` dosyaları `.gitattributes` ile LF'ye zorlanır, Linux ve macOS'ta çalışır.

## Paketleme

Tek sayfalık rapor PDF'i ve teslim zip'i:

```bash
python report/build_pdf.py          # report.md -> report.pdf (Edge veya Chrome gerekir)
python tools/make_zip.py                  # kod, veri, çıktılar, rapor; yaklaşık 3,5 MB
python tools/make_zip.py --with_models    # eğitilmiş ağırlıklar da dahil; yaklaşık 270 MB
```

## Proje Yapısı

```text
.
├── run.sh                # eğit, ince ayar yap, değerlendir
├── run_pipeline.sh       # yalnızca kayıtlı modelleri değerlendir
├── requirements.txt      # sabitlenmiş bağımlılıklar
├── src/                  # ödevin üç betiği
│   ├── train.py              # Model1 / Model2 (sıfırdan)
│   ├── fine_tune.py          # Model3 (DistilBERT ince ayar)
│   └── pipeline.py           # HF pipeline çıkarımı + metrikler
├── lib/                  # ortak kod
│   ├── common.py             # araçlar + PyTorch eğitim döngüsü
│   └── custom_tokenizers.py  # karakter ve BPE tokenizer kurucuları
├── tools/                # yardımcı betikler
│   ├── prepare_data.py       # TweetEval indir, Dataset1 / Dataset2 üret
│   ├── compare.py            # üç modelin özeti
│   ├── export_app_data.py    # sonuçları ders çalışma uygulaması için dışa aktar
│   └── make_zip.py           # teslim zip'ini üret
├── model_lab/            # yerel sonuç görüntüleyici
│   ├── serve.py              # standart kütüphane HTTP sunucusu
│   ├── ui/                   # index.html, style.css, app.js, yazı tipleri, logo
│   └── PRODUCT.md            # tasarım notları (git'te yok)
├── report/               # tek sayfalık rapor (md, css, PDF üretici, PDF)
├── data/                 # git'te data_stats.json; jsonl dosyaları üretilir
├── models/               # eğitilmiş ağırlıklar, git'te yok
└── outputs/              # tahmin ve metrikler, git'te yok
```

## Yol Haritası

- [ ] Rapora demo video bağlantısı
- [ ] 250 tweetlik test kümesi için çoklu seed ve güven aralığı
- [ ] Daha büyük sıfırdan model veya daha uzun MLM ön eğitimi
- [ ] Negatif sınıf için sınıf ağırlıklı kayıp
- [ ] Eğitilmiş ağırlıkları GitHub Release olarak yayınlama
- [ ] Tweet uzunluğu ve konuya göre hata analizi
- [ ] Model Lab ekran görüntüleri

## Katkı

1. Depoyu fork edin.
2. Dal açın: `git checkout -b my-change`.
3. Değişikliklerinizi commit edin.
4. Dalı push edin ve pull request açın.

Yolları göreli tutun, `run.sh` uçtan uca çalışmaya devam etsin. Hızlı Başlangıç'taki duman testiyle kontrol edin.

## Yazar

GitHub: [@minniesmick](https://github.com/minniesmick)

## Kaynaklar ve teşekkür

Bu proje açık çalışmaların üzerinde duruyor. Yazarlara ve bakımcılara teşekkürler.

**Veri**
- Cardiff NLP'nin [TweetEval](https://github.com/cardiffnlp/tweeteval) derlemesi, duygu görevi. Hugging Face Hub'dan [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (config `sentiment`) olarak yüklenir. Tweetler SemEval-2017 Task 4 kaynaklıdır. Veri kartı lisansı "unknown" olarak listeliyor, bu yüzden özgün kullanım koşullarına bakın; veri çalışma anında indirilir, bu depoda yeniden dağıtılmaz.
- Barbieri vd., 2020. [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/). Findings of EMNLP.
- Rosenthal vd., 2017. [SemEval-2017 Task 4: Sentiment Analysis in Twitter](https://aclanthology.org/S17-2088/).

**Modeller**
- Model3 tabanı: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (Apache-2.0), SST-2 üzerinde ince ayarlı DistilBERT kontrol noktası.
- Sanh vd., 2019. [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108).
- Devlin vd., 2018. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805). Model1 ve Model2'nin mimarisi ve MLM hedefi.
- Socher vd., 2013. [Stanford Sentiment Treebank (SST-2)](https://aclanthology.org/D13-1170/), Model3 taban kontrol noktasının verisi.

**Kütüphaneler**
[PyTorch](https://pytorch.org/) (BSD-3), [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [Datasets](https://github.com/huggingface/datasets) (Apache-2.0), [scikit-learn](https://scikit-learn.org/) (BSD-3), [NumPy](https://numpy.org/) (BSD-3).

## Lisans

MIT Lisansı ile lisanslıdır, bkz. [LICENSE](./LICENSE).

- Veri: [TweetEval](https://github.com/cardiffnlp/tweeteval) duygu (SemEval-2017 Task 4 tweetleri), çalışma anında indirilir, burada yeniden dağıtılmaz. Özgün veri kümesi koşullarına uyun.
- Model3, `distilbert-base-uncased-finetuned-sst-2-english` (Apache-2.0, Hugging Face) ile başlar, çalışma anında indirilir.
- Model Lab yazı tipleri: [Geist](https://github.com/vercel/geist-font) ve Geist Mono (Vercel), SIL Open Font License 1.1, `model_lab/ui/fonts/` içinde barındırılır.
- MEF Üniversitesi adı ve logosu (Model Lab'da görünür) MEF Üniversitesi'ne aittir, MIT lisansı kapsamında değildir; yalnızca ders bağlamını belirtmek için kullanılır.
- "Olduğu gibi" sunulur, garanti yoktur. MEF Üniversitesi bu depodan sorumlu değildir.

<div align="center">
  <br/>
  <sub>Küçük modeller, dürüst sayılar.</sub>
  <br/><br/>
  <a href="#top">↑ başa dön</a>
</div>
