# Model3 aday benchmark'ı (fine-tune öncesi, zero-shot)

Aynı 250 tweet (Dataset2), aynı ölçütler (`lib/common.py`). Hiçbir modele eğitim yapılmadı. Taban çizgisi: hep `neutral` demek doğruluk 0.460, macro-F1 0.210.

| Aday | Dil | Parametre | Doğruluk | Macro-F1 | neg F1 | neu F1 | pos F1 | Sızıntı riski |
|---|---|---|---|---|---|---|---|---|
| [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | EN | 67 M | 0.448 | 0.389 | 0.459 | 0.000 | 0.709 | yok (film yorumları, SST-2) |
| [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | EN | 125 M | 0.772 | 0.775 | 0.774 | 0.740 | 0.811 | VAR olabilir (TweetEval duygu eğitim verisi) |
| [`finiteautomata/bertweet-base-sentiment-analysis`](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) | EN | 135 M | 0.840 | 0.846 | 0.866 | 0.823 | 0.848 | VAR olabilir (SemEval-2017, TweetEval'in kaynağı) |

## Adaylar

- **`distilbert/distilbert-base-uncased-finetuned-sst-2-english`**: Mevcut Model3 tabanı. İki sınıflı: neutral hiç tahmin edemez, bu yüzden fine-tune şart.
- **`cardiffnlp/twitter-roberta-base-sentiment-latest`**: Tweet için en güçlü hazır taban; üç sınıfı zaten biliyor. Sızıntı riski yüzünden sonuçlar şişkin olabilir.
- **`finiteautomata/bertweet-base-sentiment-analysis`**: BERTweet: tweet'lerle ön eğitilmiş RoBERTa. Üç sınıflı, SemEval-2017 üzerinde ince ayarlı.

Bu koşuda çalıştırılmadı (indirme tamamlanamadı; `python tools/benchmark.py --only nlptown_multilingual` ile çalıştırılabilir):

- **`nlptown/bert-base-multilingual-uncased-sentiment`**: Sızıntısız ve çok dilli; 1-5 yıldızı 3 sınıfa topluyoruz. Alan farkı (yorum -> tweet) nedeniyle zero-shot zayıf, fine-tune ile toparlanması beklenir.

Sızıntı notu: Dataset2, TweetEval duygu bölümlerinin birleşiminden ayrıldı. Tweet'lerde eğitilmiş adayların eğitim verisi bu tweet'leri içerebilir; bu satırlar bir üst sınır olarak okunmalı, adil karşılaştırma için sızıntısız adaylarla ayrı tutulmalı.
