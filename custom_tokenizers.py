"""Model1 ve Model2 için iki farklı tokenizer, ikisi de yalnızca eğitim metinlerinden öğrenilir.

  char : karakter düzeyi (her karakter bir simge; sözlük küçük, diziler uzun)
  bpe  : alt sözcük düzeyi, ByteLevel BPE (sözlük ~8000; sık sözcükler tek parça, nadirler parçalanır)

Model3 ise hazır DistilBERT'in WordPiece tokenizer'ını kullanır (üçüncü, farklı bir tokenizer).
"""
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _wrap(tok):
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def build_char_tokenizer(texts):
    norm = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    chars = sorted({c for t in texts for c in norm.normalize_str(t)})
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = norm
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")   # her karakter ayrı simge (boşluk dahil)
    return _wrap(tok)


def build_bpe_tokenizer(texts, vocab_size=8000):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)
    tok.train_from_iterator(texts, trainer)
    return _wrap(tok)


BUILDERS = {"char": build_char_tokenizer, "bpe": build_bpe_tokenizer}
