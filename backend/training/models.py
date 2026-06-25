"""Model/processor builders for whisper (seq2seq) and wav2vec2 (CTC).

build_bundle() returns a ModelBundle with everything train.py needs:
processor, model, collator, preprocess_fn, compute_metrics, is_seq2seq.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .data import CTCCollator, WhisperCollator, load_audio
from .metrics import compute_wer_cer


def _log_samples(refs: list[str], hyps: list[str], n: int = 5) -> None:
    """Print a few (정답/예측) pairs so eval WER/CER can be sanity-checked from
    the log — a number alone can't tell garbage output from a metric bug."""
    print("[eval] 샘플 정답 vs 예측:", flush=True)
    for r, h in zip(refs[:n], hyps[:n]):
        print(f"  정답: {r!r}", flush=True)
        print(f"  예측: {h!r}", flush=True)
        print("  ---", flush=True)


@dataclass
class ModelBundle:
    processor: Any
    model: Any
    collator: Any
    preprocess_fn: Callable
    compute_metrics: Callable
    is_seq2seq: bool


def build_bundle(cfg: dict, train_texts: Optional[list[str]] = None,
                 run_dir: Optional[Path] = None) -> ModelBundle:
    if cfg["model_type"] == "whisper":
        return _build_whisper(cfg)
    elif cfg["model_type"] == "wav2vec2":
        return _build_wav2vec2(cfg, train_texts or [], run_dir)
    raise ValueError(f"Unknown model_type: {cfg['model_type']}")


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
def _build_whisper(cfg: dict) -> ModelBundle:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = cfg["base_model"]
    lang = cfg.get("language")
    task = cfg.get("task", "transcribe")

    import torch

    processor = WhisperProcessor.from_pretrained(base, language=lang, task=task)
    # whisper-large-v3 체크포인트는 fp16으로 저장돼 있고, 최신 transformers는
    # 원본 dtype을 따라 fp16으로 로드한다. fp16 마스터 가중치는 bf16 autocast와
    # 충돌해 grad_norm=nan을 내고, eval generate는 conv에서 (float input vs Half
    # bias) dtype 크래시를 낸다. fp32로 강제 로드 → bf16/fp16 mixed-precision은
    # Trainer의 autocast가 처리한다.
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)

    model.generation_config.language = lang
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if cfg.get("use_lora"):
        model = _apply_lora(model, cfg, ["q_proj", "v_proj"])

    def preprocess(batch: dict) -> dict:
        audio = load_audio(batch["audio"])
        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        _log_samples(refs, hyps)
        return compute_wer_cer(refs, hyps)

    return ModelBundle(
        processor=processor,
        model=model,
        collator=WhisperCollator(processor),
        preprocess_fn=preprocess,
        compute_metrics=compute_metrics,
        is_seq2seq=True,
    )


# ---------------------------------------------------------------------------
# wav2vec2 (CTC)
# ---------------------------------------------------------------------------
_CHARS_TO_STRIP = re.compile(r"[\,\?\.\!\-\;\:\"“”‘’]")


def _clean_text(t: str) -> str:
    """Normalize + strip punctuation. NFC keeps modern syllables composed and
    old-Hangul (아래아 ㆍ 등) jamo consistent, so the same character maps to a
    single vocab token. Applied identically for vocab build and label encoding."""
    import unicodedata

    t = unicodedata.normalize("NFC", t)
    return _CHARS_TO_STRIP.sub("", t).lower()


def _build_vocab(texts: list[str], run_dir: Path) -> str:
    """Build a char-level vocab.json from training transcripts. Returns its path."""
    chars = set()
    for t in texts:
        t = _clean_text(t)
        chars.update(t.replace(" ", ""))
    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)  # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    run_dir.mkdir(parents=True, exist_ok=True)
    vpath = run_dir / "vocab.json"
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return str(vpath)


def _build_wav2vec2(cfg: dict, train_texts: list[str], run_dir: Optional[Path]) -> ModelBundle:
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    base = cfg["base_model"]
    run_dir = run_dir or Path(".")
    vocab_path = _build_vocab(train_texts, run_dir)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    # Persist processor so checkpoints are self-contained.
    processor.save_pretrained(str(run_dir))

    model = Wav2Vec2ForCTC.from_pretrained(
        base,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    if cfg.get("use_lora"):
        # lm_head must be trained fully (CTC head was reinit for the new vocab);
        # leaving it frozen at random init makes WER stick at 1.0.
        model = _apply_lora(model, cfg, ["q_proj", "v_proj", "k_proj", "out_proj"],
                            modules_to_save=["lm_head"])
        _force_no_embedding_save(model)

    def preprocess(batch: dict) -> dict:
        audio = load_audio(batch["audio"])
        batch["input_values"] = processor(
            audio, sampling_rate=16000
        ).input_values[0]
        text = _clean_text(batch["text"])
        batch["labels"] = processor(text=text).input_ids
        return batch

    def compute_metrics(pred) -> dict:
        import numpy as np

        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(pred_ids)
        refs = processor.batch_decode(label_ids, group_tokens=False)
        _log_samples(refs, hyps)
        return compute_wer_cer(refs, hyps)

    return ModelBundle(
        processor=processor,
        model=model,
        collator=CTCCollator(processor),
        preprocess_fn=preprocess,
        compute_metrics=compute_metrics,
        is_seq2seq=False,
    )


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def _apply_lora(model, cfg: dict, default_targets: list[str],
                modules_to_save: list[str] | None = None):
    from peft import LoraConfig, get_peft_model

    lconf = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=default_targets,
        modules_to_save=modules_to_save,
        bias="none",
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()
    return model


def _force_no_embedding_save(peft_model):
    """PEFT auto-detects the resized CTC head as a 'resized embedding' and tries
    to save input/output embeddings — which Wav2Vec2 doesn't implement, crashing
    save_pretrained. The lm_head is already persisted via modules_to_save, so
    force save_embedding_layers=False on every save call."""
    import functools

    orig = peft_model.save_pretrained
    peft_model.save_pretrained = functools.partial(orig, save_embedding_layers=False)
