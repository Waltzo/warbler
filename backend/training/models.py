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

    processor = WhisperProcessor.from_pretrained(base, language=lang, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(base)

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


def _build_vocab(texts: list[str], run_dir: Path) -> str:
    """Build a char-level vocab.json from training transcripts. Returns its path."""
    chars = set()
    for t in texts:
        t = _CHARS_TO_STRIP.sub("", t).lower()
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
        model = _apply_lora(model, cfg, ["q_proj", "v_proj", "k_proj", "out_proj"])

    def preprocess(batch: dict) -> dict:
        audio = load_audio(batch["audio"])
        batch["input_values"] = processor(
            audio, sampling_rate=16000
        ).input_values[0]
        text = _CHARS_TO_STRIP.sub("", batch["text"]).lower()
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
def _apply_lora(model, cfg: dict, default_targets: list[str]):
    from peft import LoraConfig, get_peft_model

    lconf = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=default_targets,
        bias="none",
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()
    return model
