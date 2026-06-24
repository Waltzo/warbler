"""In-process inference for finetuned / base STT models.

Models are loaded lazily and cached (single-user tool) so A/B comparison and
repeated single-shot tests are fast. Supports whisper (seq2seq) and wav2vec2
(CTC), full finetune and LoRA adapters.
"""
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

_CACHE_MAX = 3
_cache: "OrderedDict[str, _Loaded]" = OrderedDict()


class _Loaded:
    def __init__(self, model, processor, model_type, device):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.device = device


def _is_lora(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").exists()


def resolve_finetuned(run_dir: Path) -> dict:
    """Return {model_type, base_model, model_dir, lora} for a completed train run."""
    cfg = json.loads((run_dir / "config.json").read_text())
    model_dir = run_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"No saved model at {model_dir}")
    return {
        "model_type": cfg["model_type"],
        "base_model": cfg["base_model"],
        "model_dir": str(model_dir),
        "lora": _is_lora(model_dir),
    }


def _load(key: str, model_type: str, source: str, base_model: Optional[str],
          lora: bool, device: str) -> _Loaded:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    import torch  # noqa: F401

    if model_type == "whisper":
        loaded = _load_whisper(source, base_model, lora, device)
    elif model_type == "wav2vec2":
        loaded = _load_wav2vec2(source, base_model, lora, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    _cache[key] = loaded
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return loaded


def _load_whisper(source, base_model, lora, device):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if lora:
        from peft import PeftModel
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, source)
        processor = WhisperProcessor.from_pretrained(source)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(source)
        processor = WhisperProcessor.from_pretrained(source)
    model.to(device).eval()
    return _Loaded(model, processor, "whisper", device)


def _load_wav2vec2(source, base_model, lora, device):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if lora:
        from peft import PeftModel
        # Adapter was trained on a rebuilt vocab; size the base CTC head to match
        # the saved processor before loading the adapter (incl. modules_to_save lm_head).
        processor = Wav2Vec2Processor.from_pretrained(source)
        model = Wav2Vec2ForCTC.from_pretrained(
            base_model, vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(model, source)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(source)
        processor = Wav2Vec2Processor.from_pretrained(source)
    model.to(device).eval()
    return _Loaded(model, processor, "wav2vec2", device)


def transcribe(audio_path: str, model_type: str, source: str,
               base_model: Optional[str] = None, lora: bool = False,
               gpu_index: int = 0, language: Optional[str] = None) -> str:
    """Load (cached) and transcribe a single audio file. Returns text."""
    import librosa
    import torch

    device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    key = f"{model_type}|{source}|{lora}|{device}"
    L = _load(key, model_type, source, base_model, lora, device)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    with torch.no_grad():
        if L.model_type == "whisper":
            feats = L.processor(audio, sampling_rate=16000,
                                return_tensors="pt").input_features.to(device)
            gen_kwargs = {"task": "transcribe"}
            if language:
                gen_kwargs["language"] = language
            ids = L.model.generate(feats, **gen_kwargs)
            return L.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        else:
            vals = L.processor(audio, sampling_rate=16000,
                               return_tensors="pt").input_values.to(device)
            logits = L.model(vals).logits
            ids = torch.argmax(logits, dim=-1)
            return L.processor.batch_decode(ids)[0].strip()
