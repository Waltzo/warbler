"""Dataset loading from a manifest (jsonl/csv) into a HF Dataset.

Manifest rows must contain:
  - audio_path : path to wav/mp3/flac (absolute, or relative to audio_root)
  - text       : reference transcript
"""
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLING_RATE = 16000


def read_manifest(manifest_path: str, audio_root: str | None = None) -> list[dict]:
    """Parse a .jsonl or .csv manifest into a list of {audio, text} rows."""
    mpath = Path(manifest_path)
    root = Path(audio_root) if audio_root else mpath.parent
    rows: list[dict] = []

    if mpath.suffix == ".jsonl":
        with open(mpath, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
    elif mpath.suffix == ".csv":
        with open(mpath, encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported manifest type: {mpath.suffix} (use .jsonl or .csv)")

    for r in raw:
        ap = r.get("audio_path") or r.get("audio") or r.get("path")
        text = r.get("text") or r.get("transcript") or r.get("sentence")
        if ap is None or text is None:
            raise ValueError(f"Manifest row missing audio_path/text: {r}")
        ap = ap if os.path.isabs(ap) else str(root / ap)
        rows.append({"audio": ap, "text": str(text)})
    return rows


def build_dataset(manifest_path: str, audio_root: str | None, eval_ratio: float):
    """Return (train_ds, eval_ds) HF Datasets with a 16kHz Audio column."""
    from datasets import Audio, Dataset

    rows = read_manifest(manifest_path, audio_root)
    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    if eval_ratio and 0 < eval_ratio < 1 and len(ds) > 1:
        split = ds.train_test_split(test_size=eval_ratio, seed=42)
        return split["train"], split["test"]
    return ds, None


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------
@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip the BOS if the tokenizer prepended it (Whisper adds it during decode).
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


@dataclass
class CTCCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_values = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_values, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.pad(labels=label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
