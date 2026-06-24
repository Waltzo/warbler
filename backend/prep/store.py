"""Corpus storage helpers — shared by the transcribe subprocess (prep.transcribe)
and the web API (app.api.corpus). Operates purely on a corpus directory path,
so it has no dependency on the web app's config.

Layout under <corpus_dir>:
    meta.json        corpus metadata
    clips/<n>.wav    sliced utterance clips (16k mono)
    segments.jsonl   one JSON object per segment
    manifest.jsonl   created on export
"""
import json
from pathlib import Path
from typing import Optional

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def meta_path(corpus_dir: Path) -> Path:
    return Path(corpus_dir) / "meta.json"


def segments_path(corpus_dir: Path) -> Path:
    return Path(corpus_dir) / "segments.jsonl"


def clips_dir(corpus_dir: Path) -> Path:
    return Path(corpus_dir) / "clips"


def read_meta(corpus_dir: Path) -> dict:
    return json.loads(meta_path(corpus_dir).read_text(encoding="utf-8"))


def write_meta(corpus_dir: Path, meta: dict) -> None:
    p = meta_path(corpus_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    tmp.replace(p)


def scan_audio_files(audio_root: Path) -> list[Path]:
    return sorted(
        p for p in Path(audio_root).rglob("*") if p.suffix.lower() in AUDIO_EXTS
    )


def read_segments(corpus_dir: Path) -> list[dict]:
    p = segments_path(corpus_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_segments(corpus_dir: Path, segments: list[dict]) -> None:
    """Atomic full rewrite (used for edits)."""
    p = segments_path(corpus_dir)
    tmp = p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    tmp.replace(p)


def append_segment(corpus_dir: Path, seg: dict) -> None:
    with open(segments_path(corpus_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def get_segment(corpus_dir: Path, seg_id: str) -> Optional[dict]:
    for s in read_segments(corpus_dir):
        if s["seg_id"] == seg_id:
            return s
    return None


def update_segment(corpus_dir: Path, seg_id: str, updates: dict) -> Optional[dict]:
    """Merge `updates` into the matching segment and rewrite. Returns it (or None)."""
    segs = read_segments(corpus_dir)
    found = None
    for s in segs:
        if s["seg_id"] == seg_id:
            s.update({k: v for k, v in updates.items() if v is not None})
            found = s
            break
    if found is not None:
        write_segments(corpus_dir, segs)
    return found


def counts(corpus_dir: Path) -> dict:
    segs = read_segments(corpus_dir)
    reviewed = sum(1 for s in segs if s.get("reviewed"))
    return {"segments": len(segs), "reviewed": reviewed}
