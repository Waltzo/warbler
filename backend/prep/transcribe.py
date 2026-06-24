"""Auto-segment + draft-transcribe a corpus of raw audio with faster-whisper.

Runs as a standalone GPU subprocess (launched by app.core.job_manager):

    CUDA_VISIBLE_DEVICES=N python -m prep.transcribe \
        --corpus-dir datasets/<id> --run-dir runs/<job_id> \
        --model large-v3 --language ko

For each audio file: Whisper (with VAD) yields timestamped segments. Each
segment is sliced to a 16k mono clip and recorded in segments.jsonl with a
draft transcript for later human correction. Progress -> runs/<job_id>/.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import store

TARGET_SR = 16000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_status(run_dir: Path, **fields) -> None:
    p = run_dir / "status.json"
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            data = {}
    data.update(fields)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(p)


def _emit(run_dir: Path, record: dict) -> None:
    record["ts"] = _now()
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(corpus_dir: str, run_dir: str, model: str, language: str) -> int:
    corpus_dir = Path(corpus_dir)
    run_dir = Path(run_dir)
    try:
        _run(corpus_dir, run_dir, model, language)
    except Exception as e:  # noqa: BLE001
        print(traceback.format_exc(), flush=True)
        _write_status(run_dir, state="failed", error=f"{type(e).__name__}: {e}",
                      finished_at=_now())
        return 1
    _write_status(run_dir, state="done", finished_at=_now())
    return 0


def _run(corpus_dir: Path, run_dir: Path, model: str, language: str) -> None:
    import librosa
    import soundfile as sf
    from faster_whisper import WhisperModel

    meta = store.read_meta(corpus_dir)
    audio_root = Path(meta["audio_root"])
    files = store.scan_audio_files(audio_root)
    print(f"[prep] {len(files)} audio file(s) under {audio_root}", flush=True)

    clips = store.clips_dir(corpus_dir)
    clips.mkdir(parents=True, exist_ok=True)
    # Fresh run: reset segments file.
    store.write_segments(corpus_dir, [])

    _write_status(run_dir, state="running", started_at=_now(),
                  total_steps=len(files))
    print(f"[prep] loading faster-whisper {model}", flush=True)
    wmodel = WhisperModel(model, device="cuda", compute_type="float16")

    seg_no = 0
    for fi, fpath in enumerate(files):
        print(f"[prep] ({fi + 1}/{len(files)}) {fpath.name}", flush=True)
        # Whole-file audio for slicing (16k mono).
        audio, _ = librosa.load(str(fpath), sr=TARGET_SR, mono=True)

        segments, info = wmodel.transcribe(
            str(fpath), language=language or None, vad_filter=True,
        )
        file_segs = 0
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            seg_no += 1
            file_segs += 1
            a = max(0, int(s.start * TARGET_SR))
            b = min(len(audio), int(s.end * TARGET_SR))
            if b <= a:
                continue
            clip_name = f"{seg_no:06d}.wav"
            sf.write(str(clips / clip_name), audio[a:b], TARGET_SR)
            store.append_segment(corpus_dir, {
                "seg_id": f"{seg_no:06d}",
                "clip": f"clips/{clip_name}",
                "source_file": str(fpath),
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "duration": round(float(s.end - s.start), 3),
                "draft_text": text,
                "text": text,
                "reviewed": False,
            })
        _emit(run_dir, {"event": "progress", "step": fi + 1,
                        "total": len(files), "file": fpath.name,
                        "file_segments": file_segs, "total_segments": seg_no})
        _write_status(run_dir, last_step=fi + 1, total_steps=len(files),
                      total_segments=seg_no)

    # Stamp counts into meta for the UI.
    meta.update({"model": model, "language": language,
                 "transcribed_at": _now(), **store.counts(corpus_dir)})
    store.write_meta(corpus_dir, meta)
    print(f"[prep] done — {seg_no} segments", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="ko")
    a = ap.parse_args()
    sys.exit(main(a.corpus_dir, a.run_dir, a.model, a.language))
