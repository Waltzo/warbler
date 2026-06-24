"""Data-prep (labeling) pipeline: raw audio -> draft transcripts -> corrected
manifest dataset. A "corpus" is a labeling project under datasets/<corpus_id>/.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import config
from ..core import job_manager
from . import datasets as datasets_api
from .system import gpu_count

# prep.store lives under backend/ (sibling of app/).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prep import store  # noqa: E402

router = APIRouter(prefix="/corpus", tags=["corpus"])


def _corpus_dir(corpus_id: str) -> Path:
    return config.DATASETS_DIR / corpus_id


def _require(corpus_id: str) -> Path:
    d = _corpus_dir(corpus_id)
    if not store.meta_path(d).exists():
        raise HTTPException(404, f"Unknown corpus: {corpus_id}")
    return d


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CorpusCreate(BaseModel):
    corpus_id: str
    audio_root: str


class TranscribeReq(BaseModel):
    model: str = "large-v3"
    language: str = "ko"
    gpu_index: int = config.DEFAULT_GPU_INDEX


class SegmentPatch(BaseModel):
    text: Optional[str] = None
    reviewed: Optional[bool] = None


class ExportReq(BaseModel):
    dataset_id: str
    only_reviewed: bool = True


# ---------------------------------------------------------------------------
# Corpus CRUD
# ---------------------------------------------------------------------------
@router.post("")
def create_corpus(req: CorpusCreate) -> dict:
    if not os.path.isdir(req.audio_root):
        raise HTTPException(400, f"audio_root not a directory: {req.audio_root}")
    d = _corpus_dir(req.corpus_id)
    if store.meta_path(d).exists():
        raise HTTPException(409, f"Corpus already exists: {req.corpus_id}")

    files = store.scan_audio_files(Path(req.audio_root))
    if not files:
        raise HTTPException(400, f"No audio files under {req.audio_root}")

    meta = {"corpus_id": req.corpus_id, "audio_root": req.audio_root,
            "num_files": len(files), "segments": 0, "reviewed": 0}
    store.write_meta(d, meta)
    return meta


@router.get("")
def list_corpus() -> list[dict]:
    out = []
    if not config.DATASETS_DIR.exists():
        return out
    for d in sorted(config.DATASETS_DIR.iterdir()):
        if d.is_dir() and store.meta_path(d).exists():
            meta = store.read_meta(d)
            meta.update(store.counts(d))
            out.append(meta)
    return out


@router.get("/{corpus_id}")
def get_corpus(corpus_id: str) -> dict:
    d = _require(corpus_id)
    meta = store.read_meta(d)
    meta.update(store.counts(d))
    return meta


# ---------------------------------------------------------------------------
# Transcribe (GPU job)
# ---------------------------------------------------------------------------
@router.post("/{corpus_id}/transcribe")
def transcribe(corpus_id: str, req: TranscribeReq) -> dict:
    d = _require(corpus_id)
    n_gpu = gpu_count()
    if n_gpu and not (0 <= req.gpu_index < n_gpu):
        raise HTTPException(400, f"gpu_index {req.gpu_index} out of range (0..{n_gpu - 1})")
    try:
        return job_manager.start_transcribe_job(
            corpus_id, str(d), req.model, req.gpu_index, req.language
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))


# ---------------------------------------------------------------------------
# Segments review
# ---------------------------------------------------------------------------
@router.get("/{corpus_id}/segments")
def list_segments(corpus_id: str, only_unreviewed: bool = False,
                  limit: int = 100, offset: int = 0) -> dict:
    d = _require(corpus_id)
    segs = store.read_segments(d)
    if only_unreviewed:
        segs = [s for s in segs if not s.get("reviewed")]
    total = len(segs)
    return {"total": total, "items": segs[offset: offset + limit]}


@router.get("/{corpus_id}/segments/{seg_id}/audio")
def segment_audio(corpus_id: str, seg_id: str):
    d = _require(corpus_id)
    clip = store.clips_dir(d) / f"{seg_id}.wav"
    if not clip.exists():
        raise HTTPException(404, f"clip not found: {seg_id}")
    return FileResponse(str(clip), media_type="audio/wav")


@router.patch("/{corpus_id}/segments/{seg_id}")
def patch_segment(corpus_id: str, seg_id: str, body: SegmentPatch) -> dict:
    d = _require(corpus_id)
    seg = store.update_segment(d, seg_id, text=body.text, reviewed=body.reviewed)
    if seg is None:
        raise HTTPException(404, f"segment not found: {seg_id}")
    return seg


# ---------------------------------------------------------------------------
# Export -> manifest + register dataset
# ---------------------------------------------------------------------------
@router.post("/{corpus_id}/export")
def export(corpus_id: str, req: ExportReq) -> dict:
    d = _require(corpus_id)
    segs = store.read_segments(d)
    if req.only_reviewed:
        segs = [s for s in segs if s.get("reviewed")]
    segs = [s for s in segs if (s.get("text") or "").strip()]
    if not segs:
        raise HTTPException(400, "No segments to export (need reviewed, non-empty text)")

    manifest = d / "manifest.jsonl"
    import json
    with open(manifest, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps({"audio_path": s["clip"], "text": s["text"]},
                               ensure_ascii=False) + "\n")

    # Reuse the datasets registry. clip paths are relative to the corpus dir.
    return datasets_api.register(datasets_api.DatasetRegister(
        dataset_id=req.dataset_id, manifest_path=str(manifest), audio_root=str(d),
    )).model_dump()
