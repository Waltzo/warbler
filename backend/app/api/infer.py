"""Inference / test endpoints: run a finetuned (or base) model on an uploaded
audio file. Supports single-shot and A/B (multiple targets, one audio)."""
import json
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .. import config
from ..core import job_manager

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
from inference import infer  # noqa: E402

router = APIRouter(prefix="/infer", tags=["infer"])


@router.get("/models")
def list_models() -> list[dict]:
    """Completed training jobs that have a saved model — usable as inference targets."""
    out = []
    for st in job_manager.list_jobs():
        if st.get("kind", "train") != "train" or st.get("state") != "done":
            continue
        run_dir = config.RUNS_DIR / st["job_id"]
        if not (run_dir / "model").exists():
            continue
        try:
            meta = infer.resolve_finetuned(run_dir)
        except FileNotFoundError:
            continue
        out.append({"job_id": st["job_id"], "name": st.get("name"),
                    "model_type": meta["model_type"], "base_model": meta["base_model"],
                    "lora": meta["lora"]})
    return out


@router.post("")
async def run_infer(
    audio: UploadFile = File(...),
    targets: str = Form(...),  # JSON list
    gpu_index: int = Form(config.DEFAULT_GPU_INDEX),
    language: str = Form(""),
) -> dict:
    try:
        target_list = json.loads(targets)
    except json.JSONDecodeError:
        raise HTTPException(400, "targets must be JSON")
    if not target_list:
        raise HTTPException(400, "no targets")

    suffix = Path(audio.filename or "a.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    results = []
    try:
        for t in target_list:
            kind = t.get("kind")
            try:
                if kind == "finetuned":
                    meta = infer.resolve_finetuned(config.RUNS_DIR / t["job_id"])
                    label = t.get("label") or f"finetuned:{t['job_id']}"
                    src, mtype, base, lora = (meta["model_dir"], meta["model_type"],
                                              meta["base_model"], meta["lora"])
                elif kind == "base":
                    label = t.get("label") or f"base:{t['base_model']}"
                    src, mtype, base, lora = (t["base_model"], t["model_type"],
                                              t["base_model"], False)
                else:
                    raise ValueError(f"unknown target kind: {kind}")

                t0 = time.perf_counter()
                text = await run_in_threadpool(
                    infer.transcribe, audio_path, mtype, src, base, lora,
                    gpu_index, language or None,
                )
                results.append({"label": label, "text": text,
                                "ms": round((time.perf_counter() - t0) * 1000)})
            except Exception as e:  # noqa: BLE001
                results.append({"label": t.get("label") or kind or "?",
                                "error": f"{type(e).__name__}: {e}"})
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {"results": results}
