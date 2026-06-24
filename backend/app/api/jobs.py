"""Training job endpoints: create / list / get / metrics / stop / SSE stream."""
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from .. import config
from ..core import job_manager, tailer
from ..schemas import TrainConfig
from .datasets import get_dataset
from .system import gpu_count

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("")
def create_job(cfg: TrainConfig) -> dict:
    ds = get_dataset(cfg.dataset_id)  # 404 if unknown

    n_gpu = gpu_count()
    if n_gpu and not (0 <= cfg.gpu_index < n_gpu):
        raise HTTPException(400, f"gpu_index {cfg.gpu_index} out of range (0..{n_gpu - 1})")

    try:
        return job_manager.start_job(cfg, ds["manifest_path"], ds["audio_root"],
                                     ds.get("audio_key"), ds.get("text_key"))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.get("")
def list_jobs() -> list[dict]:
    return job_manager.list_jobs()


@router.get("/active")
def get_active() -> dict:
    return {"job_id": job_manager.active_job()}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    st = job_manager.reconcile(job_id)
    if not st:
        raise HTTPException(404, job_id)
    return st


@router.get("/{job_id}/metrics")
def get_metrics(job_id: str) -> list[dict]:
    if not job_manager.read_status(job_id):
        raise HTTPException(404, job_id)
    return job_manager.read_metrics(job_id)


@router.post("/{job_id}/stop")
def stop(job_id: str) -> dict:
    try:
        return job_manager.stop_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, job_id)


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    if not job_manager.read_status(job_id):
        raise HTTPException(404, job_id)
    return EventSourceResponse(tailer.stream_job(job_id))
