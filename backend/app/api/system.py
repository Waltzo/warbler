"""GPU stats via nvidia-smi."""
import shutil
import subprocess

from fastapi import APIRouter

from .. import config

router = APIRouter(prefix="/system", tags=["system"])

_FIELDS = "index,name,memory.used,memory.total,utilization.gpu"


def query_gpus() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_FIELDS}", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, mem_used, mem_total, util = parts
        gpus.append({
            "index": int(idx),
            "name": name,
            "memory_used_mb": float(mem_used),
            "memory_total_mb": float(mem_total),
            "utilization_pct": float(util),
        })
    return gpus


def gpu_count() -> int:
    return len(query_gpus())


@router.get("/gpus")
def gpus() -> list[dict]:
    return query_gpus()


@router.get("/info")
def info() -> dict:
    return {"default_gpu_index": config.DEFAULT_GPU_INDEX,
            "suggested_models": config.SUGGESTED_MODELS}
