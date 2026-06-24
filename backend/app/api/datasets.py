"""Dataset registration & preview.

A dataset is just a manifest (.jsonl/.csv) on the server filesystem plus an
audio_root. We keep a small registry at datasets/registry.json.
"""
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import config
from ..schemas import DatasetInfo, DatasetRegister

router = APIRouter(prefix="/datasets", tags=["datasets"])

REGISTRY = config.DATASETS_DIR / "registry.json"


def _load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def _read_manifest_rows(manifest_path: str, audio_root: str | None) -> list[dict]:
    # Reuse the training-side parser to guarantee identical semantics.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
    from training.data import read_manifest
    return read_manifest(manifest_path, audio_root)


def get_dataset(dataset_id: str) -> dict:
    reg = _load_registry()
    if dataset_id not in reg:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")
    return reg[dataset_id]


@router.post("", response_model=DatasetInfo)
def register(req: DatasetRegister) -> DatasetInfo:
    if not os.path.exists(req.manifest_path):
        raise HTTPException(400, f"Manifest not found: {req.manifest_path}")
    audio_root = req.audio_root or str(Path(req.manifest_path).parent)

    try:
        rows = _read_manifest_rows(req.manifest_path, audio_root)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Failed to parse manifest: {e}")

    if not rows:
        raise HTTPException(400, "Manifest is empty")

    missing = [r["audio"] for r in rows[:50] if not os.path.exists(r["audio"])]
    if missing:
        raise HTTPException(
            400, f"{len(missing)} audio file(s) missing (first: {missing[0]})"
        )

    reg = _load_registry()
    reg[req.dataset_id] = {
        "dataset_id": req.dataset_id,
        "manifest_path": req.manifest_path,
        "audio_root": audio_root,
        "num_samples": len(rows),
    }
    _save_registry(reg)

    return DatasetInfo(
        dataset_id=req.dataset_id,
        manifest_path=req.manifest_path,
        audio_root=audio_root,
        num_samples=len(rows),
        preview=[{"audio": Path(r["audio"]).name, "text": r["text"]}
                 for r in rows[: config.PREVIEW_ROWS]],
    )


@router.get("", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    reg = _load_registry()
    out = []
    for d in reg.values():
        out.append(DatasetInfo(
            dataset_id=d["dataset_id"],
            manifest_path=d["manifest_path"],
            audio_root=d["audio_root"],
            num_samples=d.get("num_samples", 0),
            preview=[],
        ))
    return out


@router.get("/{dataset_id}", response_model=DatasetInfo)
def dataset_detail(dataset_id: str) -> DatasetInfo:
    d = get_dataset(dataset_id)
    rows = _read_manifest_rows(d["manifest_path"], d["audio_root"])
    return DatasetInfo(
        dataset_id=d["dataset_id"],
        manifest_path=d["manifest_path"],
        audio_root=d["audio_root"],
        num_samples=len(rows),
        preview=[{"audio": Path(r["audio"]).name, "text": r["text"]}
                 for r in rows[: config.PREVIEW_ROWS]],
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict:
    reg = _load_registry()
    if dataset_id not in reg:
        raise HTTPException(404, dataset_id)
    del reg[dataset_id]
    _save_registry(reg)
    return {"ok": True}
