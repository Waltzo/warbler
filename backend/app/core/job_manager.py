"""Manages the training subprocess lifecycle. Single-job-at-a-time.

Job artifacts live in runs/<job_id>/:
  config.json, status.json, metrics.jsonl, train.log, checkpoints/, model/
"""
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from .. import config
from ..schemas import TrainConfig

RUNS_DIR = config.RUNS_DIR

# Popen handles for jobs launched by THIS backend process. Lets us poll()/reap
# children (avoids zombies being misread as "alive"). Empty after a restart —
# then we fall back to a bare PID signal check.
_procs: dict[str, "subprocess.Popen"] = {}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _status_path(job_id: str) -> Path:
    return RUNS_DIR / job_id / "status.json"


def read_status(job_id: str) -> Optional[dict]:
    p = _status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_running(job_id: str, pid: Optional[int]) -> bool:
    """True if the job's process is still running. Prefers the Popen handle
    (which reaps zombies via poll()); falls back to a PID signal check."""
    p = _procs.get(job_id)
    if p is not None:
        return p.poll() is None
    return bool(pid) and _pid_alive(pid)


def reconcile(job_id: str) -> Optional[dict]:
    """Read status; if marked 'running' but the process has exited (and the
    subprocess never wrote its own terminal state), persist 'failed'."""
    st = read_status(job_id)
    if st and st.get("state") == "running":
        if not _is_running(job_id, st.get("pid")):
            st["state"] = "failed"
            st["error"] = st.get("error") or "process exited unexpectedly (see train.log)"
            st["finished_at"] = _now()
            _status_path(job_id).write_text(json.dumps(st, ensure_ascii=False, indent=2))
            _procs.pop(job_id, None)
    return st


def list_jobs() -> list[dict]:
    jobs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "status.json").exists():
            st = reconcile(d.name)
            if st:
                jobs.append(st)
    return jobs


def active_job() -> Optional[str]:
    """Return the job_id of a currently-running job, if any."""
    for d in RUNS_DIR.iterdir():
        if d.is_dir():
            st = reconcile(d.name)
            if st and st.get("state") == "running":
                return d.name
    return None


def _new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _launch(status: dict, cmd: list[str], gpu_index: int) -> dict:
    """Common subprocess launch: enforce single-job lock, pin GPU, write status.
    `status` must already contain job_id/kind/etc.; run_dir must exist."""
    running = active_job()
    if running:
        raise RuntimeError(f"A job is already running: {running}")

    job_id = status["job_id"]
    run_dir = RUNS_DIR / job_id
    status["state"] = "pending"
    status["created_at"] = _now()
    status["gpu_index"] = gpu_index
    (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    log_file = open(run_dir / "train.log", "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),  # backend/
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    _procs[job_id] = proc
    status["state"] = "running"
    status["pid"] = proc.pid
    status["started_at"] = _now()
    (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def start_job(cfg: TrainConfig, manifest_path: str, audio_root: str,
              audio_key: str = None, text_key: str = None) -> dict:
    """Launch a training subprocess. Raises RuntimeError if one is already running."""
    if active_job():
        raise RuntimeError(f"A job is already running: {active_job()}")

    job_id = _new_job_id()
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Merge resolved dataset paths into the config consumed by train.py.
    cfg_dict = cfg.model_dump()
    cfg_dict["manifest_path"] = manifest_path
    cfg_dict["audio_root"] = audio_root
    cfg_dict["audio_key"] = audio_key
    cfg_dict["text_key"] = text_key
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2))

    status = {
        "job_id": job_id,
        "kind": "train",
        "name": cfg.name,
        "model_type": cfg.model_type,
        "base_model": cfg.base_model,
        "dataset_id": cfg.dataset_id,
    }
    cmd = [config.PYTHON_BIN, "-m", "training.train",
           "--config", str(run_dir / "config.json")]
    return _launch(status, cmd, cfg.gpu_index)


def start_transcribe_job(corpus_id: str, corpus_dir: str, model: str,
                         gpu_index: int, language: str = "ko") -> dict:
    """Launch a faster-whisper transcription subprocess for a corpus."""
    if active_job():
        raise RuntimeError(f"A job is already running: {active_job()}")

    job_id = _new_job_id()
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "job_id": job_id,
        "kind": "transcribe",
        "name": f"transcribe:{corpus_id}",
        "corpus_id": corpus_id,
        "model_type": "faster-whisper",
        "base_model": model,
        "dataset_id": corpus_id,
    }
    cmd = [config.PYTHON_BIN, "-m", "prep.transcribe",
           "--corpus-dir", str(corpus_dir), "--run-dir", str(run_dir),
           "--model", model, "--language", language]
    return _launch(status, cmd, gpu_index)


def stop_job(job_id: str) -> dict:
    st = reconcile(job_id)
    if not st:
        raise FileNotFoundError(job_id)
    pid = st.get("pid")
    if st.get("state") == "running" and pid and _pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):  # up to ~5s grace
            if not _pid_alive(pid):
                break
            time.sleep(0.25)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
    st["state"] = "stopped"
    st["finished_at"] = _now()
    (_status_path(job_id)).write_text(json.dumps(st, ensure_ascii=False, indent=2))
    return st


def read_metrics(job_id: str) -> list[dict]:
    p = RUNS_DIR / job_id / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out
