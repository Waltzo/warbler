"""SSE event generator: tails metrics.jsonl + train.log for a running job."""
import asyncio
import json
from pathlib import Path

from .. import config
from . import job_manager


async def stream_job(job_id: str):
    """Async generator yielding sse-starlette event dicts.

    Emits past metrics/log immediately, then tails new lines, then a final
    'status' event when the job is no longer running.
    """
    run_dir = config.RUNS_DIR / job_id
    metrics_path = run_dir / "metrics.jsonl"
    log_path = run_dir / "train.log"

    metric_pos = 0
    log_pos = 0

    while True:
        # New metric lines
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                f.seek(metric_pos)
                for line in f:
                    if line.strip():
                        yield {"event": "metric", "data": line.strip()}
                metric_pos = f.tell()

        # New log lines
        if log_path.exists():
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(log_pos)
                chunk = f.read()
                log_pos = f.tell()
            if chunk:
                yield {"event": "log", "data": json.dumps({"text": chunk})}

        st = job_manager.reconcile(job_id)
        if st:
            yield {"event": "status", "data": json.dumps(st)}
            if st.get("state") in ("done", "failed", "stopped"):
                break

        await asyncio.sleep(1.0)
