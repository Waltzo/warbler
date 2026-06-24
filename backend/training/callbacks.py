"""TrainerCallback that streams progress to runs/<job_id>/ files.

  - metrics.jsonl : one JSON object per log/eval event (consumed by the web UI)
  - status.json   : current job state (read by the backend job_manager)
"""
import json
import time
from pathlib import Path

from transformers import TrainerCallback


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_status(run_dir: Path, **fields) -> None:
    """Merge fields into status.json (atomic write)."""
    path = run_dir / "status.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    data.update(fields)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


class ProgressCallback(TrainerCallback):
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def _emit(self, record: dict) -> None:
        record["ts"] = _now()
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        write_status(self.run_dir, state="running", started_at=_now(),
                     total_steps=int(state.max_steps))

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        rec = {"event": "log", "step": int(state.global_step),
               "epoch": round(state.epoch or 0, 3)}
        for k in ("loss", "learning_rate", "grad_norm"):
            if k in logs:
                rec[k] = logs[k]
        if "loss" in logs:
            self._emit(rec)
        write_status(self.run_dir, last_step=int(state.global_step),
                     total_steps=int(state.max_steps))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        rec = {"event": "eval", "step": int(state.global_step),
               "epoch": round(state.epoch or 0, 3)}
        for k, v in metrics.items():
            # eval_loss, eval_wer, eval_cer, ...
            rec[k.replace("eval_", "")] = v
        self._emit(rec)

    def on_train_end(self, args, state, control, **kwargs):
        # Final state set in train.py after model save (to mark "done" reliably).
        write_status(self.run_dir, last_step=int(state.global_step))
