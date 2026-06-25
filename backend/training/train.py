"""Training entrypoint. Runs as a standalone subprocess:

    CUDA_VISIBLE_DEVICES=<n> python -m training.train --config runs/<id>/config.json

Writes progress to runs/<id>/{metrics.jsonl,status.json}. stdout/stderr are
captured to runs/<id>/train.log by the backend job_manager.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from .callbacks import ProgressCallback, write_status
from .data import build_dataset
from .models import build_bundle


def main(config_path: str) -> int:
    cfg = json.loads(Path(config_path).read_text())
    run_dir = Path(config_path).parent

    try:
        _run(cfg, run_dir)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        write_status(run_dir, state="failed", error=f"{type(e).__name__}: {e}",
                     finished_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"))
        return 1

    write_status(run_dir, state="done",
                 finished_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


def _run(cfg: dict, run_dir: Path) -> None:
    import torch

    print(f"[train] loading dataset {cfg['dataset_id']}", flush=True)
    train_ds, eval_ds = build_dataset(
        cfg["manifest_path"], cfg.get("audio_root"), cfg.get("eval_ratio", 0.1),
        cfg.get("audio_key"), cfg.get("text_key"),
    )
    print(f"[train] train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}", flush=True)

    # Vocab must cover EVERY character in the data (train + eval). Building it
    # from train only makes eval-only chars (e.g. 씬) decode/encode to [UNK].
    all_texts = list(train_ds["text"])
    if eval_ds is not None:
        all_texts += list(eval_ds["text"])
    bundle = build_bundle(cfg, train_texts=all_texts, run_dir=run_dir)

    print("[train] preprocessing", flush=True)
    remove_cols = train_ds.column_names
    train_ds = train_ds.map(bundle.preprocess_fn, remove_columns=remove_cols)
    if eval_ds is not None:
        eval_ds = eval_ds.map(bundle.preprocess_fn, remove_columns=remove_cols)

    precision = cfg.get("precision", "fp16")
    fp16 = precision == "fp16" and torch.cuda.is_available()
    bf16 = precision == "bf16" and torch.cuda.is_available()

    do_eval = eval_ds is not None
    eval_steps = cfg.get("eval_steps", 50)
    save_steps = cfg.get("save_steps", 200)
    # load_best_model_at_end requires save_steps to be a round multiple of
    # eval_steps. If misconfigured, align them so the best checkpoint is the one
    # actually restored at the end (otherwise it would silently disable).
    if do_eval and save_steps % eval_steps != 0:
        save_steps = eval_steps

    out_dir = str(run_dir / "checkpoints")
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.get("batch_size", 8),
        per_device_eval_batch_size=cfg.get("batch_size", 8),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        learning_rate=cfg.get("learning_rate", 1e-5),
        warmup_steps=cfg.get("warmup_steps", 50),
        num_train_epochs=cfg.get("num_epochs", 3.0),
        max_steps=cfg.get("max_steps", -1),
        fp16=fp16,
        bf16=bf16,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=save_steps,
        # Keep every checkpoint so a specific step can be promoted to the served
        # model later. save_total_limit=None disables pruning.
        save_total_limit=cfg.get("save_total_limit"),
        report_to=[],
        remove_unused_columns=False,
    )
    eval_args = dict(
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps,
    )
    # Restore the lowest eval_loss checkpoint into model/ at the end of training.
    best_args = dict(
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ) if do_eval else {}

    callback = ProgressCallback(run_dir)

    if bundle.is_seq2seq:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        args = Seq2SeqTrainingArguments(
            **common, **eval_args, **best_args,
            predict_with_generate=True, generation_max_length=225,
        )
        trainer = Seq2SeqTrainer(
            args=args,
            model=bundle.model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=bundle.collator,
            compute_metrics=bundle.compute_metrics if do_eval else None,
            processing_class=bundle.processor.feature_extractor,
            callbacks=[callback],
        )
    else:
        from transformers import Trainer, TrainingArguments

        args = TrainingArguments(**common, **eval_args, **best_args)
        trainer = Trainer(
            args=args,
            model=bundle.model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=bundle.collator,
            compute_metrics=bundle.compute_metrics if do_eval else None,
            processing_class=bundle.processor,
            callbacks=[callback],
        )

    print("[train] start", flush=True)
    trainer.train()

    print("[train] saving final model", flush=True)
    final_dir = run_dir / "model"
    trainer.save_model(str(final_dir))
    bundle.processor.save_pretrained(str(final_dir))
    print(f"[train] done -> {final_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    sys.exit(main(args.config))
