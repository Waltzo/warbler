"""Pydantic schemas for API requests/responses."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

ModelType = Literal["whisper", "wav2vec2"]


class TrainConfig(BaseModel):
    """Training job configuration. Serialized to runs/<job_id>/config.json
    and consumed by training/train.py."""
    name: str = Field(..., description="Human-readable job name")
    model_type: ModelType
    base_model: str = Field(..., description="HF model id, e.g. openai/whisper-small")
    dataset_id: str

    # PEFT / LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 8
    grad_accum: int = 1
    num_epochs: float = 3.0
    max_steps: int = -1  # -1 => use num_epochs
    eval_ratio: float = 0.1  # fraction of dataset held out for eval
    eval_steps: int = 50
    save_steps: int = 200
    warmup_steps: int = 50

    # Whisper-specific
    language: Optional[str] = None  # e.g. "korean"
    task: str = "transcribe"

    # Runtime
    gpu_index: int = 0
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"


class DatasetRegister(BaseModel):
    """Register an existing dataset already on the server filesystem."""
    dataset_id: str = Field(..., description="Unique id / folder name under datasets/")
    manifest_path: str = Field(..., description="Path to manifest .jsonl/.csv")
    audio_root: Optional[str] = Field(
        None, description="Base dir to resolve relative audio_path (defaults to manifest dir)"
    )
    audio_key: Optional[str] = Field(
        None, description="Custom column name for audio path (default: audio_path/audio/path)"
    )
    text_key: Optional[str] = Field(
        None, description="Custom column name for transcript (default: text/transcript/sentence)"
    )


class DatasetInfo(BaseModel):
    dataset_id: str
    manifest_path: str
    audio_root: str
    num_samples: int
    total_duration_sec: Optional[float] = None
    audio_key: Optional[str] = None
    text_key: Optional[str] = None
    preview: list[dict]


class JobStatus(BaseModel):
    job_id: str
    name: str
    state: Literal["pending", "running", "done", "failed", "stopped"]
    model_type: str
    base_model: str
    dataset_id: str
    pid: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    last_step: Optional[int] = None
    total_steps: Optional[int] = None
