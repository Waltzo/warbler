"""Central config: paths and defaults.

Precedence: environment variable > config.toml > built-in default.
config.toml lives at the project root (override path with STT_CONFIG).
"""
import os
import tomllib
from pathlib import Path

# Project root = stt_tuner/  (two levels up from this file: app/ -> backend/ -> root)
ROOT = Path(os.environ.get("STT_ROOT", Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Load config.toml
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(os.environ.get("STT_CONFIG", ROOT / "config.toml"))


def _load_toml() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


_TOML = _load_toml()
_SERVER = _TOML.get("server", {})
_GPU = _TOML.get("gpu", {})

DATASETS_DIR = Path(os.environ.get("STT_DATASETS_DIR", ROOT / "datasets"))
RUNS_DIR = Path(os.environ.get("STT_RUNS_DIR", ROOT / "runs"))

# Python interpreter used to launch the training subprocess.
PYTHON_BIN = os.environ.get("STT_PYTHON", os.sys.executable)

# Server bind (env > config.toml > default).
SERVER_HOST = os.environ.get("STT_HOST", _SERVER.get("host", "0.0.0.0"))
SERVER_PORT = int(os.environ.get("STT_PORT", _SERVER.get("port", 8000)))

# Default GPU index to suggest in the UI (env > config.toml > default).
DEFAULT_GPU_INDEX = int(os.environ.get("STT_DEFAULT_GPU", _GPU.get("default_index", 0)))

# Preview rows returned when registering a dataset.
PREVIEW_ROWS = 10

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Suggested base models per model type (shown in NewJob form).
SUGGESTED_MODELS = {
    "whisper": [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
    ],
    "wav2vec2": [
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-xlsr-53",
        "facebook/mms-1b-all",
    ],
}
