"""Central config: paths and defaults.

Precedence: environment variable > config.toml > built-in default.
config.toml lives at the project root (override path with STT_CONFIG).
"""
import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 and older
    try:
        import tomli as tomllib  # pip install tomli
    except ModuleNotFoundError:
        tomllib = None

# Project root = stt_tuner/  (two levels up from this file: app/ -> backend/ -> root)
ROOT = Path(os.environ.get("STT_ROOT", Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Load config.toml
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(os.environ.get("STT_CONFIG", ROOT / "config.toml"))


def _parse_simple_toml(text: str) -> dict:
    """Minimal TOML reader (sections + str/int/float/bool scalars). Used as a
    dependency-free fallback when neither tomllib nor tomli is available."""
    data: dict = {}
    section = data
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = data.setdefault(line[1:-1].strip(), {})
            continue
        if "=" not in line:
            continue
        key, val = (s.strip() for s in line.split("=", 1))
        if val and val[0] in "\"'":
            section[key] = val.strip("\"'")
        elif val in ("true", "false"):
            section[key] = val == "true"
        else:
            try:
                section[key] = int(val)
            except ValueError:
                try:
                    section[key] = float(val)
                except ValueError:
                    section[key] = val
    return data


def _load_toml() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    if tomllib is not None:
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return _parse_simple_toml(_CONFIG_PATH.read_text(encoding="utf-8"))


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
