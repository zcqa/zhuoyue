"""Local meeting transcription experiments; all paths default to the project."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
for key, directory in {
    "HF_HOME": ".cache/huggingface", "XDG_CACHE_HOME": ".cache",
    "TORCH_HOME": ".cache/torch", "PIP_CACHE_DIR": ".cache/pip",
}.items():
    os.environ.setdefault(key, str(ROOT / directory))

__version__ = "1.0.0"
