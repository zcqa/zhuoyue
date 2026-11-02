from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16000


def load_audio(path: str | Path) -> np.ndarray:
    """PyAV handles wav/mp3/m4a/flac, stereo and resampling without shell FFmpeg."""
    data = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    if not len(data):
        raise ValueError("录音为空，无法转写")
    if not np.isfinite(data).all():
        raise ValueError("音频包含非有限值")
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(samples, rate):
    if rate == SAMPLE_RATE:
        return np.asarray(samples, dtype=np.float32)
    divisor = math.gcd(int(rate), SAMPLE_RATE)
    return resample_poly(samples, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
