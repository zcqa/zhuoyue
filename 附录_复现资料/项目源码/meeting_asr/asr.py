from __future__ import annotations
import importlib.metadata
import os
import time
from pathlib import Path
from . import ROOT


class LocalASR:
    def __init__(self, model="small", device="cpu", compute_type=None, threads=8, language="zh"):
        self._dll_handles = []
        if device == "cuda" and os.name == "nt":
            # Current machine has CUDA 12.1/cuDNN 8; CT2 4.4 is pinned for that ABI.
            roots = [ROOT / "models/cuda/bin"]
            cuda = os.environ.get("CUDA_PATH")
            if cuda:
                roots.append(Path(cuda) / "bin")
            roots.append(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"))
            for directory in roots:
                if directory.is_dir():
                    self._dll_handles.append(os.add_dll_directory(str(directory)))
                    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        from faster_whisper import WhisperModel
        path = Path(model)
        if not path.is_dir():
            path = ROOT / "models" / f"faster-whisper-{model}"
        if not (path / "model.bin").exists():
            raise FileNotFoundError(f"本地模型不存在：{path}。请先运行 scripts/prepare_assets.py")
        self.language = None if language == "auto" else language
        self.metadata = {
            "engine": "faster-whisper", "version": importlib.metadata.version("faster-whisper"),
            "model": path.name, "device": device,
            "compute_type": compute_type or ("float16" if device == "cuda" else "int8"),
            "cpu_threads": threads,
            "ctranslate2_version": importlib.metadata.version("ctranslate2"),
        }
        started = time.perf_counter()
        self.model = WhisperModel(str(path), device=device, compute_type=self.metadata["compute_type"],
                                  cpu_threads=threads, num_workers=1, local_files_only=True)
        self.metadata["load_elapsed_s"] = round(time.perf_counter() - started, 4)

    def transcribe(self, audio, *, realtime=False):
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio, language=self.language, vad_filter=True, word_timestamps=True,
            beam_size=1 if realtime else 5, condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
        )
        rows = []
        for i, s in enumerate(segments, 1):
            rows.append({
                "segment_id": f"s{i:05}", "start": round(s.start, 3), "end": round(s.end, 3),
                "text": s.text.strip(), "speaker_id": None,
                "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word,
                           "probability": round(w.probability, 4)} for w in (s.words or [])],
                "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            })
        elapsed = time.perf_counter() - started  # Consume the lazy generator before stopping.
        return {"duration_s": round(len(audio) / 16000, 4), "elapsed_s": round(elapsed, 4),
                "rtf": round(elapsed / (len(audio) / 16000), 4) if len(audio) else None,
                "language": info.language, "language_probability": info.language_probability,
                "engine": self.metadata, "segments": rows}
