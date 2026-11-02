from __future__ import annotations
import platform
import time
from pathlib import Path
from .audio import load_audio
from .diarization import LocalDiarizer, align_segments
from .output import export_result, write_json


def run_offline(path, output, asr, diarizer=None):
    started = time.perf_counter()
    samples = load_audio(path)
    result = asr.transcribe(samples)
    result.update(source=str(Path(path).resolve()), mode="offline", platform=platform.platform())
    write_json(Path(output) / "asr_raw.json", result)
    if diarizer:
        diarization = diarizer.process(samples)
        write_json(Path(output) / "diarization_raw.json", diarization)
        result["segments"], result["anomalies"] = align_segments(result["segments"], diarization["tracks"])
        result["diarization"] = {k: v for k, v in diarization.items() if k != "tracks"}
    else:
        result["anomalies"] = []
    result["pipeline_elapsed_s"] = round(time.perf_counter() - started, 4)
    result["pipeline_rtf"] = round(result["pipeline_elapsed_s"] / result["duration_s"], 4)
    result["measurement"] = "ASR elapsed excludes model loading and decode; pipeline includes decode+ASR+diarization, excludes model loading"
    export_result(output, result)
    return result
