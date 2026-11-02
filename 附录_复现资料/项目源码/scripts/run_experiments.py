from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr.asr import LocalASR
from meeting_asr.diarization import LocalDiarizer
from meeting_asr.offline import run_offline
from meeting_asr.evaluation import evaluate
from meeting_asr.output import read_json, write_json
from meeting_asr.realtime import run_realtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--skip-realtime", action="store_true")
    args = p.parse_args()
    output = ROOT / "outputs/experiments"
    output.mkdir(parents=True, exist_ok=True)
    asr = LocalASR(device=args.device)
    # Automatic speaker counts; no use of ground-truth speaker count during inference.
    diarizer = LocalDiarizer(threshold=.8)
    cases = read_json(ROOT / "data/samples/cases.json")
    rows = []
    for case in cases:
        if case["case"] == "E":
            continue
        name = case["case"]
        print(f"Offline case {name}", flush=True)
        result = run_offline(ROOT / case["path"], output / name, asr, diarizer if name in ("C", "D") else None)
        ev = evaluate(result, case)
        write_json(output / name / "evaluation.json", ev)
        rows.append({"case": name, "duration_s": result["duration_s"], "asr_elapsed_s": result["elapsed_s"],
                     "rtf": result["rtf"], "pipeline_elapsed_s": result["pipeline_elapsed_s"],
                     "cer": ev["text"]["cer"], "speaker_diagnostic": ev.get("speakers", {})})
    cpu_result = run_offline(ROOT / cases[0]["path"], output / "A_cpu", LocalASR(device="cpu"))
    write_json(output / "A_cpu/evaluation.json", evaluate(cpu_result, cases[0]))
    natural_path = ROOT / "data/samples/F_natural_meeting.wav"
    if natural_path.exists():
        print("Natural meeting F", flush=True)
        natural = run_offline(natural_path, output / "F", asr, diarizer)
        write_json(output / "F/evaluation.json", evaluate(natural, read_json(natural_path.with_suffix(".reference.json"))))
    if not args.skip_realtime:
        case = next(c for c in cases if c["case"] == "E")
        print("Realtime case E (paced recording replay, about 2 minutes)", flush=True)
        live = run_realtime(asr, output / "E", path=ROOT / case["path"], duration=0, pace=True)
        write_json(output / "E/evaluation.json", evaluate(live, case))
    handoff = ROOT / "deliverables"
    handoff.mkdir(exist_ok=True)
    shutil.copy2(output / "C/result.json", handoff / "02_真实转写样例.json")
    if (output / "E/events.jsonl").exists():
        shutil.copy2(output / "E/events.jsonl", handoff / "06_实时文字更新记录.jsonl")
    metadata = {"completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "platform": platform.platform(), "python": platform.python_version(), "device": args.device,
                "asr": asr.metadata, "diarization": diarizer.metadata,
                "packages": {n: importlib.metadata.version(n) for n in ["faster-whisper", "ctranslate2", "sherpa-onnx", "sounddevice", "onnxruntime", "numpy"]},
                "results": rows, "cpu_baseline": {"elapsed_s": cpu_result["elapsed_s"], "rtf": cpu_result["rtf"]},
                "cloud_status": "not_run_no_api_key", "microphone_status": "separate_device_probe_required"}
    write_json(output / "summary.json", metadata)
    from build_report import build_report
    build_report()
    print(f"Experiment summary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
