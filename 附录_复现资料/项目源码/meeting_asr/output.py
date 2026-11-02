from __future__ import annotations
import json
from pathlib import Path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp(seconds, separator=","):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{ms:03}"


def export_result(directory, result):
    directory = Path(directory)
    write_json(directory / "result.json", result)
    text = []
    subtitles = []
    for i, row in enumerate(result["segments"], 1):
        who = row.get("speaker_id") or "UNKNOWN"
        start, end = row.get("start"), row.get("end")
        span = f"{start:.2f}–{end:.2f}" if start is not None and end is not None else "时间未知"
        text.append(f"[{span}] [{who}] {row['text']}")
        if start is not None and end is not None:
            subtitles.append(f"{len(subtitles) + 1}\n{timestamp(start)} --> {timestamp(end)}\n[{who}] {row['text']}\n")
    (directory / "transcript.txt").write_text("\n".join(text), encoding="utf-8")
    (directory / "transcript.srt").write_text("\n".join(subtitles), encoding="utf-8")
