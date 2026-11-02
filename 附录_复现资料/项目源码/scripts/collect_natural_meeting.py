"""Extract only 120 s of a real AISHELL-4 meeting using HTTP Range, preserving attribution."""
from __future__ import annotations
import io
import json
from pathlib import Path
import struct
import sys
import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr.output import write_json
from prepare_assets import sha256


def main():
    api = "https://datasets-server.huggingface.co/first-rows?dataset=argmaxinc%2Faishell-4&config=default&split=test"
    r = requests.get(api, timeout=60)
    r.raise_for_status()
    row = r.json()["rows"][0]["row"]
    starts, ends, ids = row["timestamps_start"], row["timestamps_end"], row["speakers"]
    turns = [{"start": a, "end": b, "speaker_id": s} for a, b, s in zip(starts, ends, ids)]
    # Select a turn-rich window after the introductory speaker self-tests.
    def score(start):
        ts = [t for t in turns if t["start"] < start + 120 and t["end"] > start]
        overlaps = sum(min(a["end"], b["end"]) > max(a["start"], b["start"])
                       for i, a in enumerate(ts) for b in ts[i + 1:])
        return len({t["speaker_id"] for t in ts}) * 10 + overlaps + len(ts) / 10
    start = max(range(90, int(max(ends)) - 120, 30), key=score)
    url = row["audio"][0]["src"]
    r = requests.get(url, headers={"Range": "bytes=0-65535"}, timeout=60)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError("Range unsupported")
    header = r.content
    info = sf.info(io.BytesIO(header))
    if info.subtype != "PCM_16" or info.channels != 1 or info.samplerate != 16000:
        raise RuntimeError(f"Unexpected upstream WAV format: {info}")
    pos = 12
    while header[pos:pos + 4] != b"data":
        size = struct.unpack_from("<I", header, pos + 4)[0]
        pos += 8 + size + size % 2
        if pos + 8 > len(header):
            raise RuntimeError("WAV data chunk not found")
    data_offset = pos + 8
    byte_start = data_offset + start * 32000
    byte_end = byte_start + 120 * 32000 - 1
    response = requests.get(url, headers={"Range": f"bytes={byte_start}-{byte_end}"}, timeout=90)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != 120 * 32000:
        raise RuntimeError("Incomplete audio range")
    audio = np.frombuffer(response.content, dtype="<i2").astype(np.float32) / 32768
    path = ROOT / "data/samples/F_natural_meeting.wav"
    sf.write(path, audio, 16000, subtype="PCM_16")
    reference = {"case": "F", "path": str(path.relative_to(ROOT)), "duration_s": 120,
                 "constructed": False, "dataset": "AISHELL-4", "source": "https://www.openslr.org/111/",
                 "mirror": "https://huggingface.co/datasets/argmaxinc/aishell-4",
                 "license": "CC BY-SA 4.0", "creator": "Beijing Shell Shell Technology Co., Ltd. (AISHELL-4 authors)",
                 "changes": "HF mirror mono 16k waveform; extracted continuous 120 s without mixing",
                 "source_row": 0, "source_revision": "0cf6e53877b50888a93cc835d8d6beb08d3fb917",
                 "original_offset_s": start, "sha256": sha256(path),
                 "reference_type": "speaker_time_annotations_from_corpus_mirror; no text reference in this mirror",
                 "reference_turns": [{"start": max(0, t["start"] - start), "end": min(120, t["end"] - start),
                                      "speaker_id": t["speaker_id"]} for t in turns if t["start"] < start + 120 and t["end"] > start]}
    write_json(path.with_suffix(".reference.json"), reference)
    print(f"Natural meeting: {start}–{start + 120}s; {len(reference['reference_turns'])} turns; "
          f"{len(set(t['speaker_id'] for t in reference['reference_turns']))} speakers", flush=True)


if __name__ == "__main__":
    main()
