"""Collect AISHELL-1 audio with corpus references; construct reproducible A-E tests.

Only the small audio.path/transcription Parquet columns are fetched (HTTP range),
not the 378 MB audio column. Individual WAVs come from HF's dataset viewer.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr.audio import load_audio, SAMPLE_RATE
from meeting_asr.output import write_json
from prepare_assets import download, sha256

REVISION = "692ef58020d79b21f54eb25b15a4813d4f9650d7"
PARQUET = f"https://huggingface.co/datasets/carlot/AIShell/resolve/{REVISION}/data/test-00000-of-00003-25094929c3ce9a28.parquet"


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos, self.size = 0, 378238016
        self.session = requests.Session()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, count=-1):
        count = min(count if count >= 0 else self.size, self.size - self.pos)
        if count <= 0:
            return b""
        r = self.session.get(PARQUET, headers={"Range": f"bytes={self.pos}-{self.pos + count - 1}"}, timeout=90)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("服务器不支持 HTTP Range；拒绝意外下载完整数据集")
        self.pos += len(r.content)
        return r.content


def collect():
    cache = ROOT / "data/aishell_rows.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        rows = pq.ParquetFile(RangeFile()).read(columns=["audio.path", "transcription"]).to_pylist()
        write_json(cache, rows)
    groups = {}
    for index, row in enumerate(rows):
        speaker = re.search(r"S\d+", row["audio"]["path"])[0]
        groups.setdefault(speaker, []).append((index, row))
    corpus = []
    # Distinct speaker IDs are taken from original corpus filenames, not guessed.
    for speaker, members in list(groups.items())[:3]:
        offset = members[0][0]
        params = urllib.parse.urlencode({"dataset": "carlot/AIShell", "config": "default", "split": "test", "offset": offset, "length": 18})
        viewer = requests.get("https://datasets-server.huggingface.co/rows?" + params, timeout=60)
        viewer.raise_for_status()
        for item in viewer.json()["rows"]:
            idx = item["row_idx"]
            original = rows[idx]
            path = ROOT / "data/raw/aishell" / original["audio"]["path"]
            url = item["row"]["audio"][0]["src"]
            download(url, path)
            corpus.append({"path": str(path.relative_to(ROOT)), "speaker_id": speaker,
                           "reference": "".join(original["transcription"].split()), "dataset_row": idx,
                           "source": "https://datasets-server.huggingface.co/rows?" + params,
                           "revision": REVISION, "sha256": sha256(path),
                           "duration_s": len(load_audio(path)) / SAMPLE_RATE})
        print(f"Collected speaker {speaker}", flush=True)
    write_json(ROOT / "data/corpus_manifest.json", {"dataset": "AISHELL-1", "license": "Apache-2.0",
               "original_source": "https://www.openslr.org/33/", "mirror": "https://huggingface.co/datasets/carlot/AIShell",
               "reference_type": "corpus_human_transcription_not_new_manual_review", "clips": corpus})
    build(corpus)


def build(corpus):
    groups = {}
    for clip in corpus:
        groups.setdefault(clip["speaker_id"], []).append(clip)
    speakers = list(groups)
    rng = np.random.default_rng(20261102)
    cases = []
    directory = ROOT / "data/samples"
    directory.mkdir(parents=True, exist_ok=True)

    def assemble(case, selected_speakers, target, overlap=False):
        # Concatenated real human utterances, explicitly NOT a natural meeting.
        canvas = np.zeros(round((target + 20) * SAMPLE_RATE), np.float32)
        turns, cursor, i = [], .3, 0
        while cursor < target:
            who = selected_speakers[i % len(selected_speakers)]
            clip = groups[who][(i // len(selected_speakers)) % len(groups[who])]
            audio = load_audio(ROOT / clip["path"])
            start = cursor
            if overlap and i % 4 == 3:
                start = max(0, cursor - 1.8)
            end = start + len(audio) / SAMPLE_RATE
            a, b = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
            canvas[a:a + len(audio)] += audio
            turns.append({"start": round(start, 4), "end": round(end, 4), "speaker_id": who,
                          "text": clip["reference"], "source_clip": clip["path"]})
            cursor = end + .3
            i += 1
        audio = canvas[:round((cursor + .5) * SAMPLE_RATE)]
        audio *= .9 / max(.9, float(np.max(np.abs(audio))))
        sf.write(directory / f"{case}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
        metadata = {"case": case, "path": f"data/samples/{case}.wav", "duration_s": len(audio) / SAMPLE_RATE,
                    "speakers": selected_speakers, "constructed": True,
                    "construction": "real human AISHELL utterances concatenated" + (" with known overlaps" if overlap else " with 0.3s gaps"),
                    "reference_type": "original_corpus_transcripts", "reference_turns": turns,
                    "reference_text": "".join(t["text"] for t in turns), "cer_applicable": not overlap,
                    "speaker_reference_note": "clip-level construction bounds include source leading/trailing silence; not frame-level human speech annotation"}
        write_json(directory / f"{case}.reference.json", metadata)
        cases.append(metadata)
        return audio, metadata

    clean, a = assemble("A", speakers[:1], 60)
    # Deterministic 10 dB white noise + short room-like echo. A simulated far/noisy input.
    impulse = np.zeros(4801, np.float32)
    impulse[0], impulse[1600], impulse[4800] = 1, .25, .12
    reverbed = fftconvolve(clean, impulse)[:len(clean)]
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(reverbed ** 2) / (10 ** (10 / 10) * np.mean(noise ** 2)))
    noisy = reverbed + noise
    noisy *= .95 / max(.95, float(np.max(np.abs(noisy))))
    sf.write(directory / "B.wav", noisy, SAMPLE_RATE, subtype="PCM_16")
    b = dict(a, case="B", path="data/samples/B.wav", construction="A + deterministic 10dB white noise and 100/300ms echoes", noise_seed=20261102)
    write_json(directory / "B.reference.json", b)
    cases.append(b)
    assemble("C", speakers[:2], 120)
    assemble("D", speakers[:3], 120, overlap=True)
    # E tests continuous arrival at wall-clock rate using a reproducible recording.
    e_audio = load_audio(directory / "C.wav")
    sf.write(directory / "E.wav", e_audio, SAMPLE_RATE, subtype="PCM_16")
    e = dict(cases[2], case="E", path="data/samples/E.wav", construction="C replayed at wall-clock speed; not a microphone recording")
    write_json(directory / "E.reference.json", e)
    cases.append(e)
    write_json(directory / "cases.json", cases)
    print("Prepared cases: " + ", ".join(f"{c['case']}={c['duration_s']:.1f}s" for c in cases), flush=True)


if __name__ == "__main__":
    collect()
