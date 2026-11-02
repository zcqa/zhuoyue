"""Deepgram batch/streaming adapters. No network call without a configured key."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlencode
import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv
from . import ROOT
from .audio import SAMPLE_RATE, load_audio
from .output import export_result, write_json

BATCH_URL = "https://api.deepgram.com/v1/listen"
STREAM_URL = "wss://api.deepgram.com/v1/listen"


def key_or_error():
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise ValueError("云端密钥留空：未发送录音。请在 .env 中填写 DEEPGRAM_API_KEY 后重试。")
    return key


def normalize_words(words):
    result = []
    for w in words:
        label = w.get("speaker")
        result.append({"start": w.get("start"), "end": w.get("end"),
                       "text": w.get("punctuated_word", w.get("word", "")),
                       "speaker_id": f"SPEAKER_{label:02}" if isinstance(label, int) and label >= 0 else None,
                       "confidence": w.get("confidence"), "speaker_confidence": w.get("speaker_confidence")})
    return result


def group_words(words, fallback_text=""):
    segments = []
    current = None
    for w in words:
        boundary = current is None or w["speaker_id"] != current["speaker_id"]
        if current is not None and w["start"] is not None and current["end"] is not None:
            boundary |= w["start"] - current["end"] > 1
            if current["start"] is not None:
                boundary |= w["start"] - current["start"] > 10
        if boundary:
            current = {"segment_id": f"s{len(segments) + 1:05}", "start": w["start"], "end": w["end"],
                       "speaker_id": w["speaker_id"], "text": w["text"], "words": [w]}
            segments.append(current)
        else:
            current["end"] = w["end"]
            current["text"] += w["text"]
            current["words"].append(w)
    if not words and fallback_text:
        segments.append({"segment_id": "s00001", "start": None, "end": None,
                         "speaker_id": None, "text": fallback_text, "words": []})
    return segments


def normalize_batch(raw):
    alt = raw["results"]["channels"][0]["alternatives"][0]
    words = normalize_words(alt.get("words", []))
    anomalies = []
    for w in words:
        for field in ("start", "end", "speaker_id"):
            if w[field] is None:
                anomalies.append({"type": f"missing_{field}", "text": w["text"]})
    if not words:
        anomalies.append({"type": "missing_words", "message": "原始文字保留，时间和说话人均未知"})
    return {"segments": group_words(words, alt.get("transcript", "")), "anomalies": anomalies,
            "duration_s": raw.get("metadata", {}).get("duration"), "engine": {"name": "deepgram", "model": "nova-3"}}


def run_batch(path, output):
    output = Path(output)
    params = {"model": "nova-3", "language": "zh-CN", "smart_format": "true", "diarize_model": "latest"}
    write_json(output / "request_plan.json", {"url": BATCH_URL, "params": params, "source": str(Path(path).resolve()), "key": "REDACTED"})
    key = key_or_error()
    samples = load_audio(path)
    wav = io.BytesIO()
    sf.write(wav, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    started = time.perf_counter()
    response = requests.post(BATCH_URL, params=params,
                             headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
                             data=wav.getvalue(), timeout=(15, 180))
    # Preserve the provider response on failure as well, but never authorization headers.
    try:
        raw = response.json()
    except ValueError:
        raw = {"http_status": response.status_code, "body": response.text[:20000]}
    write_json(output / "cloud_raw.json", raw)
    response.raise_for_status()
    result = normalize_batch(raw)
    elapsed = time.perf_counter() - started
    result.update(elapsed_s=round(elapsed, 4), mode="cloud_batch", source=str(Path(path).resolve()))
    result["rtf"] = round(elapsed / result["duration_s"], 4) if result["duration_s"] else None
    export_result(output, result)
    return result


class CloudEventAdapter:
    def __init__(self):
        self.ids, self.history = {}, {}

    def normalize(self, raw, elapsed, sent_audio_time):
        if raw.get("type") != "Results":
            return None
        alternatives = raw.get("channel", {}).get("alternatives", [])
        if not alternatives:
            return None
        alt = alternatives[0]
        start = raw.get("start")
        # Provider start is stable for revisions; duration/text may change.
        identity = str(start) if start is not None else f"unknown_{len(self.ids)}"
        self.ids.setdefault(identity, f"cloud_{len(self.ids) + 1:05}")
        sid = self.ids[identity]
        previous = self.history.get(sid, "")
        text = alt.get("transcript", "")
        event = {"segment_id": sid, "state": "final" if raw.get("is_final") is True else "partial",
                 "text": text, "start": start,
                 "end": start + raw["duration"] if start is not None and raw.get("duration") is not None else None,
                 "emitted_at": round(elapsed, 4), "audio_time": round(sent_audio_time, 4),
                 "is_final": raw.get("is_final"), "speech_final": raw.get("speech_final"),
                 "words": normalize_words(alt.get("words", [])), "speaker_id": None,
                 "revised": bool(previous and previous != text)}
        self.history[sid] = text
        return event


def run_cloud_stream(path, output, *, microphone=False, device=None, duration=120):
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosedOK
    output = Path(output)
    params = {"model": "nova-3", "language": "zh-CN", "smart_format": "true", "diarize_model": "latest",
              "encoding": "linear16", "sample_rate": SAMPLE_RATE, "channels": 1,
              "interim_results": "true", "endpointing": 300}
    url = STREAM_URL + "?" + urlencode(params)
    write_json(output / "request_plan.json", {"url": url, "source": "microphone" if microphone else str(path), "key": "REDACTED"})
    key = key_or_error()
    stop = threading.Event()
    done = threading.Event()
    errors, finals = [], {}
    sent = [0.0]
    adapter = CloudEventAdapter()
    start = time.perf_counter()
    with connect(url, additional_headers={"Authorization": f"Token {key}"}, open_timeout=15, close_timeout=5) as ws:
        def sender():
            try:
                if microphone:
                    import sounddevice as sd
                    with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=1600) as stream:
                        while not stop.is_set() and (duration <= 0 or sent[0] < duration):
                            data, overflow = stream.read(1600)
                            if overflow:
                                errors.append("cloud_microphone_overflow")
                            ws.send(data.tobytes())
                            sent[0] += len(data) / SAMPLE_RATE
                else:
                    audio = load_audio(path)
                    if duration > 0:
                        audio = audio[:round(duration * SAMPLE_RATE)]
                    anchor = time.perf_counter()
                    for idx in range(0, len(audio), 1600):
                        if stop.is_set():
                            break
                        data = audio[idx:idx + 1600]
                        end = (idx + len(data)) / SAMPLE_RATE
                        if stop.wait(max(0, anchor + end - time.perf_counter())):
                            break
                        ws.send((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
                        sent[0] = end
                ws.send(json.dumps({"type": "CloseStream"}))
            except Exception as exc:
                errors.append(f"cloud_send_error: {type(exc).__name__}: {exc}")
            finally:
                done.set()

        worker = threading.Thread(target=sender, daemon=True)
        worker.start()
        finished_at = None
        try:
            with (output / "cloud_raw.jsonl").open("w", encoding="utf-8") as raw_log, (output / "events.jsonl").open("w", encoding="utf-8") as log:
                while True:
                    if done.is_set() and finished_at is None:
                        finished_at = time.perf_counter()
                    try:
                        message = ws.recv(timeout=1)
                    except TimeoutError:
                        if finished_at is not None and time.perf_counter() - finished_at > 30:
                            errors.append("cloud_finalize_timeout")
                            break
                        continue
                    except ConnectionClosedOK:
                        break
                    raw = json.loads(message)
                    raw_log.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    raw_log.flush()
                    if raw.get("type") == "Error":
                        errors.append("provider_error: " + str(raw.get("description", raw)))
                        break
                    event = adapter.normalize(raw, time.perf_counter() - start, sent[0])
                    if event is None:
                        continue
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    log.flush()
                    print(f"{event['segment_id']} {event['state']} {event['text']}", flush=True)
                    if event["state"] == "final" and event["text"]:
                        finals[event["segment_id"]] = event
        except KeyboardInterrupt:
            errors.append("user_stopped: 已保留已收到的事件与 final 片段")
        except Exception as exc:
            errors.append(f"cloud_receive_error: {type(exc).__name__}: {exc}")
        finally:
            stop.set()
            worker.join(timeout=5)
    rows = []
    for event in finals.values():
        rows.extend(group_words(event["words"], event["text"]))
    for i, row in enumerate(rows, 1):
        row["segment_id"] = f"s{i:05}"
    result = {"mode": "cloud_stream_microphone" if microphone else "cloud_stream_paced_replay",
              "duration_s": sent[0], "wall_elapsed_s": round(time.perf_counter() - start, 4),
              "segments": rows, "errors": errors, "status": "failed" if errors else "completed"}
    export_result(output, result)
    return result
