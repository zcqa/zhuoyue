from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np
import soundfile as sf
from .audio import SAMPLE_RATE, load_audio, resample
from .output import export_result, write_json


class RealtimeTranscriber:
    """Bounded utterance windows, revisable partials, immutable final segments.

    The audio capture thread keeps running during inference. No input is silently
    discarded. Forced long-utterance cuts keep context, filtering committed words
    by absolute audio time, rather than deleting coincidentally repeated text.
    """
    def __init__(self, asr, emit, *, interval=1.5, silence=0.8, max_window=12.0,
                 overlap=0.6, energy_threshold=0.008, clock=time.perf_counter):
        if interval <= 0 or silence <= 0 or max_window < 2 or not 0 <= overlap < max_window:
            raise ValueError("无效的实时窗口参数")
        self.asr, self.emit, self.clock = asr, emit, clock
        self.interval, self.silence, self.max_window = interval, silence, max_window
        self.overlap, self.energy_threshold = overlap, energy_threshold
        self.origin = clock()
        self.pre = deque(maxlen=3)  # 300 ms pre-roll when fed with 100 ms frames.
        self.active = []
        self.start = self.end = self.last_voice = self.last_decode = 0.0
        self.capture_end = self.origin
        self.next_id = 1
        self.committed_until = -1.0
        self.previous_text = ""
        self.first_partial = None
        self.revisions = self.changes = 0
        self.finals, self.metrics, self.anomalies = [], [], []

    @property
    def segment_id(self):
        return f"live_{self.next_id:05}"

    def feed(self, frame, audio_end, captured_at):
        frame = np.asarray(frame, dtype=np.float32)
        frame_start = audio_end - len(frame) / SAMPLE_RATE
        voice = float(np.sqrt(np.mean(frame * frame))) >= self.energy_threshold
        self.end, self.capture_end = audio_end, captured_at
        if not self.active:
            self.pre.append((frame_start, frame.copy()))
            if not voice:
                return
            self.start = self.pre[0][0]
            self.active = [a for _, a in self.pre]
            self.pre.clear()
            self.last_voice = audio_end
            self.last_decode = self.start
        else:
            self.active.append(frame.copy())
            if voice:
                self.last_voice = audio_end
        if self.end - self.last_voice >= self.silence:
            self._recognize(final=True, reason="silence")
            self._reset()
        elif self.end - self.start >= self.max_window:
            self._recognize(final=True, reason="max_window")
            context = np.concatenate(self.active)[-round(self.overlap * SAMPLE_RATE):] if self.overlap else np.empty(0, np.float32)
            self._reset()
            # Only context; no new utterance until next voiced frame.
            if len(context):
                self.pre.append((self.end - len(context) / SAMPLE_RATE, context))
        elif self.end - self.last_decode >= self.interval:
            self._recognize(final=False, reason="interval")

    def _reset(self):
        self.active = []
        self.previous_text = ""
        self.first_partial = None
        self.revisions = self.changes = 0
        self.next_id += 1

    def _recognize(self, final, reason):
        audio = np.concatenate(self.active)
        result = self.asr.transcribe(audio, realtime=True)
        now = self.clock()
        words = []
        fallback = []
        for row in result["segments"]:
            if not row.get("words"):
                fallback.append(row["text"])
            for word in row.get("words", []):
                w = dict(word, start=round(self.start + word["start"], 3), end=round(self.start + word["end"], 3))
                if (w["start"] + w["end"]) / 2 > self.committed_until + 0.001:
                    words.append(w)
        text = "".join(w["text"] for w in words).strip()
        if fallback:
            self.anomalies.append({"type": "missing_word_timestamps", "segment_id": self.segment_id, "text": "".join(fallback)})
        self.last_decode = self.end
        if not text and not (final and self.previous_text):
            return
        if text != self.previous_text:
            if self.previous_text:
                self.changes += 1
                if not text.startswith(self.previous_text):
                    self.revisions += 1
        first = self.first_partial is None and not final and bool(text)
        if first:
            self.first_partial = now
        event = {
            "segment_id": self.segment_id, "state": "final" if final else "partial", "text": text,
            "start": words[0]["start"] if words else round(self.start, 3),
            "end": words[-1]["end"] if words else round(self.end, 3), "speaker_id": None,
            "audio_time": round(self.end, 4), "capture_end": round(self.capture_end - self.origin, 4),
            "emitted_at": round(now - self.origin, 4), "first_partial_at": round(self.first_partial - self.origin, 4) if self.first_partial is not None else None,
            "final_at": round(now - self.origin, 4) if final else None,
            "capture_to_result_s": round(max(0, now - self.capture_end), 4),
            "audio_start_to_result_s": round(max(0, now - (self.capture_end - (self.end - self.start))), 4),
            "last_voice_to_result_s": round(max(0, now - (self.capture_end - (self.end - self.last_voice))), 4),
            "revision_count": self.revisions, "text_change_count": self.changes,
            "inference_s": result["elapsed_s"], "reason": reason, "words": words,
        }
        if text != self.previous_text or final:
            self.emit(event)
        self.previous_text = text
        if final:
            if text:
                self.finals.append(event)
            else:
                self.anomalies.append({"type": "final_became_empty", "segment_id": self.segment_id})
            if words:
                self.committed_until = max(self.committed_until, words[-1]["end"])
            self.metrics.append(event)

    def finish(self):
        if self.active:
            self._recognize(final=True, reason="input_end")
            self._reset()


def run_realtime(asr, output, *, path=None, microphone=False, device=None, duration=120,
                 pace=True, interval=1.5, silence=0.8, max_window=12, energy_threshold=0.008):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = queue.Queue(maxsize=600)  # 60 s buffered, then stop with explicit error.
    stop = threading.Event()
    errors, events = [], []
    origin = time.perf_counter()
    log = (output / "events.jsonl").open("w", encoding="utf-8")

    def emit(event):
        events.append(event)
        log.write(json.dumps(event, ensure_ascii=False) + "\n")
        log.flush()
        print(f"{event['segment_id']} {event['state']:7} {event['text']}", flush=True)

    engine = RealtimeTranscriber(asr, emit, interval=interval, silence=silence,
                                 max_window=max_window, energy_threshold=energy_threshold)

    def enqueue(frame, audio_end, captured_at):
        try:
            frames.put_nowait((frame, audio_end, captured_at))
            return True
        except queue.Full:
            errors.append("capture_queue_overflow: 推理落后超过60秒，已停止输入；不能视为完整转写")
            stop.set()
            return False

    def file_producer():
        try:
            samples = load_audio(path)
            anchor = time.perf_counter()
            if duration > 0:
                samples = samples[:round(duration * SAMPLE_RATE)]
            for index in range(0, len(samples), 1600):
                if stop.is_set():
                    break
                frame = samples[index:index + 1600]
                end = (index + len(frame)) / SAMPLE_RATE
                if pace and stop.wait(max(0, anchor + end - time.perf_counter())):
                    break
                if not pace:
                    # Accelerated file tests may wait for the consumer. A real
                    # microphone cannot apply this backpressure without losing audio.
                    while not stop.is_set():
                        try:
                            frames.put((frame, end, time.perf_counter()), timeout=.1)
                            break
                        except queue.Full:
                            continue
                elif not enqueue(frame, end, time.perf_counter()):
                    break
        except Exception as exc:
            errors.append(f"file_input_error: {exc}")

    def mic_producer():
        import sounddevice as sd
        try:
            details = sd.query_devices(device, "input")
            rate = SAMPLE_RATE
            try:
                sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            except sd.PortAudioError:
                rate = int(details["default_samplerate"])
            count = 0

            def callback(indata, count_frames, timing, status):
                nonlocal count
                if status:
                    errors.append(f"microphone_status: {status}")
                pcm = resample(indata[:, 0], rate)
                count += len(pcm)
                enqueue(pcm, count / SAMPLE_RATE, time.perf_counter())

            with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                                blocksize=round(rate * .1), callback=callback):
                stop.wait(duration if duration > 0 else None)
        except Exception as exc:
            errors.append(f"microphone_input_error: {exc}")

    producer = threading.Thread(target=mic_producer if microphone else file_producer, daemon=True)
    producer.start()
    captured_duration = 0.0
    try:
        with sf.SoundFile(output / "captured.wav", "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as recording:
            while producer.is_alive() or not frames.empty():
                try:
                    frame, end, capture = frames.get(timeout=.1)
                except queue.Empty:
                    continue
                recording.write(frame)
                captured_duration = end
                engine.feed(frame, end, capture)
    except KeyboardInterrupt:
        stop.set()
        errors.append("user_stopped: 已保存已处理部分；尚在队列中的帧可能未识别")
    finally:
        stop.set()
        producer.join(timeout=3)
        try:
            engine.finish()
        finally:
            log.close()
    partials = [e for e in events if e["state"] == "partial"]
    firsts = {}
    for e in partials:
        firsts.setdefault(e["segment_id"], e)

    def distribution(items, field="capture_to_result_s"):
        values = [e[field] for e in items]
        return {"count": len(values), "mean_s": round(float(np.mean(values)), 4) if values else None,
                "p50_s": round(float(np.percentile(values, 50)), 4) if values else None,
                "p95_s": round(float(np.percentile(values, 95)), 4) if values else None}

    summary = {"mode": "microphone" if microphone else "paced_file_replay" if pace else "unpaced_file_replay",
               "duration_s": round(captured_duration, 4), "wall_elapsed_s": round(time.perf_counter() - origin, 4),
               "engine": asr.metadata, "partial_count": len(partials), "final_count": len(engine.finals),
               "first_partial_latency": distribution(firsts.values()), "final_latency": distribution(engine.metrics),
               "first_partial_from_audio_start": distribution(firsts.values(), "audio_start_to_result_s"),
               "final_from_last_voice": distribution(engine.metrics, "last_voice_to_result_s"),
               "revision_count": sum(e["revision_count"] for e in engine.metrics),
               "text_change_count": sum(e["text_change_count"] for e in engine.metrics),
               "parameters": {"interval_s": interval, "silence_s": silence, "max_window_s": max_window, "energy_threshold": energy_threshold},
               "latency_definition": "capture_to_result_s = event return minus most recent input frame enqueue time; includes inference and queue delay, excludes window accumulation. audio_start_to_result_s includes window accumulation. last_voice_to_result_s includes energy endpoint wait. First partial summarized once per segment. Mic capture clock is callback arrival, not hardware ADC timestamp.",
               "errors": errors, "anomalies": engine.anomalies,
               "status": "failed" if errors else "completed", "segments": engine.finals}
    export_result(output, summary)
    write_json(output / "summary.json", {k: v for k, v in summary.items() if k != "segments"})
    if errors:
        print("输入异常：" + "; ".join(errors), flush=True)
    return summary
