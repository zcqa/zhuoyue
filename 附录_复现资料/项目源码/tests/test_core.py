from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from meeting_asr.cloud import CloudEventAdapter, normalize_batch, run_batch
from meeting_asr.diarization import assign_word, align_segments
from meeting_asr.evaluation import character_error, evaluate_speakers
from meeting_asr.output import timestamp
from meeting_asr.realtime import RealtimeTranscriber


def test_word_alignment_splits_speaker_switch_and_keeps_ambiguity():
    tracks = [{"start": 0, "end": 1, "speaker_id": "A"}, {"start": 1, "end": 2, "speaker_id": "B"}]
    segment = {"segment_id": "old", "start": 0, "end": 2, "text": "甲乙丙", "words": [
        {"start": .1, "end": .5, "text": "甲"}, {"start": .8, "end": 1.2, "text": "乙"},
        {"start": 1.3, "end": 1.8, "text": "丙"}]}
    rows, errors = align_segments([segment], tracks)
    assert [r["speaker_id"] for r in rows] == ["A", None, "B"]
    assert "".join(r["text"] for r in rows) == "甲乙丙"
    assert errors[0]["type"] == "overlap_or_speaker_boundary"


def test_overlapping_tracks_never_force_a_speaker():
    word = {"start": .4, "end": .8}
    tracks = [{"start": 0, "end": 1, "speaker_id": "A"}, {"start": .2, "end": 1.2, "speaker_id": "B"}]
    assert assign_word(word, tracks)[0] is None
    assert assign_word({"start": 1, "end": 1}, tracks)[1] == "zero_duration_word"


def test_duplicate_same_speaker_intervals_do_not_inflate_coverage():
    w = {"start": 0, "end": 1}
    tracks = [{"start": 0, "end": .3, "speaker_id": "A"}] * 4
    assert assign_word(w, tracks)[0] is None
    assert assign_word(w, tracks)[2] == pytest.approx(.3)


def test_no_word_timestamps_does_not_fabricate_speaker():
    rows, errors = align_segments([{"segment_id": "s1", "start": 0, "end": 3, "text": "文字"}],
                                  [{"start": 0, "end": 3, "speaker_id": "A"}])
    assert rows[0]["speaker_id"] is None
    assert errors[0]["type"] == "no_word_timestamps"


def test_cer_normalization_and_errors():
    assert character_error("Ａ B，中文。", "ab中文")["cer"] == 0
    result = character_error("甲乙丙丁", "甲戊丁")
    assert result["cer"] == .5
    assert result["substitutions"] == 1 and result["deletions"] == 1
    assert character_error("", "x")["cer"] is None


def test_speaker_ids_are_mapped_instead_of_compared_as_names():
    rows = [{"speaker_id": "SPEAKER_01", "words": [{"text": "a", "start": .1, "end": .9}]},
            {"speaker_id": "SPEAKER_00", "words": [{"text": "b", "start": 1.1, "end": 1.9}]}]
    ref = [{"speaker_id": "original_A", "start": 0, "end": 1}, {"speaker_id": "original_B", "start": 1, "end": 2}]
    ev = evaluate_speakers(rows, ref)
    assert ev["speaker_error_fraction"] == 0
    assert ev["mapping"]["SPEAKER_01"] == "original_A"


def test_missing_cloud_fields_stay_unknown():
    raw = {"results": {"channels": [{"alternatives": [{"words": [{"word": "你好"}]}]}]}}
    result = normalize_batch(raw)
    assert result["segments"][0]["start"] is None
    assert result["segments"][0]["speaker_id"] is None
    assert len(result["anomalies"]) == 3
    raw["results"]["channels"][0]["alternatives"][0] = {"transcript": "只有文字"}
    assert normalize_batch(raw)["segments"][0]["text"] == "只有文字"


def test_stream_revisions_share_id_and_speech_final_does_not_finalize():
    adapter = CloudEventAdapter()
    def msg(text, final=False, speech=False):
        return {"type": "Results", "start": 0, "duration": 1, "is_final": final, "speech_final": speech,
                "channel": {"alternatives": [{"transcript": text}]}}
    a = adapter.normalize(msg("我们"), .5, .5)
    b = adapter.normalize(msg("我们后续", speech=True), .8, .8)
    c = adapter.normalize(msg("我们后续测试", final=True), 1, 1)
    assert a["segment_id"] == b["segment_id"] == c["segment_id"]
    assert b["state"] == "partial" and c["state"] == "final"


def test_blank_key_never_uploads(monkeypatch, tmp_path):
    import meeting_asr.cloud as cloud
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setattr(cloud, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(cloud.requests, "post", lambda *a, **kw: pytest.fail("Unexpected upload"))
    with pytest.raises(ValueError, match="密钥留空"):
        run_batch("not-even-an-audio-file.wav", tmp_path)
    assert (tmp_path / "request_plan.json").exists()


class FakeASR:
    metadata = {"engine": "unit-test-double"}
    def __init__(self): self.calls = 0
    def transcribe(self, audio, **kwargs):
        self.calls += 1
        text = "我们后续" if self.calls == 1 else "我们需要测试"
        return {"elapsed_s": .01, "segments": [{"text": text,
                "words": [{"text": text, "start": 0, "end": len(audio) / 16000 - .05}]}]}


def test_realtime_replaces_partial_and_freezes_final():
    events = []
    clock = [10.0]
    engine = RealtimeTranscriber(FakeASR(), events.append, interval=.2, silence=.2, clock=lambda: clock[0])
    for i in range(1, 7):
        clock[0] = 10 + i * .1
        engine.feed(np.full(1600, .1, np.float32), i * .1, clock[0])
    engine.finish()
    assert len({e["segment_id"] for e in events}) == 1
    assert events[-1]["state"] == "final"
    assert events[-1]["revision_count"] >= 1
    assert len(engine.finals) == 1
    saved = json.dumps(engine.finals)
    engine.finish()
    assert json.dumps(engine.finals) == saved


def test_realtime_silence_produces_no_hallucinated_events():
    events = []
    asr = FakeASR()
    engine = RealtimeTranscriber(asr, events.append)
    for i in range(20):
        engine.feed(np.zeros(1600, np.float32), (i + 1) * .1, engine.origin + (i + 1) * .1)
    engine.finish()
    assert not events and asr.calls == 0


def test_realtime_long_windows_retain_legitimate_repeated_words():
    class Repeater:
        def transcribe(self, audio, **kwargs):
            end = len(audio) / 16000
            return {"elapsed_s": .01, "segments": [{"words": [
                {"start": 0, "end": .2, "text": "你好"}, {"start": end - .2, "end": end, "text": "你好"}]}]}
    events = []
    e = RealtimeTranscriber(Repeater(), events.append, interval=1, max_window=2, overlap=.5)
    for i in range(35):
        e.feed(np.full(1600, .1, np.float32), (i + 1) * .1, e.origin + (i + 1) * .1)
    e.finish()
    assert len(e.finals) >= 2
    assert e.finals[0]["text"] == "你好你好"
    assert e.finals[1]["text"] == "你好"  # context word removed by time, new repetition retained.


def test_srt_timestamp_rounds_through_minute():
    assert timestamp(59.9996) == "00:01:00,000"
    assert timestamp(-1) == "00:00:00,000"


def test_accelerated_file_longer_than_queue_does_not_drop_audio(tmp_path):
    import soundfile as sf
    import time
    from meeting_asr.realtime import run_realtime
    class SlowFake(FakeASR):
        def transcribe(self, audio, **kw):
            time.sleep(.01)
            return super().transcribe(audio, **kw)
    path = tmp_path / "long.wav"
    sf.write(path, np.full(65 * 16000, .1, np.float32), 16000)
    result = run_realtime(SlowFake(), tmp_path / "out", path=path, duration=0, pace=False)
    assert result["status"] == "completed"
    assert result["duration_s"] == 65
