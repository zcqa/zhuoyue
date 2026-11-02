from __future__ import annotations
from collections import defaultdict
import importlib.metadata
import time
from . import ROOT


class LocalDiarizer:
    """Pyannote segmentation ONNX + Chinese 3D-Speaker embeddings + clustering."""
    def __init__(self, num_speakers=-1, threshold=0.5, threads=4):
        import sherpa_onnx as so
        segmentation = ROOT / "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        embedding = ROOT / "models/speaker-embedding.onnx"
        for path in (segmentation, embedding):
            if not path.is_file():
                raise FileNotFoundError(f"模型不存在：{path}；请先运行 scripts/prepare_assets.py")
        config = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                num_threads=threads, provider="cpu"),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=threads, provider="cpu"),
            clustering=so.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.5,
        )
        if not config.validate():
            raise ValueError("说话人模型配置无效")
        self.pipeline = so.OfflineSpeakerDiarization(config)
        self.metadata = {"engine": "sherpa-onnx", "version": importlib.metadata.version("sherpa-onnx"),
                         "segmentation": segmentation.name, "embedding": embedding.name,
                         "num_speakers_hint": num_speakers, "cluster_threshold": threshold,
                         "identity": "anonymous_relative_ids", "mode": "offline"}

    def process(self, samples):
        t0 = time.perf_counter()
        tracks = self.pipeline.process(samples).sort_by_start_time()
        # Renumber by first appearance for human-readable stable IDs within this file only.
        ids, rows = {}, []
        for t in tracks:
            ids.setdefault(t.speaker, f"SPEAKER_{len(ids):02}")
            rows.append({"start": round(t.start, 3), "end": round(t.end, 3), "speaker_id": ids[t.speaker]})
        return {"tracks": rows, "elapsed_s": round(time.perf_counter() - t0, 4), "engine": self.metadata}


def merged_length(intervals):
    total, left, right = 0.0, None, None
    for a, b in sorted(intervals):
        if left is None:
            left, right = a, b
        elif a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + (right - left if left is not None else 0)


def assign_word(word, tracks, min_coverage=0.5, ambiguity=0.2):
    a, b = word["start"], word["end"]
    if b <= a:
        return None, "zero_duration_word", 0.0
    intervals = defaultdict(list)
    for t in tracks:
        left, right = max(a, t["start"]), min(b, t["end"])
        if right > left:
            intervals[t["speaker_id"]].append((left, right))
    scores = sorted(((merged_length(v) / (b - a), k) for k, v in intervals.items()), reverse=True)
    if not scores or scores[0][0] < min_coverage:
        return None, "insufficient_speaker_coverage", scores[0][0] if scores else 0.0
    if len(scores) > 1 and scores[1][0] >= ambiguity:
        return None, "overlap_or_speaker_boundary", scores[0][0]
    return scores[0][1], None, min(scores[0][0], 1.0)


def align_segments(segments, tracks):
    """Split at word-level speaker changes. Never invent a word boundary or a name."""
    output, anomalies = [], []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            row = dict(seg, speaker_id=None)
            output.append(row)
            anomalies.append({"type": "no_word_timestamps", "start": seg["start"], "end": seg["end"]})
            continue
        group = None
        for original in words:
            word = dict(original)
            who, reason, coverage = assign_word(word, tracks)
            word.update(speaker_id=who, speaker_coverage=round(coverage, 3))
            if reason:
                anomalies.append({"type": reason, "start": word["start"], "end": word["end"], "text": word["text"]})
            if group is None or group["speaker_id"] != who or word["start"] - group["end"] > 1.0:
                group = {"start": word["start"], "end": word["end"], "text": word["text"],
                         "speaker_id": who, "words": [word], "source_segment_id": seg["segment_id"]}
                output.append(group)
            else:
                group["end"] = word["end"]
                group["text"] += word["text"]
                group["words"].append(word)
    for i, row in enumerate(output, 1):
        row["segment_id"] = f"s{i:05}"
        row["text"] = row["text"].strip()
    return output, anomalies
