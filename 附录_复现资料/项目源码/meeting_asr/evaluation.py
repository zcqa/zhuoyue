from __future__ import annotations
from collections import Counter
import unicodedata
import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_text(text):
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith(("P", "S")))


def character_error(reference, hypothesis):
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return {"cer": None, "reference_characters": 0, "error": "empty_reference"}
    matrix = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    matrix[:, 0] = np.arange(len(ref) + 1)
    matrix[0, :] = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        for j, h in enumerate(hyp, 1):
            matrix[i, j] = min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + (r != h))
    counts = Counter()
    examples = []
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and matrix[i, j] == matrix[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                counts["substitutions"] += 1
                examples.append({"type": "substitution", "reference_index": i - 1, "hypothesis_index": j - 1,
                                 "reference": ref[max(0, i - 5):i + 4], "hypothesis": hyp[max(0, j - 5):j + 4]})
            i, j = i - 1, j - 1
        elif i and matrix[i, j] == matrix[i - 1, j] + 1:
            counts["deletions"] += 1
            examples.append({"type": "deletion", "reference_index": i - 1, "reference": ref[max(0, i - 5):i + 4]})
            i -= 1
        else:
            counts["insertions"] += 1
            examples.append({"type": "insertion", "hypothesis_index": j - 1, "hypothesis": hyp[max(0, j - 5):j + 4]})
            j -= 1
    return {"cer": round(int(matrix[-1, -1]) / len(ref), 5), "reference_characters": len(ref),
            "hypothesis_characters": len(hyp), "substitutions": counts["substitutions"],
            "deletions": counts["deletions"], "insertions": counts["insertions"],
            "normalization": "NFKC; strip punctuation/symbols/whitespace; lowercase; digits not expanded",
            "error_examples": list(reversed(examples))}


def evaluate_speakers(segments, reference_turns):
    words = [dict(w, speaker_id=w.get("speaker_id", row.get("speaker_id")))
             for row in segments for w in row.get("words", [])]
    labels = sorted({w["speaker_id"] for w in words if w["speaker_id"] is not None})
    refs = sorted({t["speaker_id"] for t in reference_turns})
    confusion = np.zeros((len(labels), len(refs)))
    matched = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        truths = {t["speaker_id"] for t in reference_turns if t["start"] <= mid < t["end"]}
        matched.append((w, truths))
        if len(truths) == 1 and w["speaker_id"] in labels:
            confusion[labels.index(w["speaker_id"]), refs.index(next(iter(truths)))] += max(.01, w["end"] - w["start"])
    mapping = {}
    if confusion.size:
        a, b = linear_sum_assignment(-confusion)
        mapping = {labels[i]: refs[j] for i, j in zip(a, b)}
    counts, errors = Counter(), []
    for w, truths in matched:
        if len(truths) > 1:
            kind = "reference_overlap_unresolved"
        elif not truths:
            kind = "word_in_source_gap"
        elif w["speaker_id"] is None:
            kind = "unknown_speaker"
        elif mapping.get(w["speaker_id"]) not in truths:
            kind = "speaker_confusion"
        else:
            kind = "correct"
        counts[kind] += 1
        if kind != "correct":
            errors.append({"type": kind, "start": w["start"], "end": w["end"], "text": w["text"],
                           "predicted": w["speaker_id"], "reference_speakers": sorted(truths)})
    denominator = counts["correct"] + counts["speaker_confusion"] + counts["unknown_speaker"]
    missed = []
    for t in reference_turns:
        if not any(t["start"] <= (w["start"] + w["end"]) / 2 < t["end"] for w in words):
            missed.append(dict(t, type="source_turn_without_recognized_words"))
    return {"mapping": mapping, "counts": dict(counts), "nonoverlap_word_count": denominator,
            "speaker_error_fraction": round((counts["speaker_confusion"] + counts["unknown_speaker"]) / denominator, 5) if denominator else None,
            "scope": "construction-reference word-midpoint diagnostic; NOT DER, NOT human verified speaker accuracy",
            "errors": errors, "missing_turn_candidates": missed, "predicted_speakers": len(labels), "reference_speakers": len(refs),
            "predicted_speakers_scope": "distinct labels on recognized words; diarization may have additional tracks without recognized text"}


def evaluate(result, reference):
    text = "".join(s["text"] for s in result["segments"])
    evaluation = {"case": reference["case"], "reference_type": reference["reference_type"]}
    evaluation["text"] = character_error(reference["reference_text"], text) if reference.get("cer_applicable", False) else {
        "cer": None, "reason": "No unique serial reference for overlapping audio, or no text reference supplied by corpus mirror"}
    if any(s.get("speaker_id") is not None for s in result["segments"]):
        evaluation["speakers"] = evaluate_speakers(result["segments"], reference["reference_turns"])
    evaluation["timestamp_issues"] = [{"segment_id": s["segment_id"], "start": s["start"], "end": s["end"]}
        for s in result["segments"] if s["start"] < 0 or s["end"] < s["start"] or s["end"] > reference["duration_s"] + .1]
    evaluation["timestamp_order_issues"] = [{"previous": a["segment_id"], "current": b["segment_id"]}
        for a, b in zip(result["segments"], result["segments"][1:]) if b["start"] < a["start"]]
    return evaluation
