"""Same-sample parameter exploration; not independent validation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr.audio import load_audio
from meeting_asr.output import read_json, write_json, export_result
from meeting_asr.diarization import LocalDiarizer, align_segments
from meeting_asr.evaluation import evaluate_speakers


def main():
    samples = load_audio(ROOT / "data/samples/F_natural_meeting.wav")
    raw = read_json(ROOT / "outputs/experiments/F/asr_raw.json")
    ref = read_json(ROOT / "data/samples/F_natural_meeting.reference.json")
    rows = []
    for name, count, threshold in [("F_t090", -1, .9), ("F_t095", -1, .95), ("F_known6", 6, .8)]:
        diarization = LocalDiarizer(count, threshold).process(samples)
        segments, anomalies = align_segments(raw["segments"], diarization["tracks"])
        ev = evaluate_speakers(segments, ref["reference_turns"])
        result = dict(raw, segments=segments, anomalies=anomalies,
                      diarization={k: v for k, v in diarization.items() if k != "tracks"})
        # ASR was reused; do not fabricate a combined timing measurement.
        result["measurement"] = "ASR reused from F; diarization timed separately; no combined pipeline benchmark"
        output = ROOT / "outputs/experiments" / name
        export_result(output, result)
        write_json(output / "diarization_raw.json", diarization)
        write_json(output / "evaluation.json", {"case": name, "speakers": ev,
                   "text": {"cer": None, "reason": "mirror has no text reference"}})
        row = {"case": name, "known_speakers": count, "threshold": threshold,
               "track_speakers": len({t["speaker_id"] for t in diarization["tracks"]}),
               "predicted_speakers": ev["predicted_speakers"], "word_speaker_error_fraction": ev["speaker_error_fraction"],
               "counts": ev["counts"], "elapsed_s": diarization["elapsed_s"]}
        rows.append(row)
        print(row, flush=True)
    write_json(ROOT / "outputs/experiments/speaker_calibration.json", {
        "note": "same-sample exploration, not held-out validation; known6 uses true count, cannot assess automatic count estimation",
        "results": rows})


if __name__ == "__main__":
    main()
