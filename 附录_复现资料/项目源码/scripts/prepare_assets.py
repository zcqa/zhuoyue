"""Download only public assets, into this project. No credentials needed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr import ROOT


def download(url, target, expected_sha=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (not expected_sha or sha256(target) == expected_sha):
        print(f"Cached: {target.relative_to(ROOT)}", flush=True)
        return target
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "meeting-asr-research/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                append = offset > 0 and response.status == 206
                with part.open("ab" if append else "wb") as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
            if expected_sha and sha256(part) != expected_sha:
                part.unlink()
                raise ValueError("SHA256 mismatch")
            part.replace(target)
            print(f"Downloaded: {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)", flush=True)
            return target
        except Exception as exc:
            print(f"Retry {attempt + 1}/5: {target.name}: {exc}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prepare(model="small", endpoint="https://huggingface.co"):
    manifest = []
    old_manifest_path = ROOT / "data/asset_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8")) if old_manifest_path.exists() else []
    old_hashes = {r["path"].replace("\\", "/"): r["sha256"] for r in old_manifest}
    repo = f"Systran/faster-whisper-{model}"
    with urllib.request.urlopen(f"{endpoint}/api/models/{repo}", timeout=45) as r:
        info = json.load(r)
    revision = info["sha"]
    previous_revisions = {r.get("revision") for r in old_manifest if f"faster-whisper-{model}/" in r["path"].replace("\\", "/")}
    if previous_revisions:
        # Reuse the pinned previously downloaded model rather than mixing revisions.
        revision = next(iter(previous_revisions))
    for name in ("config.json", "tokenizer.json", "vocabulary.txt", "model.bin", "README.md"):
        url = f"{endpoint}/{repo}/resolve/{revision}/{name}"
        dest = f"models/faster-whisper-{model}/{name}"
        path = download(url, ROOT / dest, old_hashes.get(dest))
        manifest.append({"path": str(path.relative_to(ROOT)), "url": url, "sha256": sha256(path), "revision": revision})
    base = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    assets = [
        ("speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", "models/downloads/segmentation.tar.bz2"),
        ("speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx", "models/speaker-embedding.onnx"),
        ("speaker-segmentation-models/0-four-speakers-zh.wav", "data/raw/four-speakers-zh.wav"),
        ("speaker-segmentation-models/1-two-speakers-en.wav", "data/raw/two-speakers-en.wav"),
    ]
    for tail, dest in assets:
        path = download(base + tail, ROOT / dest, old_hashes.get(dest))
        manifest.append({"path": dest, "url": base + tail, "sha256": sha256(path)})
    with tarfile.open(ROOT / "models/downloads/segmentation.tar.bz2") as archive:
        # Reject escaping paths and links even when downloading an upstream archive.
        for member in archive.getmembers():
            dest = (ROOT / "models" / member.name).resolve()
            if not dest.is_relative_to((ROOT / "models").resolve()) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(ROOT / "models", filter="data")
    target = ROOT / "data/asset_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    prepare(args.model, args.endpoint)
