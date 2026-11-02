from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
from . import ROOT


def parser():
    p = argparse.ArgumentParser(description="本地语音转写、说话人区分、实时转写及 Deepgram 对照")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="列出麦克风设备")
    for name in ("offline", "realtime", "cloud", "cloud-stream"):
        cmd = sub.add_parser(name)
        cmd.add_argument("audio", nargs="?", help="输入文件；实时命令可改用 --mic")
        cmd.add_argument("--output", type=Path)
        if name in ("offline", "realtime"):
            cmd.add_argument("--model", default="small")
            cmd.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
            cmd.add_argument("--compute-type")
            cmd.add_argument("--threads", type=int, default=8)
            cmd.add_argument("--language", default="zh")
        if name == "offline":
            cmd.add_argument("--diarize", action="store_true")
            cmd.add_argument("--speakers", type=int, default=-1, help="已知人数；-1 自动估计")
            cmd.add_argument("--cluster-threshold", type=float, default=.8)
        if name in ("realtime", "cloud-stream"):
            cmd.add_argument("--mic", action="store_true")
            cmd.add_argument("--input-device", type=int)
            cmd.add_argument("--duration", type=float, default=120, help="秒，0 为整个文件/麦克风直到 Ctrl+C")
        if name == "realtime":
            cmd.add_argument("--interval", type=float, default=1.5)
            cmd.add_argument("--silence", type=float, default=.8)
            cmd.add_argument("--max-window", type=float, default=12)
            cmd.add_argument("--energy-threshold", type=float, default=.008)
            cmd.add_argument("--no-pace", action="store_true", help="加速逻辑测试；不代表真实实时延迟")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    if not args.audio and not getattr(args, "mic", False):
        p.error("请提供音频文件或 --mic")
    if args.audio and getattr(args, "mic", False):
        p.error("音频文件与 --mic 只能选一种输入")
    output = args.output or ROOT / "outputs" / f"{args.command}_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        if args.command in ("offline", "realtime"):
            from .asr import LocalASR
            asr = LocalASR(args.model, args.device, args.compute_type, args.threads, args.language)
        if args.command == "offline":
            from .offline import run_offline
            from .diarization import LocalDiarizer
            diarizer = LocalDiarizer(args.speakers, args.cluster_threshold) if args.diarize else None
            result = run_offline(args.audio, output, asr, diarizer)
            print(f"音频 {result['duration_s']:.2f}s；ASR {result['elapsed_s']:.2f}s；RTF {result['rtf']}")
        elif args.command == "realtime":
            from .realtime import run_realtime
            result = run_realtime(asr, output, path=args.audio, microphone=args.mic, device=args.input_device,
                                  duration=args.duration, pace=not args.no_pace, interval=args.interval,
                                  silence=args.silence, max_window=args.max_window, energy_threshold=args.energy_threshold)
        elif args.command == "cloud":
            from .cloud import run_batch
            result = run_batch(args.audio, output)
        else:
            from .cloud import run_cloud_stream
            result = run_cloud_stream(args.audio, output, microphone=args.mic, device=args.input_device, duration=args.duration)
        print(f"结果目录：{Path(output).resolve()}")
        return 1 if result.get("status") == "failed" else 0
    except Exception as exc:
        from .output import write_json
        write_json(Path(output) / "error.json", {"type": type(exc).__name__, "message": str(exc), "command": args.command})
        print(f"无法完成：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
