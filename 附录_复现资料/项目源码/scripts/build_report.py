"""Build handoff materials from saved REAL results without rerunning models."""
from __future__ import annotations
import csv
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr.output import read_json, write_json
from meeting_asr.evaluation import evaluate, evaluate_speakers


def percent(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def build_report():
    source = ROOT / "outputs/experiments"
    target = ROOT / "deliverables"
    target.mkdir(exist_ok=True)
    cases = read_json(ROOT / "data/samples/cases.json")
    records, errors, evaluations = [], [], {}
    for case in cases:
        name = case["case"]
        path = source / name / "result.json"
        if not path.exists():
            continue
        result = read_json(path)
        ev = evaluate(result, case)
        write_json(source / name / "evaluation.json", ev)
        evaluations[name] = ev
        record = {"case": name, "scene": case["construction"], "duration_s": result["duration_s"],
                  "device": result["engine"]["device"], "model": result["engine"]["model"],
                  "asr_elapsed_s": result.get("elapsed_s"), "rtf": result.get("rtf"),
                  "pipeline_elapsed_s": result.get("pipeline_elapsed_s"), "cer": ev["text"].get("cer"),
                  "predicted_speakers": ev.get("speakers", {}).get("predicted_speakers"),
                  "first_partial_p50_s": result.get("first_partial_from_audio_start", {}).get("p50_s"),
                  "final_from_last_voice_p50_s": result.get("final_from_last_voice", {}).get("p50_s")}
        records.append(record)
        errors.extend(dict(e, case=name, category="text_edit_against_corpus") for e in ev["text"].get("error_examples", []))
        errors.extend(dict(e, case=name, category="speaker_diagnostic") for e in ev.get("speakers", {}).get("errors", []))
        errors.extend(dict(e, case=name, category="alignment_ambiguity") for e in result.get("anomalies", []))
    natural_ref_path = ROOT / "data/samples/F_natural_meeting.reference.json"
    natural_ref = read_json(natural_ref_path) if natural_ref_path.exists() else None
    calibrations = []
    if natural_ref:
        for name in ("F", "F_t090", "F_t095", "F_known6"):
            path = source / name / "result.json"
            if not path.exists():
                continue
            r = read_json(path)
            ev = evaluate_speakers(r["segments"], natural_ref["reference_turns"])
            ev["scope"] = "word-midpoint diagnostic against AISHELL-4 corpus mirror speaker intervals; NOT DER"
            evaluations[name] = {"speakers": ev, "text": {"cer": None, "reason": "mirror has no text reference"}}
            write_json(source / name / "evaluation.json", evaluations[name])
            raw = read_json(source / name / "diarization_raw.json")
            calibrations.append({"case": name, "threshold": raw["engine"]["cluster_threshold"],
                                 "known_count": raw["engine"]["num_speakers_hint"],
                                 "track_speakers": len({t["speaker_id"] for t in raw["tracks"]}),
                                 "word_label_speakers": ev["predicted_speakers"],
                                 "diagnostic_error_fraction": ev["speaker_error_fraction"],
                                 "counts": ev["counts"], "diarization_elapsed_s": raw["elapsed_s"]})
            errors.extend(dict(e, case=name, category="natural_meeting_speaker_diagnostic") for e in ev["errors"])
    summary = {"generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "hardware": {"cpu": "Intel Core i9-13900HX, 24 physical/32 logical cores", "ram_gb": 16,
                            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB", "platform": platform.platform()},
               "python": platform.python_version(),
               "versions": {n: importlib.metadata.version(n) for n in ("faster-whisper", "ctranslate2", "sherpa-onnx", "sounddevice", "onnxruntime")},
               "results": records, "natural_meeting_parameter_exploration": calibrations,
               "cloud": "implemented, local transport tests passed, real provider not called; API key empty",
               "microphone": read_json(ROOT / "outputs/microphone_probe/summary.json") if (ROOT / "outputs/microphone_probe/summary.json").exists() else None}
    write_json(source / "summary.json", summary)
    write_json(target / "实验记录.json", summary)
    write_json(target / "错误记录.json", {"note": "自动对照发现的编辑差异、错分和未知片段；不等同新增人工审核结论", "errors": errors})
    with (target / "实验记录.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    shutil.copy2(source / "C/result.json", target / "02_真实转写样例.json")
    shutil.copy2(source / "C/asr_raw.json", target / "02_原始转写输出.json")
    shutil.copy2(source / "C/diarization_raw.json", target / "02_原始说话人输出.json")
    e_path = source / "E/events.jsonl"
    live = read_json(source / "E/summary.json") if (source / "E/summary.json").exists() else None
    if e_path.exists():
        shutil.copy2(e_path, target / "06_实时文字更新记录.jsonl")
        events = [json.loads(line) for line in e_path.read_text(encoding="utf-8").splitlines()]
        first_id = events[0]["segment_id"]
        example = [{k: e[k] for k in ("segment_id", "state", "text", "audio_time", "emitted_at", "revision_count")}
                   for e in events if e["segment_id"] == first_id]
        snippet = "\n".join(json.dumps(e, ensure_ascii=False) for e in example)
        note = f"""# 向 06 交接：实时文字更新方式

以下来自本机真实运行的 E 样本，不是手写示例。

```jsonl
{snippet}
```

按 `segment_id` 维护一条记录：partial 到来时替换该记录的文字，不要追加成多段发言；final 到来时将其冻结。本地 final 是程序的停顿/最长窗口规则确认结果，仍可能识别错误，不代表人工校对完成。

`audio_time` 为音频推进秒数；`emitted_at` 为会话开始后本地单调时钟秒数。`capture_to_result_s` 只包含最近输入块排队和推理，不能作为从开始说话到看到文字的耗时。`audio_start_to_result_s` 包含窗口等待；`last_voice_to_result_s` 包含能量端点等待。麦克风 capture 时钟取回调到达时间，未声称是硬件 ADC 时间。

已提交文字通过绝对词时间过滤上下文重叠，不能只按相同文字删除，否则会误删真实的重复发言。强制切窗、停顿误判仍可能造成断词和漏字，参见错误记录。`revision_count` 统计原文被回改；`text_change_count` 还包含正常续写。

本地实时事件的 `speaker_id=null`：本轮说话人分析独立在离线录音上运行。需要标签时，对保存的 `captured.wav` 再执行离线 `--diarize`。

Deepgram 对接保留原始 `is_final` / `speech_final` 和原始 JSONL。只有 `is_final=true` 将对应片段置为 final；`speech_final` 仅表示检测到停顿，不能代替此前所有 final 片段。云端密钥为空，相关云端事件只有单元/本地协议测试，没有伪造远端实测样例。
"""
        (target / "06_实时文字变化说明.md").write_text(note, encoding="utf-8")
    lines = ["# 语音转写实现与实验报告", "", f"生成时间：{summary['generated_at']}（北京时间）。", "",
             "已实现本地离线转写、词/片段时间戳、本地说话人分段与文字对齐、麦克风近实时转写、录音按实时速度回放，以及 Deepgram 文件/流式适配。云端密钥保持为空，未调用真实付费 API。", "",
             "## 运行环境与复现", "", "Windows；i9-13900HX；16 GB 内存；RTX 4060 Laptop 8 GB。Python " + summary["python"] + "。版本记录见实验记录.json。",
             "本机使用 small：GPU float16，CPU int8。ASR 耗时包含惰性结果迭代，排除模型加载和音频解码；pipeline 耗时包含解码、ASR、说话人分析，排除模型加载。RTF 是离线速度比值，不是实时显示延迟。", "",
             "## 对应截图的交付", "", "| 要求 | 已完成的证据 |", "|---|---|",
             "| 本地转写、文字效果、处理耗时、片段时间 | A/B/C/D 结果、CPU 对照、CER 与错误记录 |",
             "| 实时文字变化与延迟 | E 持续回放的事件 JSONL、延迟分布；真实麦克风短时采集探测 |",
             "| 多人区分、错分与无法判断 | C/D/F 原始说话人时间段、对齐结果、空标签及诊断记录 |",
             "| 向 02 提供实际输出 | 02_真实转写样例.json 及两个原始结果文件 |",
             "| 向 06 说明更新方式 | 06_实时文字更新记录.jsonl 与说明文档 |", "",
             "## 离线结果", "", "| 样本 | 音频秒数 | ASR 秒数 | RTF | CER | 场景 |", "|---|---:|---:|---:|---:|---|"]
    scenes = {"A": "单人清晰", "B": "A 加10 dB噪声/回声", "C": "两人轮流（拼接）", "D": "三人且重叠（混音）"}
    for record in records:
        if record["case"] == "E":
            continue
        lines.append(f"| {record['case']} | {record['duration_s']:.2f} | {record['asr_elapsed_s']:.2f} | {record['rtf']:.4f} | {percent(record['cer'])} | {scenes[record['case']]} |")
    cpu = read_json(source / "A_cpu/result.json")
    cpu_ev = evaluate(cpu, cases[0])
    write_json(source / "A_cpu/evaluation.json", cpu_ev)
    lines += [f"| A CPU | {cpu['duration_s']:.2f} | {cpu['elapsed_s']:.2f} | {cpu['rtf']:.4f} | {percent(cpu_ev['text']['cer'])} | CPU int8 对照 |", "",
              "CER 参考来自 AISHELL 原始人工转写，本轮没有新增人工听写。标准为 NFKC、去标点/符号/空白、英文字母小写；数字不转汉字，繁简也未自动折算。D 同时说话没有唯一串行参考，未计算 CER。时间戳通过范围与顺序检查，尚未做逐词人工听音核验，不能据此宣称毫秒级准确。", "",
              "A–E 由三位真实人声的 54 条录音构建，有原始文本和源文件名；A/B 约1分钟，C/D/E 约2分钟。拼接和加噪过程、种子、重叠区间均可复现，不能代表自然会议。", "",
              "## 说话人诊断与自然会议", "", "C 自动得到2个文字标签；D 自动得到3个。对齐依赖 ASR 词时间及分段重叠；不足覆盖、多说话人覆盖或零长度词保留 null，并记录原因。声纹 ID 仅在当前文件内有效，与真实姓名无关。", "",
              "额外使用 AISHELL-4 自然会议连续120秒，包含6名说话人的语料时间标注；保留真实打断和交叠。下表是同一录音上的参数探索，并非独立测试集上的调优结论。", "",
              "| F 配置 | 聚类时间轨道人数 | 获得文字的标签人数 | 非交叠词诊断错误比例 |", "|---|---:|---:|---:|"]
    for c in calibrations:
        label = f"阈值 {c['threshold']}，自动人数" if c["known_count"] < 0 else f"提供已知人数 {c['known_count']}"
        lines.append(f"| {label} | {c['track_speakers']} | {c['word_label_speakers']} | {percent(c['diagnostic_error_fraction'])} |")
    lines += ["", "诊断把识别词的中点与参考说话人时间区间对照，先通过最优一对一映射对齐匿名编号，再统计错分和未知。它受词时间错误、漏字和参考边界影响，不是 DER，也不是完整的说话人准确率。重叠区间单独统计。已知人数模式有标签没获得文字时仍可能漏转写，不能通过指定人数掩盖。", "",
              "自然会议默认阈值0.8有明显过度拆分；0.95在此样本上改善，但不能保证适用于其他录音。重叠语音仍会丢失其中一人的文字；本实现没有声源分离，也没有实时声纹追踪。", ""]
    if live:
        lines += ["## 实时结果与计时口径", "",
                  f"E 按墙钟速度输入 {live['duration_s']:.2f} 秒，实际运行 {live['wall_elapsed_s']:.2f} 秒；{live['partial_count']} 个 partial 事件、{live['final_count']} 个 final、{live['revision_count']} 次回改。CER 为 {percent(evaluations['E']['text']['cer'])}，明显差于长录音离线识别，不能把功能跑通等同高准确率。", "",
                  "| 指标 | P50 | P95 |", "|---|---:|---:|"]
        for name, key in [("含窗口积累：片段开始→首个 partial", "first_partial_from_audio_start"),
                          ("含端点等待：最后检测语音→final", "final_from_last_voice"),
                          ("仅最近输入块→首次 partial", "first_partial_latency"),
                          ("仅最近输入块→final", "final_latency")]:
            m = live.get(key, {})
            if m.get("p50_s") is not None:
                lines.append(f"| {name} | {m['p50_s']:.3f}s | {m['p95_s']:.3f}s |")
        lines += ["", "配置为1.5秒更新间隔、0.8秒能量停顿、最长12秒窗口、0.6秒上下文重叠。final 是应用层分段规则，并非 Faster-Whisper 原生流式状态。固定能量阈值在噪声/弱声环境可能切错句，需要调节或替换为专用流式 VAD。", "",
                  "麦克风实际采集约2.9秒，链路完成且无输入错误，但本次未检测到有效讲话，因此没有可报告的真人现场识别准确率。两分钟指标来自录音实时回放，未冒充真人两分钟麦克风实验。", ""]
    lines += ["## 验证、推荐路线及限制", "",
              "已通过本地 HTTP/WebSocket 协议测试、空密钥禁止上传测试、说话人边界/重叠测试、实时文字修订/最终冻结/静音/长窗口重复发言测试，以及 CER 计算测试。具体数量见 outputs/verification/tests.xml。另在 Python socket.connect 被阻断时成功加载本地模型并完成 CPU 转写和说话人分析（outputs/verification/offline_network_blocked.json）。这不是操作系统级隔离测试。", "",
              "建议先采用本地离线 small+时间戳作为基础；本机 GPU 明显快于 CPU。多人功能先核查未知和重叠片段，人数已知时可提供提示；实时结果用于即时查看，结束后可对 captured.wav 离线复核。Deepgram 代码和请求计划已完成，真实效果、网络延迟及费用待填入云端密钥后在同批样本上比较。", "",
              "首次下载需要网络；下载完成后本地推理使用项目模型路径并禁止自动拉取。模型、音频、虚拟环境和缓存均在当前项目；GPU 模式复用本机已有 CUDA/cuDNN。没有实现 AI 纪要、业务界面或系统集成。", "",
              "来源与许可详见 SOURCES.md。"]
    (target / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {target / '实验报告.md'}", flush=True)
    return summary


if __name__ == "__main__":
    build_report()
