"""Package the four requested result categories, with evidence and reproduction sources."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting_asr.output import read_json, write_json


def package():
    name = "语音转写_结果交付包_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    destination = ROOT / "release" / name
    destination.mkdir(parents=True, exist_ok=False)

    def copy(source, target):
        original = ROOT / source
        if not original.is_file():
            raise FileNotFoundError(f"Required deliverable missing: {original}")
        result = destination / target
        result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, result)

    mappings = {
        "deliverables/实验报告.md": "01_转写实验结果/实验报告.md",
        "deliverables/实验记录.csv": "01_转写实验结果/实验记录.csv",
        "deliverables/实验记录.json": "01_转写实验结果/实验记录.json",
        "deliverables/错误记录.json": "01_转写实验结果/错误记录.json",
        "deliverables/02_真实转写样例.json": "02_带时间和说话人标记的样例/02_真实转写样例.json",
        "deliverables/02_原始转写输出.json": "02_带时间和说话人标记的样例/02_原始转写输出.json",
        "deliverables/02_原始说话人输出.json": "02_带时间和说话人标记的样例/02_原始说话人输出.json",
        "outputs/experiments/C/transcript.txt": "02_带时间和说话人标记的样例/可直接阅读的文字样例.txt",
        "outputs/experiments/C/transcript.srt": "02_带时间和说话人标记的样例/带时间和说话人的字幕.srt",
        "deliverables/06_实时文字更新记录.jsonl": "03_实时转写记录/06_实时文字更新记录.jsonl",
        "deliverables/06_实时文字变化说明.md": "03_实时转写记录/06_实时文字变化说明.md",
        "outputs/experiments/E/summary.json": "03_实时转写记录/实时延迟与修订统计.json",
        "outputs/experiments/E/transcript.txt": "03_实时转写记录/最终文字.txt",
        "SOURCES.md": "SOURCES.md",
    }
    for original, target in mappings.items():
        copy(original, target)

    # Include the complete offline evidence, including failures and parameter trials.
    for case in ("A", "A_cpu", "B", "C", "D", "F", "F_t090", "F_t095", "F_known6"):
        for source in (ROOT / "outputs/experiments" / case).iterdir():
            if source.is_file():
                copy(source.relative_to(ROOT), Path("01_转写实验结果/原始运行证据") / case / source.name)
    for original in (ROOT / "data/samples").iterdir():
        if original.is_file():
            copy(original.relative_to(ROOT), Path("附录_复现资料/测试录音及参考") / original.name)
    for filename in ("asset_integrity.json", "deliverable_check.json", "offline_network_blocked.json", "tests.xml"):
        copy(Path("outputs/verification") / filename, Path("附录_复现资料/验证记录") / filename)
    for directory in ("meeting_asr", "scripts", "tests"):
        for source in (ROOT / directory).glob("*.py"):
            copy(source.relative_to(ROOT), Path("附录_复现资料/项目源码") / source.relative_to(ROOT))
    for filename in ("README.md", "SOURCES.md", "start.ps1", "setup.ps1", "requirements.txt", "requirements.lock.txt", ".env.example"):
        copy(filename, Path("附录_复现资料/项目源码") / filename)
    for filename in ("corpus_manifest.json", "asset_manifest.json"):
        copy(Path("data") / filename, Path("附录_复现资料/来源清单") / filename)

    recommendation = """# 推荐方案及限制

推荐先交付本地 Faster-Whisper small 的离线转写和词/片段时间；本机 GPU 可加速，CPU int8 也已跑通。实时结果用于即时查看，录音结束后可离线复核。说话人分析采用本地 ONNX 分段、声纹和聚类，交接匿名编号与未知标签。

多人区分仍有错误。自然会议默认阈值出现过度拆分，阈值调整改善了当前样本，但未在独立语料上验证泛化能力。交叠讲话可能漏掉其中一人，不能把提供人数当作准确识别人声的证据。时间戳没有逐词人工听音核验。

实时性能来自123.58秒录音按墙钟速度输入，不是两分钟真人麦克风对话。麦克风仅进行了约2.9秒采集链路探测，该次无有效讲话。实时 final 是应用分段规则下的稳定文字，仍可能识别错误。

云端 Deepgram 文件与流式适配已实现，密钥留空，未调用真实付费 API。只有本地 HTTP/WebSocket 协议测试，不能提供实际云端效果、网络延迟或费用结论。OpenAI 对照属于原文档可选项，未作为本次核心依赖。

详细实测数值、原始证据和局限见 ../01_转写实验结果/实验报告.md；数据来源及许可见根目录 SOURCES.md。
"""
    (destination / "04_推荐方案及限制").mkdir()
    (destination / "04_推荐方案及限制/推荐方案及限制说明.md").write_text(recommendation, encoding="utf-8")
    instructions = """# 交付说明（先读本文件）

本包对应任务截图要求的四类交付。所有模型结果来自实际本机录音实验；代码测试中的固定响应不当作真实云端实验。

| 截图交付项 | 包内位置 | 使用方式 |
|---|---|---|
| 转写实验结果 | 01_转写实验结果 | 先看实验报告.md；CSV用于表格查看，JSON和原始运行证据用于复核 |
| 带时间和说话人标记的文字样例 | 02_带时间和说话人标记的样例 | 文字样例和字幕可直接阅读；02_真实转写样例.json供02读取；另附两个原始输出 |
| 实时转写记录 | 03_实时转写记录 | 先看文字变化说明.md，再按segment_id读取JSONL；统计JSON记录延迟口径 |
| 推荐方案及限制 | 04_推荐方案及限制 | 单独提交推荐方案及限制说明.md，并与实验报告一起阅读 |

## 如何交付

1. 向负责汇总/老师/项目负责人发送整个ZIP；说明先阅读本文件和实验报告。
2. 向02发送第02目录，配套录音为附录_复现资料/测试录音及参考/C.wav。JSON的start/end单位为秒，speaker_id为匿名编号；null代表不能可靠归属，不是丢失字段。
3. 向06发送第03目录，配套录音为附录_复现资料/测试录音及参考/E.wav。相同segment_id的partial应替换，final才冻结，不能把每次更新追加成新发言。
4. 若需要重跑，使用附录中的源码、README和下载脚本准备本地环境与模型；包内不含虚拟环境、依赖缓存和约515MiB模型。本包用于交结果与源码，不能解压后直接当作免安装程序运行。

## 阅读和复核

MD是UTF-8文本，可用VS Code、Obsidian或文本编辑器打开；CSV可用Excel打开；TXT可直接阅读，SRT可与对应录音配合播放。JSON/JSONL为程序使用的结构化文本，不是Word文档。

附录含A–F测试录音和对应参考、来源清单及源码。A–E是原始真人片段构建的控制实验，F是自然会议截取。素材许可和改动见根目录SOURCES.md；AISHELL-4衍生音频遵循CC BY-SA 4.0。

SHA256SUMS.txt用于核对文件完整性，交付清单.json列出每个文件。保留错误记录和限制说明，不能只截取成功样例声称所有场景已达到高准确率。
"""
    (destination / "交付说明_先读.md").write_text(instructions, encoding="utf-8")
    entries = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            entries.append({"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(destination / "交付清单.json", {"file_count_excluding_manifest": len(entries), "files": entries})
    (destination / "SHA256SUMS.txt").write_text("\n".join(f"{r['sha256']}  {r['path']}" for r in entries) + "\n", encoding="utf-8")
    archive_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination.parent))
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP corruption: {bad}")
    print(json.dumps({"folder": str(destination), "zip": str(archive_path),
                      "zip_bytes": archive_path.stat().st_size, "files": len(entries) + 2,
                      "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    package()
