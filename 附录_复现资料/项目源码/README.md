# 本地语音转写实验工具

已实现：离线录音转写与词/片段时间、多人说话人分段及对齐、麦克风近实时转写、录音实时回放、真实实验记录、02/06 交接样例，以及 Deepgram 文件/流式接口。云端密钥为空；本地功能不需要密钥。

优先阅读 [实验报告](deliverables/实验报告.md)，结果由本机实际运行生成，不是文档中的示意 JSON。

## 当前电脑直接运行

在 `D:\lab_demo\zhuoyue` 打开 PowerShell。已建立 `.venv` 并下载所需模型、测试音频，无需再次安装。

```powershell
# 菜单：离线、实时回放、麦克风、实验、设备及测试
.\start.ps1

# 本地离线 + 说话人区分。默认 CPU，可完全离线运行。
.\.venv\Scripts\python.exe -m meeting_asr offline data\samples\C.wav --diarize --output outputs\my_offline

# 本机 GPU 加速；人数已知时可增加 --speakers 2，未知时自动估计。
.\.venv\Scripts\python.exe -m meeting_asr offline data\samples\C.wav --device cuda --diarize --output outputs\my_gpu

# 原速持续输入，终端连续输出 partial/final；0 表示回放整个文件。
.\.venv\Scripts\python.exe -m meeting_asr realtime data\samples\E.wav --device cuda --duration 0 --output outputs\my_realtime

# 查看设备后，选择物理麦克风。当前机器设备2是内置麦克风阵列；设备编号可能变化。
.\.venv\Scripts\python.exe -m meeting_asr devices
.\.venv\Scripts\python.exe -m meeting_asr realtime --mic --input-device 2 --device cuda --duration 120 --output outputs\my_microphone
```

`start.ps1` 默认使用 CPU，兼容性较好；直接命令可指定 GPU。若 PowerShell 拒绝执行脚本，可以直接使用上述 Python 命令，无需修改系统执行策略。

## 输入、输出与参数

输入支持 PyAV 可解码的 WAV、MP3、M4A、FLAC 等录音，自动转换为单声道16 kHz。建议先确认音频未损坏。

离线目录包含：

- `asr_raw.json`：原始 ASR 片段、词时间、词概率及 ASR 性能。
- `diarization_raw.json`：独立说话人时间轨道、阈值、人数提示和耗时。
- `result.json`：对齐后的统一片段、匿名说话人及异常记录。
- `transcript.txt` / `transcript.srt`：可直接阅读的时间文本与字幕。

实时目录包含：

- `events.jsonl`：同一 `segment_id` 的 partial 修订与 final 事件，每次写入立即刷新。
- `captured.wav`：实际送入本地引擎的录音，可事后离线复核。
- `result.json`、`summary.json`、文字和字幕：最终结果、延迟分布与输入异常。

本地说话人编号只在当前文件内有意义，不代表身份；重叠/低覆盖/零长度词可输出 `speaker_id=null`。有说话人轨道但没有识别文字时，不能算转写成功。

重要参数：

| 命令 | 参数 | 含义 |
|---|---|---|
| offline | `--language zh/auto/en` | 指定语言或自动判断 |
| offline | `--speakers 2` | 已知说话人数；默认自动估计 |
| offline | `--cluster-threshold 0.8` | 自动聚类阈值；增大一般减少人数，需按录音测试 |
| realtime | `--interval 1.5` | 每次重新识别之间的音频间隔 |
| realtime | `--silence 0.8` | 能量低于阈值达到该时长后，确认片段 |
| realtime | `--max-window 12` | 最长片段窗口，达到后带上下文继续 |
| realtime | `--energy-threshold 0.008` | 归一化 PCM 的 RMS 语音起止阈值，弱声可降低 |
| realtime | `--duration 0` | 文件全部回放；麦克风持续到 Ctrl+C |
| realtime | `--no-pace` | 只用于加速逻辑排查；此模式的时间不能作实时延迟结论 |

本地实时是滑动窗口近实时识别，final 来自应用规则。说话人识别作为独立离线功能运行；要为实时会话加说话人标签，可执行：

```powershell
.\.venv\Scripts\python.exe -m meeting_asr offline outputs\my_microphone\captured.wav --diarize --device cuda --output outputs\my_microphone_review
```

如果长时间推理跟不上输入，程序最多缓冲约60秒，之后停止并明确记录溢出，不会静默丢音后仍声称完整成功。Ctrl+C 会保留已处理数据并标记中断，未消费的队列内容可能未识别。

## 云端方案（密钥留空）

`.env` 和 `.env.example` 中的 `DEEPGRAM_API_KEY` 均为空。只有主动执行云端命令并配置有效密钥时才会上传录音。本地命令不读取或依赖这个密钥。

```powershell
.\.venv\Scripts\python.exe -m meeting_asr cloud data\samples\C.wav --output outputs\deepgram_batch
.\.venv\Scripts\python.exe -m meeting_asr cloud-stream data\samples\E.wav --duration 0 --output outputs\deepgram_stream
.\.venv\Scripts\python.exe -m meeting_asr cloud-stream --mic --input-device 2 --duration 120 --output outputs\deepgram_mic
```

批量使用 Nova-3 中文普通话及 `diarize_model=latest`；流式使用 WebSocket、`interim_results=true`、`endpointing=300`。保存原始响应及规范化结果；缺失时间或标签保持 null。空密钥命令只写请求计划和错误说明，返回非零退出码，不发送录音。

流式麦克风端要求所选设备支持16 kHz单声道；本地麦克风端额外支持回退到设备默认采样率并重采样。`speech_final` 只作停顿提示，不能代替 `is_final`。真实云端识别效果、响应字段、网络延迟及费用均尚未实测。

OpenAI 文字对照是原实施文档中的可选项，本次没有作为核心依赖实现；文件、实时、说话人三项云端适配集中在 Deepgram。

## 从头复现下载与实验

需要64位 Python 3.11、网络及足够磁盘空间。所有 Python 包安装在项目 `.venv`，pip/Hugging Face/PyTorch 缓存指向项目 `.cache`，模型在 `models`、音频在 `data`。推理只加载本地路径，缺模型时报错，不自动联网补下载。

```powershell
.\setup.ps1

# 或逐步执行
py -3.11 -m venv .venv
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache\pip'
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts\prepare_assets.py
.\.venv\Scripts\python.exe scripts\collect_samples.py
.\.venv\Scripts\python.exe scripts\collect_natural_meeting.py
.\.venv\Scripts\python.exe scripts\run_experiments.py --device cuda
.\.venv\Scripts\python.exe scripts\tune_speakers.py
.\.venv\Scripts\python.exe scripts\build_report.py
```

`run_experiments.py` 执行 A–D、A 的 CPU 对照、存在时的 F 自然会议，以及 E 原速实时回放，约有两分钟必须等待音频推进；同名输出会覆盖。`--skip-realtime` 可跳过回放；已有 E 结果仍保留。`build_report.py` 只读取结果生成交接材料，不重复推理。阈值探索在同一 F 样本上进行，不能当作独立验证。

本机 GPU 使用已有 CUDA12.1/cuDNN8，固定 CTranslate2 4.4.0；其他机器若缺这些动态库，请先使用 `--device cpu`。仅检测到 NVIDIA 显卡不代表依赖库已齐备。Python 3.12/3.13 和其他 CUDA 版本未在本项目验证。

## 测试和材料位置

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs\verification\tests.xml
```

测试中的假设识别结果和本地 HTTP/WebSocket 服务仅用于代码验证，位于临时目录，与真实录音实验分开；没有将测试夹具当作云端实测结果。

| 文件/目录 | 内容 |
|---|---|
| `deliverables/实验报告.md` | 对应截图四项工作的实测报告与限制 |
| `deliverables/实验记录.csv` / `.json` | 性能、CER、实时延迟及配置 |
| `deliverables/错误记录.json` | 文字编辑差异、说话人错分和未知片段 |
| `deliverables/02_真实转写样例.json` | 真实双人输出及时间标签 |
| `deliverables/06_实时文字更新记录.jsonl` | 完整真实实时事件流 |
| `deliverables/06_实时文字变化说明.md` | 更新、确认和计时口径 |
| `outputs/experiments` | 全部原始输出、评估与阈值对照 |
| `data/corpus_manifest.json` | 54条真实录音的出处、说话人源ID、人工文本和哈希 |
| `data/samples/*.reference.json` | 样本构建与参考标注；F另含源会议偏移 |
| `data/asset_manifest.json` | 模型及附加测试音频的下载地址、版本、SHA256 |
| `SOURCES.md` | 官方实现依据、数据来源和许可 |

本项目交付范围不包含 AI 纪要、业务产品界面、文档生成服务或系统集成。
