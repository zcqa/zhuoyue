# 实现依据、数据来源与许可

以下资料在实现时已访问；下载文件的实际 SHA256、模型提交号和原始地址见 `data/asset_manifest.json`、`data/corpus_manifest.json` 及 `data/samples/*.reference.json`。

## 代码与模型

- [Faster-Whisper 官方 README](https://github.com/SYSTRAN/faster-whisper)：本地转写、惰性迭代、词时间、CPU int8、CUDA兼容说明。按官方 CUDA12/cuDNN8 对应说明固定 CTranslate2 4.4.0，复用本机现有动态库。
- [small 转换模型](https://huggingface.co/Systran/faster-whisper-small)：本次提交 `536b0662742c02347bc0e980a01041f333bce120`。模型卡保存在模型目录；实际推理只读取本地文件。
- [sherpa-onnx 官方说话人文档](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html)与[官方 Python 示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py)：使用语音分段模型、3D-Speaker 声纹提取和聚类。没有安装整个 PyTorch/pyannote.audio 管线，采用官方提供的 ONNX 路线完成本地同类任务。
- [分段模型发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)：下载 `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`。包内 MIT LICENSE 和 README 已保留在模型目录。
- [声纹模型发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)：下载 `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`，项目内保存为 `models/speaker-embedding.onnx`。该文件更名未修改模型内容；源URL和哈希在清单中。

项目代码是本次任务的实现；上游库和模型保留各自许可，不能把本项目代码的使用范围当作上游模型的许可说明。

## 主要实验音频

### AISHELL-1：A–E 可复现控制样本

- 原作者：北京希尔贝壳科技有限公司（Beijing Shell Shell Technology Co., Ltd.），AISHELL-1 论文作者 Hui Bu 等。
- [原始发布与许可](https://www.openslr.org/33/)：Apache License 2.0。
- [本次使用的公开镜像](https://huggingface.co/datasets/carlot/AIShell)，提交 `692ef58020d79b21f54eb25b15a4813d4f9650d7`。
- 使用测试集54条原始人声，按源文件名确认三位不同说话人 `S0764/S0765/S0766`。没有用语音合成冒充真人录音。
- 原始文本来自语料人工标注，不是本次模型输出，也不是本轮重新人工听写。
- 改动：统一16 kHz单声道、拼接、增加间隔、按已知时间混入重叠；B增加固定种子的10 dB噪声和回声。每段来源、顺序、时间及参考文字均保留在 reference.json。A–E不是自然会议录音。
- 采集器只通过 HTTP Range 读取 Parquet 的路径/文本列，再下载54个小音频，没有下载整个20 GB镜像或完整15 GB原始语料。

### AISHELL-4：F 自然会议

- 原作者：北京希尔贝壳科技有限公司及 AISHELL-4 作者 Yihui Fu 等。
- [原始发布](https://www.openslr.org/111/)；数据许可为 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。AISHELL-4 仓库的代码许可与音频数据许可不同，本项目按数据许可记录。
- [本次使用镜像](https://huggingface.co/datasets/argmaxinc/aishell-4)，提交 `0cf6e53877b50888a93cc835d8d6beb08d3fb917`，测试集第0行。
- 改动：从镜像的16 kHz单声道 WAV 中截取原始时间1560–1680秒，共120秒，保持原来的交谈顺序、停顿与交叠，未混入其他音频。衍生音频及其标注沿用 CC BY-SA 4.0。
- 通过 HTTP Range 仅取120秒需要的数据和WAV头部，不下载整场会议或整套语料。原始说话人时间标注随片段平移并截断保存。
- 该镜像只提供说话人时间标注，没有文字参考，因此 F 未计算 CER；模型转写不能反过来冒充参考文本。

### 官方附加诊断样例

`data/raw/four-speakers-zh.wav`、`two-speakers-en.wav` 来自 sherpa-onnx 官方分段模型发布页，官方示例直接使用前者。它们用于初始兼容/诊断，与有明确数据许可的 A–F 主实验分开记录。发布页未单列这两段录音的独立许可，不能据此推定其中人声/音乐拥有任意再分发授权。

## 云端接口

- [Nova-3 模型与语言](https://developers.deepgram.com/docs/models-languages-overview)：普通话配置。
- [说话人标记](https://developers.deepgram.com/docs/diarization)：`diarize_model=latest`、词级时间和匿名标签，批量与流式版本不同。
- [临时文字和端点检测](https://developers.deepgram.com/docs/understand-endpointing-interim-results)：`interim_results`、`is_final`、`speech_final` 的语义。

云端密钥为空；官方说明仅用于编写适配器，不代替实际付费 API 测试。本地协议测试的固定响应仅用于验证代码，不作为识别效果或网络延迟结论。
