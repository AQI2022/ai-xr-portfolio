# AI XR 多模态交互助手

组合文字、语音和图像检测上下文，把模型回答与受控场景操作连接起来。界面有浏览器协议演示和真正的 Unity C# 客户端；不是只返回固定对话的截图工程。

## 实际链路

浏览器上传图像 → YOLO 检测 → 将检测结果带入 XR 助手；语音可用浏览器输入，也可调用服务端 `/speech/transcribe`（faster-whisper tiny）。`/agent/chat` 接收文本、会话 ID 和 JSON 场景状态，调用本地 Qwen 或兼容服务。输出文本可经 `/speech/synthesize` 合成 WAV，Unity 播放。

模型可选择知识检索和场景动作工具。场景动作只有 rotate/highlight/reset/show_step；先返回待确认票据，确认后客户端才修改场景。票据绑定会话，5 分钟失效且不能重放；模型不能直接执行 Shell 或调用任意 Unity 方法。会话记忆保存在 SQLite，最多保留 24 条，可主动清除。

视觉部分已实现并可运行目标检测上下文，不把 YOLO 说成 VLM。`/vision/describe` 是可配置的外部 VLM 入口，本次没有图像语言模型服务凭据，因此未验证该远程路径。语音使用系统 TTS，不包含声音克隆或数字人面部口型模型。

## Unity

用 Unity 2022.3 打开 `unity/XRPortfolio`。菜单 `AI XR/Generate Demo Scene` 自动生成摄像机、立方体、UI 和客户端组件。服务 URL 在 Inspector 设置或通过启动参数 `-server-url` 指定，默认本机 `http://127.0.0.1:8000`；界面可输入问题、发送、提议旋转、确认动作和播放回答。

批处理构建：

```powershell
& "<Unity Editor path>" -batchmode -quit -nographics -projectPath "<repo>/unity/XRPortfolio" -executeMethod BuildPortfolio.BuildWindows -logFile "<repo>/data/unity-build.log"
```

构建输出在 `Builds/Windows`，不进入 Git。网络请求使用 UnityWebRequest 协程、超时和忙状态，返回后在主线程更新场景。当前实体是 demo_cube；assembly/math_surface 是后端协议预留目标，Unity 演示不假装包含真实工业/数学模型。

本次构建验证使用 Unity 2022.3.13f1c1 Windows Standalone；没有安装 OpenXR/MRTK 运行时，也没有执行头显跟踪或空间锚点测试。它是适配 XR 的交互客户端原型，不是已验证的 HoloLens 应用。

## 可复现验证

```bash
python scripts/seed_demo.py
python scripts/verify_models.py
```

真实验证包括本地 Qwen 产生 `rotate demo_cube 45` 工具调用、确认消费、TTS 生成 WAV、Whisper 转写该英语音频。音频是合成干净样本，不代表噪声环境识别率。Agent 工具协议、跨会话拒绝和重放拒绝另由单元测试覆盖。
