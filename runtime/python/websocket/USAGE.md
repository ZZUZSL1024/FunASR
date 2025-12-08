# Websocket & REST 服务启动与调用指南

本文档说明如何启动离线转写/声纹注册（Flask HTTP）服务，以及实时转写（WebSocket）服务，并演示如何让两套服务共用本地声纹库。

## 目录
- [环境准备](#环境准备)
- [启动离线转写与声纹注册服务](#启动离线转写与声纹注册服务)
- [启动实时转写服务](#启动实时转写服务)
- [声纹注册接口](#声纹注册接口)
- [离线转写接口](#离线转写接口)
- [实时转写接口](#实时转写接口)
- [共享声纹库说明](#共享声纹库说明)

## 环境准备
1. 安装依赖（建议已安装 `pip`）：
   ```bash
   pip install -r requirements.txt
   ```
2. 确保 GPU 环境正确（如无 GPU 会自动使用 CPU）。

## 启动离线转写与声纹注册服务
该服务提供 HTTP 接口，默认监听 `0.0.0.0:10099`。

```bash
cd runtime/python/websocket
python funASR-Service.py
```

启动后会自动创建 `uploads/` 目录用于临时存储上传的音频，并在当前目录生成共享声纹库文件 `speaker_db.json`。

## 启动实时转写服务
实时转写通过 WebSocket 提供，默认监听 `0.0.0.0:10095`。

```bash
cd runtime/python/websocket
python funasr_wss_server.py
```

如需启用 TLS，可通过 `--certfile` 和 `--keyfile` 指定证书，或调整端口、设备等参数：
```bash
python funasr_wss_server.py --port 10095 --device cuda --ngpu 1
```

## 声纹注册接口
- **URL**: `POST http://<host>:10099/Register_Speaker`
- **参数**:
  - `file`：必填，WAV/MP3 等音频文件（`form-data` 文件字段）。
  - `speaker_name`：必填，声纹在本地库中的名称。
- **返回**: `status`、`speaker_name`、`result`（向量列表）。

示例：
```bash
curl -X POST "http://localhost:10099/Register_Speaker" \
  -F "file=@/path/to/audio.wav" \
  -F "speaker_name=alice"
```

调用成功后，声纹向量会追加到 `speaker_db.json`，供后续离线/实时识别使用。

## 离线转写接口
- **URL**: `POST http://<host>:10099/AsrCamWithIdentify`
- **参数**:
  - `file`：必填，待转写音频文件。
  - `identify_speakers`：可选，`true`/`false`（字符串），开启后会对输出句子进行声纹匹配。
  - `speaker_db`：可选，JSON 字典字符串。如不传，则自动使用本地 `speaker_db.json`；传入可在已有库基础上追加/覆盖。
- **返回**: `status`、`result`（包含时间戳、文本、匹配到的 `spk_name` 与相似度）。

示例（使用本地声纹库比对）：
```bash
curl -X POST "http://localhost:10099/AsrCamWithIdentify" \
  -F "file=@/path/to/meeting.wav" \
  -F "identify_speakers=true"
```

## 实时转写接口
- **URL**: `ws://<host>:10095`（默认端口，可根据启动参数调整）
- **交互流程**:
  1. 建立 WebSocket 连接。
  2. 发送 JSON 设定参数（可选），如：
     ```json
     {"chunk_size": "5,10", "mode": "2pass", "wav_name": "demo"}
     ```
  3. 持续发送音频二进制帧；根据 VAD 自动切分，或通过 `is_speaking` 字段控制结束。
  4. 服务返回识别结果 JSON；当 `is_final` 为 `false` 且 `mode` 包含 `offline`/`online` 字样时代表对应阶段结果。

实时识别会在每次离线二次识别阶段自动加载最新的 `speaker_db.json`，将当前说话人与本地声纹库匹配。

## 共享声纹库说明
- 声纹库文件路径：`runtime/python/websocket/speaker_db.json`。
- 注册接口会写入/追加；实时与离线接口使用时自动读取最新版本。
- 如需手工维护，可直接编辑 JSON 中的 `{"speaker_name": [embedding...]}` 映射，但建议通过注册接口写入。
