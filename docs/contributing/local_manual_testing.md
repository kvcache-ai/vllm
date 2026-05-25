# 本地手测指南（vLLM）

本文档面向**本地开发/手测**，按「从快到慢」列出可复制的命令。所有 Python 命令均使用仓库根目录下的 `.venv`，并通过 `uv` 管理依赖（见 [AGENTS.md](../../AGENTS.md)）。

当前分支若包含 **ECMooncakeConnector**（`add-mooncake-ec-connector`），文末有对应专项测试。

---

## 0. 前置条件

| 项 | 说明 |
|----|------|
| OS | Linux + NVIDIA GPU（CUDA） |
| Python | 3.12，由 `uv venv` 创建 |
| 网络 | 拉模型需访问 Hugging Face；若 `huggingface.co` 不可达，使用镜像（见下） |
| 命令前缀 | 下文默认已 `cd` 到仓库根目录 |

### 一键初始化（推荐首次）

```bash
cd /path/to/vllm
bash scripts/dev-setup-minimal.sh
```

脚本会：创建 `.venv`、可编辑安装 vLLM（预编译 wheel）、跑最小单元测试、预拉 `facebook/opt-125m`、跑 GPU 离线推理冒烟。

### 国内 / 无外网 Hugging Face

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 能直连 HF 时可 unset
```

### 手动安装（不用脚本时）

```bash
uv venv --python 3.12 --seed --managed-python
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest tblib
```

跑更多 pytest 用例时，安装测试依赖：

```bash
# x86_64 可用锁文件；其他平台用 .in
uv pip install -r requirements/test/cuda.in
```

### 快速自检

```bash
.venv/bin/python -c "import vllm, torch; print(vllm.__version__, torch.cuda.is_available())"
```

---

## 1. 无 GPU 冒烟（约 1 秒）

不加载模型，只验证 pytest 与 conftest 能工作。

```bash
.venv/bin/python -m pytest tests/test_logger.py::test_trace_function_call -v
```

---

## 2. GPU 离线推理（约 20–120 秒）

### 2.1 快速冒烟（推荐先跑）

`enforce_eager` 跳过 CUDA graph 编译，启动更快。

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 若未缓存模型，先拉取（约 1 分钟）
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m')"

.venv/bin/python scripts/smoke-inference.py
```

期望输出包含非空的 `Prompt` / `Output` 行。

### 2.2 官方示例（4 条 prompt）

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
.venv/bin/python examples/basic/offline_inference/basic.py
```

首次运行可能需 **1–2 分钟**（torch.compile + CUDA graph）；之后会快很多。

---

## 3. HTTP 服务 + curl（OpenAI 兼容 API）

### 3.1 启动服务

终端 1：

```bash
cd /path/to/vllm
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

.venv/bin/vllm serve facebook/opt-125m \
  --enforce-eager \
  --max-model-len 128 \
  --gpu-memory-utilization 0.5 \
  --disable-log-stats
```

默认监听 `http://127.0.0.1:8000`，就绪后日志出现 `Application startup complete`。

### 3.2 健康检查

终端 2：

```bash
curl -s http://127.0.0.1:8000/health
# 期望 HTTP 200
curl -s http://127.0.0.1:8000/v1/models | .venv/bin/python -m json.tool
```

### 3.3 Completions API（opt-125m 适用）

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, my name is",
    "max_tokens": 32,
    "temperature": 0
  }' | .venv/bin/python -m json.tool
```

期望：`choices[0].text` 非空，`usage.completion_tokens` > 0。

### 3.4 Chat API 说明

`facebook/opt-125m` **没有** chat template，`/v1/chat/completions` 会返回 400。要测 Chat，请换 instruct 模型，例如：

```bash
# 需更多显存与下载时间
.venv/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct --enforce-eager --max-model-len 512
```

### 3.5 停止服务

```bash
pkill -f "vllm serve facebook/opt-125m"
# 或 Ctrl+C 停掉终端 1
```

---

## 4. Mooncake EC 连接器（本分支专项）

依赖与文档：

- 安装：`uv pip install mooncake-transfer-engine`（CUDA 13 主机见 [mooncake_connector_usage.md](../features/mooncake_connector_usage.md) 中的 `mooncake-transfer-engine-cuda13`）
- EPD 背景：[disagg_encoder.md](../features/disagg_encoder.md)
- 集成说明：[tests/v1/ec_connector/integration/README.md](../../tests/v1/ec_connector/integration/README.md)

额外 Python 包（EC e2e / registry）：`pyzmq httpx fastapi uvicorn`

```bash
uv pip install mooncake-transfer-engine pyzmq httpx fastapi uvicorn
```

### 4.1 单元测试（无需真实 Mooncake，mock TransferEngine）

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_mooncake_connector.py -v
```

约 11 个用例，纯 CPU/mock，**单 GPU 即可**。

### 4.2 双进程 TransferEngine 冒烟（需 2× GPU）

Producer 在 `cuda:0`，Consumer 在 `cuda:1`，经 HTTP registry + Mooncake 拉取 tensor。

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export MOONCAKE_EC_PROTOCOL=tcp   # 无 RDMA 时用 tcp；有 RDMA 可改为 rdma

# 方式 A：直接跑脚本（打印 PASSED）
.venv/bin/python tests/v1/ec_connector/integration/test_ec_mooncake_transfer_e2e.py

# 方式 B：pytest
.venv/bin/python -m pytest tests/v1/ec_connector/integration/test_ec_mooncake_transfer_e2e.py -v
```

期望最后一行：`ECMooncake two-process transfer e2e: PASSED` 或 pytest `1 passed`。

检查 GPU 数量：

```bash
.venv/bin/python -c "import torch; print('cuda devices:', torch.cuda.device_count())"
```

### 4.3 KV Mooncake 单元测试（可选）

```bash
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_connector.py -v
```

### 4.4 全链路 EPD + Mooncake EC（重，需多 GPU + 大模型）

默认模型 `Qwen/Qwen2.5-VL-3B-Instruct`，需 **至少 2–3 张 GPU** 与较长时间。

```bash
# 文本 prompt 快速 sanity（比多模态快）
USE_MM_PROMPTS=0 ./tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh

# 完整多模态
./tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
```

常用环境变量：

```bash
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" \
GPU_SINGLE=0 GPU_E=1 GPU_PD=2 \
MOONCAKE_EC_PROTOCOL=tcp \
USE_MM_PROMPTS=0 \
./tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
```

### 4.5 通用 EPD 正确性（非 Mooncake EC 专用）

```bash
USE_MM_PROMPTS=0 ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

---

## 5. 推荐手测顺序（清单）

按顺序勾选，前面通过再跑后面：

| 步骤 | 命令 | GPU | 耗时（量级） |
|------|------|-----|----------------|
| ☐ 环境 | `bash scripts/dev-setup-minimal.sh` | 1 | 数分钟（含 pip） |
| ☐ L1 单元 | `pytest tests/test_logger.py::test_trace_function_call -v` | 0 | <5s |
| ☐ L2 离线 | `python scripts/smoke-inference.py` | 1 | ~30s |
| ☐ L3 HTTP | `vllm serve` + `curl /v1/completions` | 1 | serve ~20s，curl <1s |
| ☐ L4 EC 单测 | `pytest tests/v1/ec_connector/unit/test_ec_mooncake_connector.py -v` | 0 | ~10s |
| ☐ L5 EC e2e | `python tests/v1/ec_connector/integration/test_ec_mooncake_transfer_e2e.py` | 2 | ~1–3min |
| ☐ L6 EPD 全链路 | `run_epd_mooncake_ec_full_pipeline.sh` | 2–3 | 数十分钟 |

---

## 6. 常见问题

### `Network is unreachable` 拉模型失败

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新 `snapshot_download` 或启动 `vllm serve`。

### `ModuleNotFoundError: tblib`

```bash
uv pip install tblib
```

### `mooncake-transfer-engine is required`

```bash
uv pip install mooncake-transfer-engine
```

### EC e2e：`Need at least 2 CUDA devices`

单卡机器只能跑 **4.1 单元测试** 和 **第 2–3 节** 推理/HTTP；双卡及以上再跑 4.2。

### Chat 400 / chat template

换带 instruct/chat 模板的模型，或只用 `/v1/completions`。

### 端口占用

```bash
ss -tlnp | grep 8000
pkill -f "vllm serve"
```

### 查看 serve 日志

前台启动时直接看终端；后台可用：

```bash
# 若用 nohup
tail -f /tmp/vllm-serve.log
```

---

## 7. 相关文件

| 路径 | 用途 |
|------|------|
| [scripts/dev-setup-minimal.sh](../../scripts/dev-setup-minimal.sh) | 最小环境 + 单元 + 离线推理 |
| [scripts/smoke-inference.py](../../scripts/smoke-inference.py) | 单 prompt GPU 冒烟 |
| [examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py) | 官方离线示例 |
| [AGENTS.md](../../AGENTS.md) | 贡献规范、pytest、pre-commit |
| [openai_compatible_server.md](../serving/openai_compatible_server.md) | HTTP API 说明 |
| [mooncake_connector_usage.md](../features/mooncake_connector_usage.md) | KV Mooncake PD 分离 |

---

## 8. 记录手测结果（PR 模板参考）

```
环境: Linux, GPU xN, CUDA x.x, 分支 add-mooncake-ec-connector @ <short-sha>
HF_ENDPOINT: https://hf-mirror.com / (none)

- [ ] test_logger.py::test_trace_function_call
- [ ] scripts/smoke-inference.py
- [ ] vllm serve + curl /v1/completions
- [ ] test_ec_mooncake_connector.py (unit)
- [ ] test_ec_mooncake_transfer_e2e.py (2 GPU)
- [ ] run_epd_mooncake_ec_full_pipeline.sh (optional)

备注: ...
```
