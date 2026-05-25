#!/usr/bin/env bash
# Minimal vLLM dev env: editable install + unit smoke + GPU inference smoke.
set -euo pipefail
cd "$(dirname "$0")/.."

# Use hf-mirror when huggingface.co is unreachable (common in CN). Unset to use the default hub.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SMOKE_MODEL="${SMOKE_MODEL:-facebook/opt-125m}"

echo "==> Git: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

if [[ ! -d .venv ]]; then
  uv venv --python 3.12 --seed --managed-python
fi

echo "==> Installing vLLM (precompiled)..."
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

echo "==> Installing minimal test deps..."
uv pip install pytest tblib

echo "==> Import check..."
.venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__); print('cuda', torch.cuda.is_available())"

echo "==> Minimal unit test..."
.venv/bin/python -m pytest tests/test_logger.py::test_trace_function_call -q

echo "==> Prefetch smoke model (${SMOKE_MODEL})..."
.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("${SMOKE_MODEL}")
print("model cache:", path)
PY

echo "==> GPU inference smoke..."
.venv/bin/python scripts/smoke-inference.py

echo "==> Done."
echo "    Offline:  .venv/bin/python examples/basic/offline_inference/basic.py"
echo "    HTTP:     see docs/contributing/local_manual_testing.md (vllm serve + curl)"
