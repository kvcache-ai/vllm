#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Full Mooncake validation: tensor (EC+KV) + KV P/D + 1E+1P+1D EPD.
#
# Usage (repo root):
#   ./tests/v1/ec_connector/integration/run_full_mooncake_e2e.sh
#
# Env:
#   MODEL, HF_ENDPOINT, MOONCAKE_EC_PROTOCOL (default rdma)
#   GPU_* port overrides, USE_MM_PROMPTS, LOG_PATH, SKIP_TENSOR=1, SKIP_KV_PD=1

set -euo pipefail

# Ignore inherited SKIP_TENSOR=1 from the shell; set FORCE_SKIP_TENSOR=1 to skip tensor phases.
if [[ "${FORCE_SKIP_TENSOR:-0}" == "1" ]]; then
  export SKIP_TENSOR=1
else
  export SKIP_TENSOR=0
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1
export PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MOONCAKE_EC_PROTOCOL="${MOONCAKE_EC_PROTOCOL:-rdma}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
USE_MM_PROMPTS="${USE_MM_PROMPTS:-1}"
MM_FLAG=""
[[ "$USE_MM_PROMPTS" == "1" ]] && MM_FLAG="--use_mm_prompts"
# Text-only Mooncake EPD: short prompts + capped tokens so 1P+1D matches 1E+1P+1D.
EPD_MOONCAKE_FLAGS=""
if [[ "$USE_MM_PROMPTS" != "1" ]]; then
  EPD_MOONCAKE_FLAGS="--mooncake-full-e2e --max-tokens ${VLLM_EPD_MAX_TOKENS:-48}"
fi

LOG_PATH="${LOG_PATH:-/tmp/vllm_full_mooncake_e2e}"
BASELINE_FILE="${BASELINE_FILE:-${LOG_PATH}/baseline_single.json}"
BASELINE_PD_FILE="${BASELINE_PD_FILE:-${LOG_PATH}/baseline_1p1d_mooncake.json}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
RESULT_LOG="${LOG_PATH}/run_full_mooncake_e2e.log"

mkdir -p "$LOG_PATH"
exec > >(tee -a "$RESULT_LOG") 2>&1

GPU_SINGLE="${GPU_SINGLE:-0}"
GPU_E="${GPU_E:-1}"
GPU_P="${GPU_P:-2}"
GPU_D="${GPU_D:-3}"
# Dedicated GPUs for KV P+D smoke (avoid OOM on busy cards).
GPU_KV_P="${GPU_KV_P:-4}"
GPU_KV_D="${GPU_KV_D:-5}"

GPU_MEM_KV_PD="${GPU_MEM_KV_PD:-0.55}"
# A10 22GB: VL-3B baseline/PD need lower util than 0.75 to survive MM profile_run.
GPU_MEM_BASELINE="${GPU_MEM_BASELINE:-0.54}"
GPU_MEM_PD="${GPU_MEM_PD:-0.50}"
GPU_MEM_ENCODER="${GPU_MEM_ENCODER:-0.30}"
# Phase 2 KV smoke uses 2048; EPD/MM needs longer context (set EPD_MAX_MODEL_LEN).
MAX_MODEL_LEN_KV="${MAX_MODEL_LEN_KV:-2048}"
EPD_MAX_MODEL_LEN="${EPD_MAX_MODEL_LEN:-8192}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
ENDPOINT_PORT="${ENDPOINT_PORT:-10002}"
KV_PREFILL_PORT="${KV_PREFILL_PORT:-8010}"
KV_DECODE_PORT="${KV_DECODE_PORT:-8020}"
KV_PROXY_PORT="${KV_PROXY_PORT:-8011}"

EC_MOONCAKE_REGISTRY_PORT="${EC_MOONCAKE_REGISTRY_PORT:-19018}"
EC_REGISTRY_HOST="${EC_REGISTRY_HOST:-127.0.0.1}"
EC_REGISTRY_URL="http://${EC_REGISTRY_HOST}:${EC_MOONCAKE_REGISTRY_PORT}"
export EC_MOONCAKE_REGISTRY_PORT EC_REGISTRY_HOST MOONCAKE_EC_PROTOCOL EC_REGISTRY_URL

PY="${PY:-${GIT_ROOT}/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

_find_free_port() {
  "$PY" -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()"
}

# Always use project venv for serve (avoid system python3.13 / broken _C.abi3.so).
if [[ -x "${GIT_ROOT}/.venv/bin/vllm" ]]; then
  VLLM_SERVE=("${GIT_ROOT}/.venv/bin/vllm" serve)
else
  VLLM_SERVE=("$PY" -m vllm.entrypoints.cli.main serve)
fi

# sync_after_transfer + chunk size for RDMA reliability; skip verify_transfer_integrity
# in serve smoke (env/hash path hits cudaMemcpy issues on some CUDA builds).
KV_JSON_PRODUCER='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"sync_after_transfer":true,"max_transfer_bytes":262144}}'
KV_JSON_CONSUMER='{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"sync_after_transfer":true,"max_transfer_bytes":262144}}'

ENC_EC_JSON=$("$PY" - <<PY
import json, os
print(json.dumps({
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "rdma"),
        "registry_http_port": int(os.environ.get("EC_MOONCAKE_REGISTRY_PORT", "19018")),
    },
}, separators=(",", ":")))
PY
)

PREFILL_EC_JSON=$("$PY" - <<PY
import json, os
print(json.dumps({
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_connector_extra_config": {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "rdma"),
        "remote_registry_url": os.environ["EC_REGISTRY_URL"],
    },
}, separators=(",", ":")))
PY
)

wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
    until curl -sf -o /dev/null \"localhost:${port}/health\" \
      || curl -sf -o /dev/null \"localhost:${port}/v1/models\"; do
      sleep 2
    done
  "
}

wait_for_epd_proxy() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
    until curl -sf -o /dev/null \"localhost:${port}/health\"; do sleep 2; done
  "
}

wait_for_mooncake_proxy() {
  local port=$1
  sleep 3
  # Mooncake proxy has no GET /health; probe chat API (works for VL instruct models).
  timeout 600 bash -c "
    until curl -sf -o /dev/null -X POST \"localhost:${port}/v1/chat/completions\" \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0,\"seed\":42}'; do
      sleep 5
    done
  "
}

cleanup_instances() {
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  pkill -f "disagg_epd_proxy.py" 2>/dev/null || true
  pkill -f "mooncake_connector_proxy.py" 2>/dev/null || true
  pkill -f "toy_proxy_server.py" 2>/dev/null || true
  sleep 3
}

trap 'cleanup_instances; kill $(jobs -pr) 2>/dev/null || true' EXIT INT TERM

phase_header() {
  echo ""
  echo "################################################################"
  echo "# $1"
  echo "################################################################"
}

run_tensor_ec() {
  phase_header "PHASE 1a: EC tensor verify (RDMA SHA256)"
  export MOONCAKE_EC_PROTOCOL="${MOONCAKE_EC_PROTOCOL}"
  "$PY" "${GIT_ROOT}/scripts/verify_ec_mooncake_rdma.py"

  phase_header "PHASE 1b: EC two-process e2e (pytest)"
  "$PY" -m pytest \
    "${GIT_ROOT}/tests/v1/ec_connector/integration/test_ec_mooncake_transfer_e2e.py" -v
}

run_tensor_kv_unit() {
  phase_header "PHASE 1c: KV Mooncake unit (transfer ptr + metadata)"
  "$PY" -m pytest \
    "${GIT_ROOT}/tests/v1/kv_connector/unit/test_mooncake_connector.py" -v \
    --tb=short -q \
    -p pytest_asyncio
}

run_kv_pd_smoke() {
  phase_header "PHASE 2: KV Mooncake P+D smoke (sync_after_transfer + max_transfer_bytes)"
  cleanup_instances

  local bootstrap_port
  bootstrap_port=$(_find_free_port)
  export VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port"
  echo "Mooncake bootstrap port (KV P+D): $bootstrap_port"
  echo "GPU_MEM_KV_PD=${GPU_MEM_KV_PD} (GPUs ${GPU_KV_P}/${GPU_KV_D})"
  export VLLM_MOONCAKE_VERIFY_TRANSFER_INTEGRITY=0
  export VLLM_MOONCAKE_SYNC_AFTER_TRANSFER=1

  CUDA_VISIBLE_DEVICES="$GPU_KV_P" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$KV_PREFILL_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_KV_PD" \
    --max-model-len "$MAX_MODEL_LEN_KV" \
    --kv-transfer-config "$KV_JSON_PRODUCER" \
    >"${LOG_PATH}/kv_prefill.log" 2>&1 &
  local pid_p=$!

  CUDA_VISIBLE_DEVICES="$GPU_KV_D" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$KV_DECODE_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_KV_PD" \
    --max-model-len "$MAX_MODEL_LEN_KV" \
    --kv-transfer-config "$KV_JSON_CONSUMER" \
    >"${LOG_PATH}/kv_decode.log" 2>&1 &
  local pid_d=$!

  wait_for_server "$KV_PREFILL_PORT"
  wait_for_server "$KV_DECODE_PORT"

  local proxy_dir="${GIT_ROOT}/examples/online_serving/disaggregated_serving/mooncake_connector"
  "$PY" "${proxy_dir}/mooncake_connector_proxy.py" \
    --prefill "http://127.0.0.1:${KV_PREFILL_PORT}" "$bootstrap_port" \
    --decode "http://127.0.0.1:${KV_DECODE_PORT}" \
    --port "$KV_PROXY_PORT" \
    >"${LOG_PATH}/kv_proxy.log" 2>&1 &
  local pid_proxy=$!

  wait_for_mooncake_proxy "$KV_PROXY_PORT"

  "$PY" - <<PY
import openai
client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:${KV_PROXY_PORT}/v1")
r = client.chat.completions.create(
    model="${MODEL}",
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=16,
    temperature=0.0,
    seed=42,
)
text = r.choices[0].message.content
print("KV P+D chat completion:", repr(text))
assert text and len(text.strip()) > 0, "empty completion"
PY

  kill "$pid_p" "$pid_d" "$pid_proxy" 2>/dev/null || true
  sleep 2
  cleanup_instances
  echo "PHASE 2 PASS: KV Mooncake P+D + sync/chunk flags"
}

prefetch_model() {
  phase_header "PHASE 3: Prefetch model ${MODEL}"
  "$PY" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL}")
print("cached:", p)
PY
}

run_baseline_single() {
  phase_header "PHASE 4: Baseline single instance"
  cleanup_instances
  echo "GPU_MEM_BASELINE=${GPU_MEM_BASELINE} (GPU ${GPU_SINGLE}) max_model_len=${EPD_MAX_MODEL_LEN}"
  CUDA_VISIBLE_DEVICES="$GPU_SINGLE" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$ENDPOINT_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_BASELINE" \
    --max-model-len "$EPD_MAX_MODEL_LEN" \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    >"${LOG_PATH}/baseline_single.log" 2>&1 &
  local pid=$!
  wait_for_server "$ENDPOINT_PORT"
  "$PY" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:${ENDPOINT_PORT}" \
    --model_name "$MODEL" \
    --mode baseline \
    --baseline_file "$BASELINE_FILE" \
    $MM_FLAG
  kill "$pid" 2>/dev/null || true
  cleanup_instances
}

run_baseline_1p1d_mooncake() {
  phase_header "PHASE 5: Baseline 1P+1D (Mooncake KV)"
  cleanup_instances
  local bootstrap_port
  bootstrap_port=$(_find_free_port)
  export VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port"
  echo "Mooncake bootstrap port (1P+1D): $bootstrap_port"

  echo "GPU_MEM_PD=${GPU_MEM_PD} (GPUs ${GPU_P}/${GPU_D}) max_model_len=${EPD_MAX_MODEL_LEN}"
  CUDA_VISIBLE_DEVICES="$GPU_P" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_PD" \
    --max-model-len "$EPD_MAX_MODEL_LEN" \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --kv-transfer-config "$KV_JSON_PRODUCER" \
    >"${LOG_PATH}/b15_p.log" 2>&1 &
  local pid_p=$!

  CUDA_VISIBLE_DEVICES="$GPU_D" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$DECODE_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_PD" \
    --max-model-len "$EPD_MAX_MODEL_LEN" \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --kv-transfer-config "$KV_JSON_CONSUMER" \
    >"${LOG_PATH}/b15_d.log" 2>&1 &
  local pid_d=$!

  wait_for_server "$PREFILL_PORT"
  wait_for_server "$DECODE_PORT"

  local proxy_dir="${GIT_ROOT}/examples/online_serving/disaggregated_serving/mooncake_connector"
  "$PY" "${proxy_dir}/mooncake_connector_proxy.py" \
    --prefill "http://127.0.0.1:${PREFILL_PORT}" "$bootstrap_port" \
    --decode "http://127.0.0.1:${DECODE_PORT}" \
    --port "$ENDPOINT_PORT" \
    >"${LOG_PATH}/b15_proxy.log" 2>&1 &
  local pid_proxy=$!

  wait_for_mooncake_proxy "$ENDPOINT_PORT"

  "$PY" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:${ENDPOINT_PORT}" \
    --model_name "$MODEL" \
    --mode baseline_pd \
    --baseline_file "$BASELINE_PD_FILE" \
    $MM_FLAG $EPD_MOONCAKE_FLAGS

  kill "$pid_p" "$pid_d" "$pid_proxy" 2>/dev/null || true
  cleanup_instances
}

run_epd_1e1p1d_mooncake() {
  phase_header "PHASE 6: 1E+1P+1D (ECMooncake + Mooncake KV)"
  cleanup_instances
  [[ -f "$BASELINE_PD_FILE" ]] || { echo "Missing $BASELINE_PD_FILE"; exit 1; }

  local bootstrap_port
  bootstrap_port=$(_find_free_port)
  export VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port"
  echo "Mooncake bootstrap port (1E+1P+1D): $bootstrap_port"
  declare -a PIDS=()

  echo "GPU_MEM_ENCODER=${GPU_MEM_ENCODER} GPU_MEM_PD=${GPU_MEM_PD} (E/P/D=${GPU_E}/${GPU_P}/${GPU_D})"
  CUDA_VISIBLE_DEVICES="$GPU_E" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_ENCODER" \
    --max-model-len "$EPD_MAX_MODEL_LEN" \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "$ENC_EC_JSON" \
    >"${LOG_PATH}/1e1p1d_encoder.log" 2>&1 &
  PIDS+=($!)

  CUDA_VISIBLE_DEVICES="$GPU_P" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_PD" \
    --max-model-len "$EPD_MAX_MODEL_LEN" \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "$PREFILL_EC_JSON" \
    --kv-transfer-config "$KV_JSON_PRODUCER" \
    >"${LOG_PATH}/1e1p1d_prefill.log" 2>&1 &
  PIDS+=($!)

  CUDA_VISIBLE_DEVICES="$GPU_D" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$DECODE_PORT" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_PD" \
    --max-model-len "$EPD_MAX_MODEL_LEN" \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --kv-transfer-config "$KV_JSON_CONSUMER" \
    >"${LOG_PATH}/1e1p1d_decode.log" 2>&1 &
  PIDS+=($!)

  wait_for_server "$ENCODE_PORT"
  wait_for_server "$PREFILL_PORT"
  wait_for_server "$DECODE_PORT"

  "$PY" "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$ENDPOINT_PORT" \
    --encode-servers-urls "http://localhost:${ENCODE_PORT}" \
    --prefill-servers-urls "http://localhost:${PREFILL_PORT}" \
    --decode-servers-urls "http://localhost:${DECODE_PORT}" \
    >"${LOG_PATH}/1e1p1d_proxy.log" 2>&1 &
  PIDS+=($!)

  wait_for_epd_proxy "$ENDPOINT_PORT"

  "$PY" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:${ENDPOINT_PORT}" \
    --model_name "$MODEL" \
    --mode disagg \
    --baseline_file "$BASELINE_PD_FILE" \
    $MM_FLAG $EPD_MOONCAKE_FLAGS

  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  cleanup_instances
}

install_test_deps() {
  phase_header "PHASE 0: Test dependencies"
  if command -v uv &>/dev/null; then
    uv pip install pytest-asyncio openai tblib mooncake-transfer-engine pyzmq httpx fastapi uvicorn
  else
    "$PY" -m pip install pytest-asyncio openai tblib mooncake-transfer-engine pyzmq httpx fastapi uvicorn
  fi
  "$PY" -c "import pytest_asyncio; print('pytest-asyncio ok')"
}

main() {
  echo "Full Mooncake E2E @ $(date -Iseconds)"
  echo "MODEL=$MODEL MOONCAKE_EC_PROTOCOL=$MOONCAKE_EC_PROTOCOL"
  echo "LOG_PATH=$LOG_PATH RUN_FROM_PHASE=${RUN_FROM_PHASE:-1}"

  install_test_deps
  "$PY" -c "import vllm; import vllm._C; print('venv ok', vllm.__version__)"

  prefetch_model

  # RUN_FROM_PHASE=N: run phases 4–6 when N=4; run 5–6 when N=5; only 6 when N=6.
  # (Phases 1–2 run when N<=1 or N<=2; see conditions below.)
  local from="${RUN_FROM_PHASE:-${START_PHASE:-1}}"
  if [[ "$from" -le 1 ]]; then
    [[ "${SKIP_TENSOR:-0}" != "1" ]] && run_tensor_ec && run_tensor_kv_unit
  fi
  if [[ "$from" -le 2 ]]; then
    [[ "${SKIP_KV_PD:-0}" != "1" ]] && run_kv_pd_smoke
  fi
  if [[ "$from" -le 4 ]]; then
    run_baseline_single
  fi
  if [[ "$from" -le 5 ]]; then
    run_baseline_1p1d_mooncake
  fi
  if [[ "$from" -le 6 ]]; then
    run_epd_1e1p1d_mooncake
  fi

  phase_header "ALL PHASES PASSED"
  echo "Results: $RESULT_LOG"
}

main "$@"
