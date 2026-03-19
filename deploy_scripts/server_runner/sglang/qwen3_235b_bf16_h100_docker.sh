#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN -- Hugging Face token with access to the model
# HF_HUB_CACHE -- Path to the Hugging Face hub cache directory
# MODEL -- Model name or path (e.g., Qwen/Qwen3-235B-A22B-Thinking-2507)
# PORT -- Port number for the server (default: 30000)
# TP -- Tensor parallelism degree (default: 8)
# EXPERT_MODE -- Expert distribution recorder mode (default: stat)
# CONFIG_FILE -- (Optional) Config file; if fixed_length_mode, auto-sets chunked-prefill-size

CHUNKED_PREFILL_ARG=""
if [ -n "${CONFIG_FILE}" ] && [ -f "${CONFIG_FILE}" ]; then
    TARGET_INPUT=$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
if cfg.get('fixed_length_mode'):
    print(cfg.get('target_input_tokens', 0))
else:
    print(0)
" 2>/dev/null)
    if [ "$TARGET_INPUT" -gt 0 ] 2>/dev/null; then
        CHUNKED_PREFILL_SIZE=$((TARGET_INPUT + 50))
        CHUNKED_PREFILL_ARG="--chunked-prefill-size ${CHUNKED_PREFILL_SIZE}"
        echo "Fixed-length mode: chunked-prefill-size=${CHUNKED_PREFILL_SIZE}"
    fi
fi

set -x
python -m moe_cap.systems.sglang \
--model-path ${MODEL} \
--port ${PORT:-30000} \
--expert-distribution-recorder-mode ${EXPERT_MODE:-stat} \
--tp-size ${TP:-8} ${CHUNKED_PREFILL_ARG}