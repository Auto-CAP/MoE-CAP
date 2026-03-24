#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN -- Hugging Face token with access to the model
# HF_HUB_CACHE -- Path to the Hugging Face hub cache directory
# MODEL -- Model name or path (e.g., Qwen/Qwen3-235B-A22B-Thinking-2507)
# PORT -- Port number for the server (default: 30000)
# TP -- Tensor parallelism degree (default: 8)
# EXPERT_MODE -- Expert distribution recorder mode (default: stat)
set -x
python -m moe_cap.systems.sglang \
--model-path ${MODEL} \
--port ${PORT:-30000} \
--expert-distribution-recorder-mode ${EXPERT_MODE:-stat} \
--tp-size ${TP:-8}
