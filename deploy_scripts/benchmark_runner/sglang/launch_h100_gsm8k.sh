#!/usr/bin/bash

# === Required Env Vars ===
# HF_TOKEN -- Hugging Face token with access to the model
# HF_HUB_CACHE_MOUNT -- Path to mount the Hugging Face hub cache directory
# MODEL -- Model name or path (e.g., Qwen/Qwen3-235B-A22B-Thinking-2507)
# TP -- Tensor parallelism degree (default: 8)
# PORT -- Port number for the server (default: 30000)
# CONFIG_FILE -- Path to config file (e.g., configs/gsm8k_qwen3_235b_a22b.yaml)
# OUTPUT_DIR -- Output directory for results (default: outputs/)
# EXPERT_MODE -- Expert distribution recorder mode (default: stat)
# IMAGE -- Docker image to use

HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/home/ubuntu/hf_hub_cache/}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
EXPERT_MODE="${EXPERT_MODE:-stat}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/}"
CONFIG_FILE="${CONFIG_FILE:-configs/gsm8k_qwen3_235b_a22b.yaml}"

server_name="bmk-server"
client_name="bmk-client"

CHUNKED_PREFILL_ARG=""
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
    echo "Fixed-length mode: setting chunked-prefill-size=${CHUNKED_PREFILL_SIZE} (target_input=${TARGET_INPUT} + 50)"
fi

# Launch server container
set -x
docker run --rm -d --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:/root/.cache/huggingface/ \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e MODEL -e TP -e PORT -e EXPERT_MODE \
-e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$IMAGE \
-c "python -m moe_cap.systems.sglang --model-path \${MODEL} --port \${PORT} --expert-distribution-recorder-mode \${EXPERT_MODE} --tp-size \${TP} ${CHUNKED_PREFILL_ARG}"

set +x
echo "Waiting for server to start..."
# Wait for server to be ready
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ "INFO:     Application startup complete" ]]; then
        echo "Server started successfully"
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

# Check if server container is still running
if ! docker ps --format "{{.Names}}" | grep -q "$server_name"; then
    echo "Server container launch failed."
    docker logs $server_name
    exit 1
fi

# Launch client container for benchmarking
set -x
docker run --rm --network=host --name=$client_name \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash \
$IMAGE \
-c "python -m moe_cap.runner.openai_api_profile --config-file $CONFIG_FILE --output_dir $OUTPUT_DIR --api-url http://localhost:${PORT}/v1/completions --backend sglang"

# Stop server container
docker stop $server_name