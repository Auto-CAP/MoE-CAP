#!/bin/bash
# Automated benchmark runner for all model/batch size/config combinations
# This script will:
# 1. Loop through all batch sizes (1, 32, 64, 128)
# 2. Loop through all models (Qwen1.5-MoE, DeepSeek-V2-Lite)
# 3. For each combination, start the server, run tests, then kill the server

set -e

# Configuration
PORT=30000
TP=8  # Tensor parallelism: 1 for single GPU, 8 for 8 GPUs
BATCH_SIZES=(128)
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/root/MoE-CAP/expert_records

# Model and config mapping
declare -A MODEL_CONFIGS
MODEL_CONFIGS["Qwen/Qwen1.5-MoE-A2.7B-Chat"]="qwen1.5"
MODEL_CONFIGS["deepseek-ai/DeepSeek-V2-Lite-Chat"]="dsv2_lite"

# Task types (4k and 13k)
TASKS=("4k_1k" "13k_1k")

# Results directory
RESULTS_DIR="/root/MoE-CAP/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to wait for server to be ready
wait_for_server() {
    local max_attempts=120  # 10 minutes timeout
    local attempt=0
    log "Waiting for server to be ready on port $PORT..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 || \
           curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            log "Server is ready!"
            sleep 5  # Give it a few more seconds to fully initialize
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    
    log "ERROR: Server failed to start within timeout"
    return 1
}

# Function to kill any existing server on the port
kill_server() {
    log "Killing any existing server on port $PORT..."
    # Kill by port
    fuser -k $PORT/tcp 2>/dev/null || true
    # Also try to kill python processes related to sglang
    pkill -f "moe_cap.systems.sglang" 2>/dev/null || true
    pkill -f "sglang" 2>/dev/null || true
    sleep 10  # Wait for processes to fully terminate
}

# Function to start the server
start_server() {
    local model=$1
    local batch_size=$2
    
    log "Starting server with model=$model, batch_size=$batch_size, tp=$TP"
    
    python -m moe_cap.systems.sglang \
        --model-path "$model" \
        --port $PORT \
        --expert-distribution-recorder-mode stat \
        --tp-size $TP \
        --max-running-requests $batch_size \
        > "$RESULTS_DIR/server_${model//\//_}_bs${batch_size}.log" 2>&1 &
    
    SERVER_PID=$!
    log "Server started with PID $SERVER_PID"
}

# Function to run benchmark
run_benchmark() {
    local config_file=$1
    local batch_size=$2
    local output_prefix=$3
    
    log "Running benchmark with config=$config_file, batch_size=$batch_size"
    
    python -m moe_cap.runner.openai_api_profile \
        --config-file "$config_file" \
        --api-url "http://localhost:$PORT/v1/completions" \
        --backend sglang \
        --ignore-eos \
        --server-batch-size $batch_size \
        2>&1 | tee "$RESULTS_DIR/${output_prefix}.log"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log "Benchmark completed successfully: $config_file"
    else
        log "WARNING: Benchmark failed with exit code $exit_code: $config_file"
    fi
    
    return $exit_code
}

# Main execution loop
main() {
    log "=========================================="
    log "Starting benchmark suite"
    log "Batch sizes: ${BATCH_SIZES[*]}"
    log "Models: ${!MODEL_CONFIGS[*]}"
    log "Tasks: ${TASKS[*]}"
    log "=========================================="
    
    local total_runs=$((${#BATCH_SIZES[@]} * ${#MODEL_CONFIGS[@]} * ${#TASKS[@]}))
    local current_run=0
    local failed_runs=0
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        for model in "${!MODEL_CONFIGS[@]}"; do
            config_suffix="${MODEL_CONFIGS[$model]}"
            
            log "------------------------------------------"
            log "Configuration: model=$model, batch_size=$batch_size"
            log "------------------------------------------"
            
            # Kill any existing server
            kill_server
            
            # Start new server
            start_server "$model" "$batch_size"
            
            # Wait for server to be ready
            if ! wait_for_server; then
                log "ERROR: Failed to start server for model=$model, batch_size=$batch_size"
                kill_server
                failed_runs=$((failed_runs + ${#TASKS[@]}))
                continue
            fi
            
            # Run all tasks for this model/batch_size combination
            for task in "${TASKS[@]}"; do
                current_run=$((current_run + 1))
                # Use special config with num_samples=256 for batch_size=128
                if [ $batch_size -eq 128 ]; then
                    config_file="configs/fixed_${task}_${config_suffix}_bs128.yaml"
                else
                    config_file="configs/fixed_${task}_${config_suffix}.yaml"
                fi
                output_prefix="result_${config_suffix}_${task}_bs${batch_size}"
                
                log "[$current_run/$total_runs] Running: $config_file with batch_size=$batch_size"
                
                if ! run_benchmark "$config_file" "$batch_size" "$output_prefix"; then
                    failed_runs=$((failed_runs + 1))
                fi
                
                # Small delay between runs
                sleep 5
            done
            
            # Kill server before next model/batch_size combination
            kill_server
        done
    done
    
    log "=========================================="
    log "Benchmark suite completed!"
    log "Total runs: $total_runs"
    log "Failed runs: $failed_runs"
    log "Results saved to: $RESULTS_DIR"
    log "=========================================="
}

# Run main function
main "$@"
