# MoE-CAP Benchmarking Guide

This guide describes how to run fixed-length performance benchmarks for Mixture-of-Experts models across different hardware configurations.

## Overview

We benchmark models on two tasks:
- **4K-1K**: 4000 input tokens, 1000 output tokens (using GSM8K dataset)
- **13K-1K**: 13000 input tokens, 1000 output tokens (using LongBench V2 dataset)

## Hardware Configurations

| Hardware | Description |
|----------|-------------|
| 1xH100 | Single NVIDIA H100-SXM (80GB) |
| 8xH100 | 8x NVIDIA H100-SXM with NVLink |
| 1xH200 | Single NVIDIA H200 (141GB) |
| 8xH200 | 8x NVIDIA H200 with NVLink |
| 1xA100 | Single NVIDIA A100-80G-SXM4 |
| 8xA100 | 8x NVIDIA A100-80G with NVLink |

## Models

| Model | HuggingFace ID | Size |
|-------|----------------|------|
| Qwen1.5-MoE | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 14.3B total, 2.7B activated |
| DeepSeek-V2-Lite | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 16B total, 2.4B activated |
| Mixtral-8x7B | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 46.7B total, 12.9B activated |
| Mixtral-8x22B | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 141B total, 39B activated |
| DeepSeek-R1 | `deepseek-ai/DeepSeek-R1` | 671B total, 37B activated |

## Batch Sizes

- **1**: Single request (latency-focused)
- **32**: Medium batch
- **64**: Large batch
- **128**: Maximum batch

---

## Configuration Files

### 4K-1K Configuration (`configs/fixed_4k_1k_<model>.yaml`)

```yaml
# Fixed-length benchmarking: 4K input tokens, 1K output tokens
# This mode is for pure performance benchmarking (TPOT, throughput) without accuracy evaluation

dataset_names: ["gsm8k"]

# No accuracy metrics for fixed-length benchmarking
metrics: []

model_id: <MODEL_HF_ID>

# Enable fixed-length mode
fixed_length_mode: true
target_input_tokens: 4000   # 4K input tokens
target_output_tokens: 1000  # 1K output tokens
num_samples: 100            # Number of benchmark samples
```

### 13K-1K Configuration (`configs/fixed_13k_1k_<model>.yaml`)

```yaml
# Fixed-length benchmarking: 13K input tokens, 1K output tokens
# This mode is for pure performance benchmarking (TPOT, throughput) without accuracy evaluation

dataset_names: ["longbench_v2"]

# No accuracy metrics for fixed-length benchmarking
metrics: []

model_id: <MODEL_HF_ID>

# Enable fixed-length mode
fixed_length_mode: true
target_input_tokens: 13000  # 13K input tokens
target_output_tokens: 1000  # 1K output tokens
num_samples: 100            # Number of benchmark samples
```

---

## Running Benchmarks

We support two inference backends: **SGLang** and **vLLM**. Both use the same benchmark configuration and runner.

### Environment Setup

> ⚠️ **Important**: SGLang and vLLM require **separate conda environments** due to dependency conflicts.

```bash
# Create SGLang environment
conda create -n sglang python=3.10 -y
conda activate sglang
pip install sglang==0.5.8  # Or install from source

# Create vLLM environment (separate!)
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm==0.11.0  # Use version 0.11.0
```

---

### Option A: Using SGLang

#### Step 1: Launch the SGLang Server

```bash
conda activate sglang

export MODEL="<MODEL_HF_ID>"
export PORT=30000
export TP=<TENSOR_PARALLELISM>  # 1 for single GPU, 8 for 8 GPUs
export BATCH_SIZE=<BATCH_SIZE>  # 1, 32, 64, or 128

# Launch SGLang server
python -m moe_cap.systems.sglang \
  --model-path $MODEL \
  --port $PORT \
  --expert-distribution-recorder-mode stat \
  --tp-size $TP \
  --max-running-requests $BATCH_SIZE
```

#### Step 2: Run the Benchmark (SGLang)

```bash
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_<model>.yaml \
  --api-url http://localhost:30000/v1/completions \
  --backend sglang \
  --ignore-eos \
  --server-batch-size $BATCH_SIZE
```

---

### Option B: Using vLLM

> ⚠️ **Note**: Use a separate conda environment with `vllm==0.11.0`

#### Step 1: Launch the vLLM Server

```bash
conda activate vllm

export MODEL="<MODEL_HF_ID>"
export PORT=8000
export TP=<TENSOR_PARALLELISM>  # 1 for single GPU, 8 for 8 GPUs
export BATCH_SIZE=<BATCH_SIZE>  # 1, 32, 64, or 128

# Launch vLLM server
python -m moe_cap.systems.vllm \
  --model $MODEL \
  --port $PORT \
  --host 0.0.0.0 \
  --tensor-parallel-size $TP \
  --enable-expert-distribution-metrics \
  --max-num-batched-tokens 131072 \
  --max-num-seqs $BATCH_SIZE
```

#### Step 2: Run the Benchmark (vLLM)

```bash
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_<model>.yaml \
  --api-url http://0.0.0.0:8000/v1/completions \
  --backend vllm \
  --ignore-eos \
  --server-batch-size $BATCH_SIZE
```

---

## Experiment Matrix

### ✅ Completed Experiments

| Model | Hardware | Task | Batch Sizes | Status |
|-------|----------|------|-------------|--------|
| Qwen1.5-MoE | 1xH100 | 4K-1K | 1, 32 | ✅ Done |
| Qwen1.5-MoE | 1xH100 | 13K-1K | 1 | ✅ Done |
| DeepSeek-V2-Lite | 1xH100 | 4K-1K | 1, 32 | ✅ Done |
| DeepSeek-V2-Lite | 1xH100 | 13K-1K | 1 | ✅ Done |

### 📋 Pending Experiments

#### Qwen1.5-MoE & DeepSeek-V2-Lite (Small Models)

| Hardware | Task | Batch Sizes |
|----------|------|-------------|
| 1xA100 | 4K-1K | 1, 32 |
| 1xA100 | 13K-1K | 1 |
| 1xH200 | 4K-1K | 1, 32, 64, 128 |
| 1xH200 | 13K-1K | 1, 32, 64 |
| 8xH100 | 4K-1K | 1, 32, 64, 128 |
| 8xH100 | 13K-1K | 1, 32, 64, 128 |
| 8xA100 | 4K-1K | 1, 32, 64, 128 |
| 8xA100 | 13K-1K | 1, 32, 64, 128 |
| 8xH200 | 4K-1K | 1, 32, 64, 128 |
| 8xH200 | 13K-1K | 1, 32, 64, 128 |

#### Mixtral-8x7B, Mixtral-8x22B (Large Models - 8 GPUs Only)

| Hardware | Task | Batch Sizes |
|----------|------|-------------|
| 8xH100 | 4K-1K | 1, 32, 64, 128 |
| 8xH100 | 13K-1K | 1, 32, 64, 128 |
| 8xA100 | 4K-1K | 1, 32, 64, 128 |
| 8xA100 | 13K-1K | 1, 32, 64, 128 |
| 8xH200 | 4K-1K | 1, 32, 64, 128 |
| 8xH200 | 13K-1K | 1, 32, 64, 128 |

#### DeepSeek-R1 
| Hardware | Task | Batch Sizes |
|----------|------|-------------|
| 8xH200 | 4K-1K | 1, 32, 64, 128 |
| 8xH200 | 13K-1K | 1, 32, 64, 128 |

---

## Quick Reference Scripts

### SGLang Examples

#### Qwen1.5-MoE on 1xA100 (4K-1K, BS=1) - SGLang

```bash
# Terminal 1: Launch server
conda activate sglang
export MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat"
python -m moe_cap.systems.sglang \
  --model-path $MODEL \
  --port 30000 \
  --expert-distribution-recorder-mode stat \
  --tp-size 1 \
  --max-running-requests 1

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_qwen1_5.yaml \
  --api-url http://localhost:30000/v1/completions \
  --backend sglang \
  --ignore-eos \
  --server-batch-size 1
```

#### DeepSeek-V2-Lite on 8xH100 (13K-1K, BS=64) - SGLang

```bash
# Terminal 1: Launch server
conda activate sglang
export MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
python -m moe_cap.systems.sglang \
  --model-path $MODEL \
  --port 30000 \
  --expert-distribution-recorder-mode stat \
  --tp-size 8 \
  --max-running-requests 64

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_13k_1k_dsv2_lite.yaml \
  --api-url http://localhost:30000/v1/completions \
  --backend sglang \
  --ignore-eos \
  --server-batch-size 64
```

---

### vLLM Examples

> ⚠️ **Note**: Use separate conda environment with `vllm==0.11.0`

#### Qwen1.5-MoE on 1xH100 (4K-1K, BS=1) - vLLM

```bash
# Terminal 1: Launch server
conda activate vllm
export MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat"
python -m moe_cap.systems.vllm \
  --model $MODEL \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --enable-expert-distribution-metrics \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 1

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_qwen1_5.yaml \
  --api-url http://0.0.0.0:8000/v1/completions \
  --backend vllm \
  --ignore-eos \
  --server-batch-size 1
```

#### DeepSeek-V2-Lite on 1xH100 (4K-1K, BS=32) - vLLM

```bash
# Terminal 1: Launch server
conda activate vllm
export MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
python -m moe_cap.systems.vllm \
  --model $MODEL \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --enable-expert-distribution-metrics \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 32

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_dsv2_lite.yaml \
  --api-url http://0.0.0.0:8000/v1/completions \
  --backend vllm \
  --ignore-eos \
  --server-batch-size 32
```

#### Qwen1.5-MoE on 1xH100 (13K-1K, BS=1) - vLLM

```bash
# Terminal 1: Launch server
conda activate vllm
export MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat"
python -m moe_cap.systems.vllm \
  --model $MODEL \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --enable-expert-distribution-metrics \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 1

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_13k_1k_qwen1_5.yaml \
  --api-url http://0.0.0.0:8000/v1/completions \
  --backend vllm \
  --ignore-eos \
  --server-batch-size 1
```

---

### Large Model Examples (8 GPUs)

#### Mixtral-8x22B on 8xH200 (4K-1K, BS=128) - SGLang

```bash
# Terminal 1: Launch server
conda activate sglang
export MODEL="mistralai/Mixtral-8x22B-Instruct-v0.1"
python -m moe_cap.systems.sglang \
  --model-path $MODEL \
  --port 30000 \
  --expert-distribution-recorder-mode stat \
  --tp-size 8 \
  --max-running-requests 128

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_mixtral_8x22b.yaml \
  --api-url http://localhost:30000/v1/completions \
  --backend sglang \
  --ignore-eos \
  --server-batch-size 128
```

#### DeepSeek-R1 on 8xH200 (4K-1K, BS=32) - SGLang

```bash
# Terminal 1: Launch server
conda activate sglang
export MODEL="deepseek-ai/DeepSeek-R1"
python -m moe_cap.systems.sglang \
  --model-path $MODEL \
  --port 30000 \
  --expert-distribution-recorder-mode stat \
  --tp-size 8 \
  --max-running-requests 32

# Terminal 2: Run benchmark
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/fixed_4k_1k_deepseek_r1.yaml \
  --api-url http://localhost:30000/v1/completions \
  --backend sglang \
  --ignore-eos \
  --server-batch-size 32
```

---

## Output Metrics

The benchmark will output:
- **TTFT (Time To First Token)**: Latency until first token is generated (ms)
- **TPOT (Time Per Output Token)**: Average time per output token (ms)
- **Throughput**: Tokens per second
- **Total time**: End-to-end benchmark duration
- **server_batch_size**: The server's max-running-requests setting (recorded via `--server-batch-size`)

---

## Notes

1. **Environment Setup**:
   - SGLang and vLLM require **separate conda environments**
   - vLLM: Use `pip install vllm==0.11.0` (specific version required)
   - SGLang: Use `pip install sglang[all]` or install from source

2. **Memory Requirements**:
   - Qwen1.5-MoE, DeepSeek-V2-Lite: Can run on single GPU
   - Mixtral-8x7B: Requires 8 GPUs for full precision
   - Mixtral-8x22B: Requires 8 GPUs with high memory
   - DeepSeek-R1: Requires 8 GPUs with maximum memory

3. **Tensor Parallelism**:
   - SGLang: Use `--tp-size 1` for single GPU, `--tp-size 8` for 8 GPUs
   - vLLM: Use `--tensor-parallel-size 1` for single GPU, `--tensor-parallel-size 8` for 8 GPUs

4. **Batch Size Control**:
   - SGLang: `--max-running-requests $BATCH_SIZE`
   - vLLM: `--max-num-seqs $BATCH_SIZE`
   - Start with smaller batch sizes if OOM occurs
   - Larger batch sizes improve throughput but increase latency

5. **Port Configuration**:
   - SGLang default: `30000`
   - vLLM default: `8000`
   - Make sure the `--api-url` in runner matches the server port

6. **13K-1K Context**:
   - Requires more GPU memory than 4K-1K
   - Some batch sizes may not be achievable on certain hardware
