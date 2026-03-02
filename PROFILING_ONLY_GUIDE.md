# MoE-CAP: Profiling-Only Mode

Run MoE-CAP to collect **TTFT / TPOT / Throughput / Cost** metrics only, without recording expert activation distributions.

## When to use this mode

- You want pure performance benchmarking (latency, throughput)
- You don't need expert activation analysis
- You are running a non-MoE model or a model whose expert layers are not yet supported
- You want faster benchmark runs with lower overhead

---

## Quick Start

### Via config file

**1. Create a config YAML**

```yaml
# configs/my_profile_only.yaml
model_id: Qwen/Qwen3-30B-A3B
dataset_names: ["gsm8k"]
metrics: []                   # skip accuracy evaluation

fixed_length_mode: true       # fixed-length input/output for controlled benchmarking
target_input_tokens: 4096
target_output_tokens: 1024
num_samples: 100

profiling_only: true          # disable expert distribution recording
precision: bfloat16
```

**2. Start the inference server**

vLLM:
```bash
python -m moe_cap.systems.vllm \
    --model Qwen/Qwen3-30B-A3B \
    --port 8000 \
    --tensor-parallel-size 4
```

SGLang (must set env var to skip expert hooks):
```bash
MOE_CAP_PROFILING_ONLY=1 python -m moe_cap.systems.sglang \
    --model-path Qwen/Qwen3-30B-A3B \
    --port 30000 \
    --tp-size 4
```

**3. Run the benchmark**

```bash
python -m moe_cap.runner.openai_api_profile \
    --config-file configs/my_profile_only.yaml \
    --api-url http://localhost:8000/v1/completions \
    --output_dir outputs/profile_only/
```

---

### Via CLI (no config file)

```bash
python -m moe_cap.runner.openai_api_profile \
    --model_name Qwen/Qwen3-30B-A3B \
    --datasets gsm8k \
    --metrics \
    --fixed-length-mode \
    --target-input-tokens 4096 \
    --target-output-tokens 1024 \
    --num-samples 100 \
    --api-url http://localhost:8000/v1/completions \
    --output_dir outputs/profile_only/ \
    --profiling-only
```

---

### Using a preset config

Several preset configs for 4K/13K input lengths are included under `configs/`. These already set `metrics: []` and `fixed_length_mode: true`:

```bash
python -m moe_cap.runner.openai_api_profile \
    --config-file configs/fixed_4k_1k_qwen1.5.yaml \
    --api-url http://localhost:8000/v1/completions \
    --profiling-only
```

---

## Key options

| Option | Config field | Description |
|--------|-------------|-------------|
| `--profiling-only` | `profiling_only: true` | Skip expert distribution recording; only collect TTFT/TPOT/throughput |
| `--fixed-length-mode` | `fixed_length_mode: true` | Pad/truncate inputs to a fixed token length; forces `ignore_eos` on |
| `--target-input-tokens N` | `target_input_tokens: N` | Fixed input token length (requires `fixed_length_mode`) |
| `--target-output-tokens N` | `target_output_tokens: N` | Fixed output token length (requires `fixed_length_mode`) |
| `--num-samples N` | `num_samples: N` | Number of requests to send |
| `--metrics` (empty) | `metrics: []` | Skip accuracy metric computation |
| `--server-batch-size N` | — | Number of concurrent requests; omit to send all at once |
| `--backend` | — | `vllm`, `sglang`, or `auto` (default: `auto`) |

---

## Server-side notes

### vLLM

No extra setup is needed. When `--profiling-only` is set, the client skips calls to
`/start_batch_recording`, `/stop_batch_recording`, and `/dump_batch_recording`.
The server can be launched as a plain vLLM server (with or without `moe_cap.systems.vllm`).

### SGLang

Set the environment variable **before** starting the server:

```bash
MOE_CAP_PROFILING_ONLY=1 python -m moe_cap.systems.sglang ...
```

Without this variable, the server will still attempt to hook expert layers even if the
client runs in profiling-only mode.

---

## Output

Results are written to `<output_dir>/<org>/<model>/`:

```
outputs/profile_only/
└── Qwen/
    └── Qwen3-30B-A3B/
        ├── cap_metrics_gsm8k_20260302_120000.json   # summary metrics
        └── detailed_results_gsm8k.jsonl             # per-request details
```

**Summary JSON fields (selected):**

| Field | Description |
|-------|-------------|
| `prefill_tpot_ms` | Average prefill time per token (ms) |
| `decode_tpot_ms` | Average decode time per output token (ms) |
| `e2e_s` | Total wall-clock time (seconds) |
| `cost` | Estimated GPU cost |
| `gpu_type` | Detected GPU type and count |
| `profiling_only` | `true` — confirms expert recording was skipped |
| `total_requests` | Total requests sent |
| `successful_requests` | Requests that completed successfully |

---

## Fallback behavior

If server records are unavailable (e.g., plain vLLM without `moe_cap.systems.vllm`),
the profiler falls back to a `batch_size=1` approximation and prints:

```
WARNING: No server records available. Using fallback with batch_size=1 approximation.
This will NOT reflect actual continuous batching behavior!
```

For accurate continuous-batching metrics, use `moe_cap.systems.vllm` or `moe_cap.systems.sglang`
as the server even in profiling-only mode.
