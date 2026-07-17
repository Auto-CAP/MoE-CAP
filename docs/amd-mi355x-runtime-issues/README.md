# AMD MI355X Runtime Issue Reproduction Notes

This document tracks serving failures observed while running MoE-CAP benchmarks on AMD MI355X GPUs. Error excerpts are copied from saved logs. Artifact paths point to the benchmark host.

Each section separates:

- **Proven evidence:** directly observed in logs or artifacts.
- **Suspected cause:** a working hypothesis that still needs a minimal kernel-level reproduction.
- **Quick unblock:** a temporary workaround or comparison point, not necessarily a verified fix.

## Summary

| Model | Backend and configuration | Observed problem | Result |
|---|---|---|---|
| Qwen3-235B-A22B-Instruct-2507-FP8 | SGLang TP4, EP1, client batch size 1 | LongBench GPU memory fault; GSM8K and Arena-Hard silent output degradation | No clean `bs=1` baseline |
| DeepSeek-R1 | vLLM TP8, expert parallel, AITER | AITER FP8 GEMM compile failure or first-request sampling timeout | No valid exact-configuration result |
| Kimi-K2.5 | vLLM TP4/TP8, AITER | MLA constraint, 384-expert routing rejection, GPU fault, and corrupted 16K-token output | No clean vLLM baseline |

---

## 1. SGLang: Qwen3-235B-A22B-Instruct-2507-FP8

### 1.1 Observed setup

```text
Image: moecap-sglang-amd:0.5.14
SGLang: 0.5.14.dev20260708+g108a183f6b
Hardware: 4x AMD Instinct MI355X 288GB
Tensor parallelism: 4
Expert parallelism: 1
Client server_batch_size: 1
Observed average prefill batch size: 1.0
Observed average decode batch size: 1.0
```

Server command recorded by the runner:

```text
python -m moe_cap.systems.sglang_rocm \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --host 127.0.0.1 \
  --port 33000 \
  --trust-remote-code \
  --tp 4 \
  --ep-size 1 \
  --mem-fraction-static 0.68 \
  --expert-distribution-recorder-mode stat
```

Relevant runner environment:

```text
PYTHONPATH=/workspace/MoE-CAP
PYTHONDONTWRITEBYTECODE=1
HF_HOME=/root/.cache/huggingface
HF_HUB_CACHE=/root/.cache/huggingface/hub
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=<run-specific server_records path>
```

The runner did not set a Qwen-specific `SGLANG_USE_AITER` override. The server selected its ROCm/AITER paths internally.

The artifact metadata records `precision: bfloat16` even though the model identifier is explicitly FP8. This is a metadata-reporting bug. It does **not** prove that the server loaded BF16 weights, and it is not treated as a cause of the runtime failures below.

### 1.2 Proven: fresh LongBench runs crash with GPU memory-access faults

Two fresh LongBench v1 runs failed with the same class of server-side fault.

#### Run `20260715-200713`

```text
Rows: 256
Successful requests: 162
Failed requests: 94
First failed index: 162
First failed input length: 18,376 tokens
```

Artifact:

```text
/home/amd/moecap-results/moe/amd/sglang/qwen3-235b-a22b-instruct-2507-fp8/longbench_v1_256samples/mi355xx4/batch-size-1/20260715-200713/
```

#### Run `20260716-084010`

```text
Rows: 256
Successful requests: 25
Failed requests: 231
First failed index: 25
First failed input length: 16,735 tokens
```

Artifact:

```text
/home/amd/moecap-results/moe/amd/sglang/qwen3-235b-a22b-instruct-2507-fp8/longbench_v1_256samples/mi355xx4/batch-size-1/20260716-084010/
```

Representative server errors:

```text
Memory access fault by GPU node-5 ... Reason: Unknown.
Memory access fault by GPU node-3 ... Reason: Unknown.
Memory access fault by GPU node-2 ... Reason: Unknown.
Fatal Python error: Aborted
Subprocess scheduler_3 ... crashed with exit code -6.
```

The second run used a fresh server and failed much earlier, so this is not explained by a stale process. Neither run logged an out-of-memory error. After the scheduler died, recorder endpoints became unreachable; the resulting `TTFT=0` and downstream `ZeroDivisionError` were consequences of the crash, not its root cause.

### 1.3 Proven: GSM8K and Arena-Hard complete but output quality collapses

Run `20260716-161459` completed all 256 HTTP requests on both datasets. The GSM8K and Arena-Hard server logs contain no GPU memory fault, OOM, fatal abort, or scheduler crash.

#### GSM8K

```text
Rows/success/nonempty: 256/256/256
Fixed-extractor correct: 126/256
Accuracy: 49.22%
No extracted final answer: 61
Extracted but wrong answer: 69
```

The fixed-extractor audit is stored separately from the immutable generation artifact:

```text
/home/amd/moecap-results/derived-audits/qwen235_sglang_bs1_gsm8k_20260716-161459_fixed_extractor.json
```

Raw artifact:

```text
/home/amd/moecap-results/moe/amd/sglang/qwen3-235b-a22b-instruct-2507-fp8/gsm8k_256samples/mi355xx4/batch-size-1/20260716-161459/
```

#### Arena-Hard

```text
Rows/success: 256/256
Non-empty outputs: 211
Blank outputs: 45
Outputs at the 16,384-token cap: 40
Judge wins/losses/ties/errors: 29/226/1/0
Reported Arena-Hard win rate: 11.62%
Judge: gpt-4.1
Baseline: gpt-4-0613
```

Artifact:

```text
/home/amd/moecap-results/moe/amd/sglang/qwen3-235b-a22b-instruct-2507-fp8/arena-hard_256samples/mi355xx4/batch-size-1/20260716-161459/
```

### 1.4 Suspected cause, not proven

The combination of:

1. illegal GPU memory access on long single-request prefill shapes;
2. blank or cap-length generations without a server-side exception; and
3. a previously successful default-continuous-batching comparison

is consistent with a numerical or memory-correctness problem in the SGLang ROCm execution path. AITER GEMM, attention, MoE kernels, and the ROCm runtime are all still candidates. The saved fault lines do not name the failing kernel, so no one component should be presented as the proven root cause.

### 1.5 Quick unblock and reproduction request

For Qwen there is no verified fix to try: default continuous batching is a temporary known-good comparison point, not a fix for the `bs=1` path.

Meanwhile to reproduce your exact setup, please share:
• The full SGLang launch command with all arguments

• The complete env-var block as launched

• The container image + tag

---

## 2. vLLM: DeepSeek-R1

### 2.1 Requested configuration

```text
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MOE=1
SAFETENSORS_FAST_GPU=1
TP=8
Expert parallel enabled
reasoning_parser=deepseek_r1
```

### 2.2 Proven: AITER FP8 GEMM compilation failure

Artifact:

```text
/home/amd/moecap-results/smoke-moe/vllm/deepseek-r1/20260714-051900/server_failure.log
```

Verbatim traceback excerpt:

```text
File "/opt/python/lib/python3.13/site-packages/aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py", line 103, in gemm_a8w8_blockscale
    _gemm_a8w8_blockscale_kernel[grid](
File "/opt/python/lib/python3.13/site-packages/triton/backends/amd/compiler.py", line 264, in make_ttgir
    pm.run(mod, 'make_ttgir')
RuntimeError: PassManager::run failed
```

### 2.3 Proven: first-request sampling timeout

Artifact:

```text
/home/amd/moecap-results/smoke-moe/vllm/deepseek-r1/20260714-054954/warmup_failure.log
```

Verbatim error:

```text
TimeoutError: RPC call to sample_tokens timed out.
Parent process exited, terminating worker queues
AsyncLLM output_handler failed.
```

Both failures occurred before a valid functional smoke completed. They are not answer-extraction or evaluation failures.

### 2.4 Quick unblock and reproduction request

Requested quick-unblock hypothesis (not yet verified by these artifacts):

For DeepSeek quick unblock to try: adding `--enforce-eager` avoids the failing compile path while keeping AITER and MoE on.

The proven failure is still inside the AITER Triton FP8 GEMM compile/JIT path. Treat `--enforce-eager` as a diagnostic trial until an otherwise identical functional smoke confirms it.

Meanwhile to reproduce your exact setup, please share:
• The full `vllm serve` command with all arguments

• The complete env-var block as launched

• The container image + tag

---

## 3. vLLM: Kimi-K2.5

Kimi exhibited several independent failures; no single flag is known to address all of them.

### 3.1 Proven: TP8 AITER MLA head-divisibility assertion

Artifact:

```text
/home/amd/moecap-results/smoke-moe/vllm/kimi-k2.5/20260714-091101/server_failure.log
```

Verbatim error:

```text
File "/opt/python/lib/python3.13/site-packages/aiter/ops/attention.py", line 812, in get_mla_metadata_info_v1
    assert num_head_qo % 16 == 0
AssertionError
EngineCore failed to start.
```

This is a proven AITER MLA constraint after TP partitioning.

### 3.2 Proven: AITER grouped-topk rejects 384 experts

Artifact:

```text
/home/amd/moecap-results/smoke-moe/vllm/kimi-k2.5/20260714-131826/server_failure.log
```

Verbatim error:

```text
File "/opt/python/lib/python3.13/site-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py", line 153, in rocm_aiter_grouped_topk
RuntimeError: num_experts must be a power of 2, but got 384
```

### 3.3 Proven: TP4 LongBench GPU memory-access faults

First artifact:

```text
/home/amd/moecap-results/moe/amd/vllm/kimi-k2.5/longbench_v1_256samples/mi355xx4/batch-size-default/20260714-072333/server_failure.log
```

```text
Memory access fault by GPU node-2 ... Reason: Write access to a read-only page.
Worker proc VllmWorker-0 died unexpectedly, shutting down executor.
```

Second artifact:

```text
/home/amd/moecap-results/moe/amd/vllm/kimi-k2.5/longbench_v1_256samples/mi355xx4/batch-size-default/20260714-081214/server_failure.log
```

```text
Memory access fault by GPU node-2 ... Reason: Unknown.
```

The faults occurred at reported GPU KV-cache usage of 98.8% and 80.7%, respectively, so the evidence does not support reducing the problem to simple KV-cache exhaustion.

### 3.4 Proven: HTTP-successful but corrupted 16K-token outputs

Artifact:

```text
/home/amd/moecap-results/moe/amd/vllm/kimi-k2.5/arena-hard_256samples/mi355xx8/batch-size-default/20260714-141057/moonshotai/Kimi-K2.5/output_data_arena-hard_20260714_162532.jsonl
```

Affected responses begin coherently and then degrade into repeated punctuation until the 16,384-token cap:

```text
output_token_count=16384
...tail: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Run outcome:

```text
HTTP-successful outputs: 256/256
Trailing repeated-punctuation failures: 168/256
Valid outputs sent to judge: 88/256
```

The MLA assertion and 384-expert rejection are proven constraints from their tracebacks. A shared ROCm/AITER cause for the memory fault and corrupted generations remains suspected, not proven.

### 3.5 Quick unblock and reproduction request

For Kimi there is no single verified unblock to try: the MLA constraint, 384-expert rejection, memory-access faults, and corrupted long generations are independent observed failures.

Meanwhile to reproduce your exact setup, please share:
• The full `vllm serve` command with all arguments

• The complete env-var block as launched

• The container image + tag
