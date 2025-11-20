# vLLM Expert Distribution Report - Quick Start Guide

This guide shows how to use the expert distribution recording functionality with vLLM.


### Manual API Control

Start the server normally:
```bash
python -m moe_cap.systems.vllm Qwen/Qwen1.5-MoE-A2.7B --tensor-parallel-size 4 --enforce-eager 
```

**Step 1: Configure recording mode**
```bash
curl -X POST "http://localhost:8000/configure_expert_distribution?mode=per_pass"
```

Available modes:
- `stat` - Aggregate statistics (default)
- `per_pass` - Per-forward-pass metrics
- `per_token` - Per-token expert selections

**Step 2: Start recording**
```bash
curl -X POST "http://localhost:8000/start_expert_distribution"
```

**Step 3: Make generation requests**
```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen1.5-MoE-A2.7B", "prompt": "write Hello World in python", "max_tokens": 100}'
```

**Step 4: View summary (clean output)**
```bash
curl -X POST "http://localhost:8000/dump_expert_distribution?summary_only=true" | python3 -m json.tool
```

**Step 5: Stop recording (optional)**
```bash
curl -X POST "http://localhost:8000/stop_expert_distribution"
```

---

## Example Output

### Summary Output (API)
```json
{
  "status": "success",
  "num_workers": 4,
  "summary": {
    "workers": [
      {
        "rank": 0,
        "recording_mode": "per_pass",
        "num_layers": 24,
        "num_experts": 60,
        "num_records": 7,
        "sample_record": {
          "forward_pass_id": 0,
          "rank": 0,
          "total_activated_experts": 359.0,
          "avg_activated_per_layer": 14.958,
          "expert_utilization": 0.2493
        }
      }
    ]
  }
}
```

### JSONL File (Auto-Recording)
```json
{"forward_pass_id": 1, "batch_size": 1, "latency": 0.123, "seq_lens_sum": 10, "forward_mode": "prefill", "expert_activation": 26.08}
{"forward_pass_id": 2, "batch_size": 1, "latency": 0.045, "seq_lens_sum": 11, "forward_mode": "decode", "expert_activation": 24.5}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/configure_expert_distribution?mode={mode}` | POST | Configure recording mode (`stat`, `per_pass`, `per_token`) |
| `/start_expert_distribution` | POST | Start recording |
| `/stop_expert_distribution` | POST | Stop recording |
| `/dump_expert_distribution?summary_only=true` | POST | Get summary statistics (default: clean output) |
| `/dump_expert_distribution?summary_only=false` | POST | Get full detailed data (for debugging) |
| `/expert_distribution_status` | GET | Check if expert distribution is available |

---

## Recording Modes

### `stat` Mode
- **Purpose**: Aggregate statistics across all forward passes
- **Output**: Cumulative expert counts
- **Use case**: Overall model behavior analysis

### `per_pass` Mode
- **Purpose**: Per-forward-pass expert activation metrics
- **Output**: One record per forward pass with activation statistics
- **Use case**: Analyzing individual forward passes

### `per_token` Mode
- **Purpose**: Detailed per-token expert selections
- **Output**: Expert selections for each token
- **Use case**: Fine-grained analysis (can be very large)

---

## Tips

1. **Use `summary_only=true`** for clean, readable output (default)
2. **Use `summary_only=false`** only for debugging (returns full arrays)
3. **Auto-recording** writes to JSONL file automatically

---
