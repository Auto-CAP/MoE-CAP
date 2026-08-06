"""Engine-uniform scheduled_tokens on per-pass records.

SGLang's per-pass seq_lens_sum is the RUNNING CONTEXT sum: under chunked
prefill every chunk re-counts all previously computed context, so summing it
over a request's chunks exceeds the prompt length. vLLM's seq_lens_sum is
the tokens scheduled in the pass. scheduled_tokens carries the vLLM meaning
on both engines; seq_lens_sum stays byte-identical to its historical
per-engine meaning.
"""

import ast
import types
from pathlib import Path
from typing import Dict

SGLANG_PATH = Path(__file__).parents[1] / "moe_cap" / "systems" / "sglang.py"
VLLM_PATH = Path(__file__).parents[1] / "moe_cap" / "systems" / "vllm.py"


def _load_sglang_helpers():
    wanted = {
        "_to_cpu_list",
        "_build_prefill_per_req_info",
        "_scheduled_tokens_for_record",
    }
    tree = ast.parse(SGLANG_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    missing = wanted - {node.name for node in nodes}
    assert not missing, f"sglang.py helpers not found: {sorted(missing)}"
    fake_torch = types.SimpleNamespace(Tensor=type("Tensor", (), {}))
    fake_logger = types.SimpleNamespace(
        warning=lambda *a, **k: None, debug=lambda *a, **k: None
    )
    namespace = {
        "torch": fake_torch,
        "logger": fake_logger,
        "Dict": Dict,
        "ForwardBatch": object,  # annotation stand-ins
        "ServerArgs": object,
        "_prefill_pool_counters": {},
        "_prefill_pool_last_seq": {},
        "_prefill_empty_reads_warned": False,
        "_prefill_no_orig_lens_warned": False,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SGLANG_PATH), "exec"),
        namespace,
    )
    return namespace


class _Batch:
    """Minimal ForwardBatch stand-in for one prefill pass."""

    def __init__(self, req_indices, extend_lens, seq_lens, orig_lens=None):
        self.req_pool_indices = req_indices
        self.extend_seq_lens_cpu = extend_lens
        self.seq_lens_cpu = seq_lens
        self.seq_lens = seq_lens
        self.orig_seq_lens = orig_lens


class _ServerArgs:
    chunked_prefill_size = 2048


def test_sglang_chunked_prefill_seq_lens_sum_recounts_context():
    """The defect and its fix, on one synthetic chunked prefill.

    One request, 3000-token prompt, chunk budget 2048: chunk 1 runs 2048
    tokens (running context 2048), chunk 2 runs 952 (running context 3000).
    seq_lens_sum over the two passes reads 5048 -- 2048 tokens re-counted --
    while the new field sums the actual per-pass work to exactly the prompt.
    """
    helpers = _load_sglang_helpers()
    build = helpers["_build_prefill_per_req_info"]
    scheduled = helpers["_scheduled_tokens_for_record"]

    chunk1 = _Batch([7], [2048], [2048], orig_lens=[3000])
    chunk2 = _Batch([7], [952], [3000], orig_lens=[3000])

    seq_lens_sums = []
    scheduled_sums = []
    for batch in (chunk1, chunk2):
        per_req_info = build(batch, _ServerArgs())
        # What the historical record carries as seq_lens_sum (running context).
        seq_lens_sums.append(int(batch.seq_lens_cpu[0]))
        scheduled_sums.append(scheduled("prefill", 1, per_req_info))

    # The historical field re-counts running context across chunks...
    assert sum(seq_lens_sums) == 5048
    # ...the engine-uniform field sums the actual per-pass tokens.
    assert scheduled_sums == [2048, 952]
    assert sum(scheduled_sums) == 3000


def test_sglang_batched_prefill_sums_extend_over_batch():
    helpers = _load_sglang_helpers()
    build = helpers["_build_prefill_per_req_info"]
    scheduled = helpers["_scheduled_tokens_for_record"]
    batch = _Batch([1, 2, 3], [100, 250, 75], [100, 250, 75], orig_lens=[100, 250, 75])
    per_req_info = build(batch, _ServerArgs())
    assert scheduled("prefill", 3, per_req_info) == 425


def test_sglang_decode_pass_is_one_token_per_request():
    helpers = _load_sglang_helpers()
    scheduled = helpers["_scheduled_tokens_for_record"]
    assert scheduled("decode", 17, None) == 17
    assert scheduled("decode", None, None) is None


def test_sglang_empty_reads_omit_rather_than_guess():
    helpers = _load_sglang_helpers()
    scheduled = helpers["_scheduled_tokens_for_record"]
    assert scheduled("prefill", 4, []) is None
    assert scheduled("prefill", 4, None) is None


def test_sglang_prompt_len_mirrors_vllm_semantics():
    """per_req_info gains prompt_len = admission prompt length (orig_len).

    total_len keeps its historical running-total meaning untouched: on the
    second chunk it reads 3000 because 3000 tokens are computed so far, and
    prompt_len reads 3000 because that is the admission prompt length --
    equal here, but different quantities (a cached prefix would split them).
    """
    helpers = _load_sglang_helpers()
    build = helpers["_build_prefill_per_req_info"]

    chunk1 = _Batch([7], [2048], [2048], orig_lens=[3000])
    info1 = build(chunk1, _ServerArgs())[0]
    assert info1["total_len"] == 2048  # running total, unchanged semantics
    assert info1["prompt_len"] == 3000  # admission prompt length
    assert info1["orig_len"] == 3000  # historical field untouched

    # Builds without orig_seq_lens cannot witness the prompt length: the
    # field is omitted, never guessed.
    bare = _Batch([9], [500], [500], orig_lens=None)
    info_bare = build(bare, _ServerArgs())[0]
    assert "prompt_len" not in info_bare


def _load_vllm_finalize():
    wanted = {"_finalize_pass_records"}
    tree = ast.parse(VLLM_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert nodes, "_finalize_pass_records not found in vllm.py"
    namespace = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(VLLM_PATH), "exec"),
        namespace,
    )
    return namespace


def test_vllm_records_carry_scheduled_tokens():
    """On vLLM seq_lens_sum already is the scheduled sum; the uniform field
    republishes it on every per-pass record."""
    helpers = _load_vllm_finalize()
    records = [
        {"batch_size": 3, "seq_lens_sum": 4096, "forward_mode": "prefill"},
        {"batch_size": 16, "seq_lens_sum": 16, "forward_mode": "decode"},
    ]
    out = helpers["_finalize_pass_records"](records)
    assert out is records
    for rec in out:
        assert rec["scheduled_tokens"] == rec["seq_lens_sum"]


def test_both_vllm_recording_branches_route_through_finalize():
    source = VLLM_PATH.read_text(encoding="utf-8")
    assert source.count("_finalize_pass_records(records_to_add)") == 2


def test_both_sglang_record_builders_attach_scheduled_tokens():
    tree = ast.parse(SGLANG_PATH.read_text(encoding="utf-8"))
    for name in ("forward_expert_record", "forward_profiling_only"):
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        calls = {
            n.func.id
            for n in ast.walk(func)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_scheduled_tokens_for_record" in calls, name
