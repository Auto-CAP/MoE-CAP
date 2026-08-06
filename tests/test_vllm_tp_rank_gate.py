"""The recorder gate must key on the in-group TP rank, not the global rank.

vLLM's GroupCoordinator.rank is the GLOBAL rank: in any layout with more
than one group (DP or PP times TP) it is non-zero for every member of every
group but the first, so a gate comparing it to 0 silences whole groups'
recorders. The gate must use the rank within the group.
"""

import ast
import sys
import types
from functools import lru_cache
from pathlib import Path

SOURCE_PATH = Path(__file__).parents[1] / "moe_cap" / "systems" / "vllm.py"
WANTED = {"_safe_int", "_tp_rank_in_group", "_get_tp_rank"}


def _load_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in WANTED
    ]
    missing = WANTED - {node.name for node in nodes}
    assert not missing, f"vllm.py helpers not found: {sorted(missing)}"
    namespace = {"lru_cache": lru_cache}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace


class _FakeGroup:
    def __init__(self, rank, rank_in_group=None, ranks=None):
        self.rank = rank
        if rank_in_group is not None:
            self.rank_in_group = rank_in_group
        if ranks is not None:
            self.ranks = ranks


def _install_fake_parallel_state(tp_group):
    """Install a fake vllm.distributed.parallel_state exposing get_tp_group."""
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tp_group = lambda: tp_group
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = parallel_state
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.distributed = distributed
    saved = {
        name: sys.modules.get(name)
        for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state")
    }
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.distributed"] = distributed
    sys.modules["vllm.distributed.parallel_state"] = parallel_state
    return saved


def _restore_modules(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_second_group_leader_records():
    """A non-first TP group's leader has global rank != 0 but must record.

    Simulated layout: two TP groups of four (e.g. DP=2 x TP=4). The second
    group's leader is global rank 4, in-group rank 0. The unfixed gate
    returned the global rank, so this worker never recorded and the second
    replica's passes vanished from the trace.
    """
    helpers = _load_helpers()
    saved = _install_fake_parallel_state(
        _FakeGroup(rank=4, rank_in_group=0, ranks=[4, 5, 6, 7])
    )
    try:
        assert helpers["_get_tp_rank"]() == 0
    finally:
        _restore_modules(saved)


def test_non_leader_in_first_group_stays_gated():
    """Global rank 0's group peers must still not record (one writer per pass)."""
    helpers = _load_helpers()
    saved = _install_fake_parallel_state(
        _FakeGroup(rank=3, rank_in_group=3, ranks=[0, 1, 2, 3])
    )
    try:
        assert helpers["_get_tp_rank"]() == 3
    finally:
        _restore_modules(saved)


def test_single_group_unchanged():
    """Pure-TP single-group layouts behave exactly as before the fix."""
    helpers = _load_helpers()
    for global_rank in (0, 1, 7):
        saved = _install_fake_parallel_state(
            _FakeGroup(rank=global_rank, rank_in_group=global_rank, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        )
        try:
            assert helpers["_get_tp_rank"]() == global_rank
        finally:
            _restore_modules(saved)


def test_unavailable_parallel_state_defaults_to_zero():
    helpers = _load_helpers()

    def _raise():
        raise RuntimeError("parallel state not initialized")

    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tp_group = _raise
    saved = {
        name: sys.modules.get(name)
        for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state")
    }
    sys.modules["vllm.distributed.parallel_state"] = parallel_state
    try:
        assert helpers["_get_tp_rank"]() == 0
    finally:
        _restore_modules(saved)


def test_group_without_rank_in_group_falls_back_to_ranks_index():
    """Coordinators lacking rank_in_group derive the position from ranks."""
    helpers = _load_helpers()
    group = _FakeGroup(rank=5, ranks=[4, 5, 6, 7])
    assert helpers["_tp_rank_in_group"](group) == 1
    assert helpers["_tp_rank_in_group"](None) == 0


def test_no_global_rank_gate_left_in_source():
    """No recording gate may read the coordinator's global rank directly."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "tp_group.rank if" not in source
    assert source.count("_tp_rank_in_group(get_tp_group())") >= 2
