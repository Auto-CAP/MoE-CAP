import ast
import os
from pathlib import Path


def _source():
    return (Path(__file__).parents[1] / "moe_cap/systems/vllm_rocm.py").read_text()


def test_spawn_is_configured_before_hardware_import():
    source = _source()
    assert source.index('os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")') < source.index(
        "import moe_cap.utils.hardware_utils"
    )


def test_spawn_default_does_not_override_operator_choice(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "forkserver"


def test_spawn_default_is_used_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    tree = ast.parse(_source())
    expr = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "setdefault"
    )
    exec(compile(ast.Module([expr], type_ignores=[]), "vllm_rocm.py", "exec"), {"os": os})
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
