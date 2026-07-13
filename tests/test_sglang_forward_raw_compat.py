import ast
import inspect
from functools import lru_cache
from pathlib import Path


def _load_helpers():
    path = Path(__file__).parents[1] / "moe_cap/systems/sglang.py"
    tree = ast.parse(path.read_text())
    wanted = {
        "_forward_raw_accepts_skip_attn",
        "_call_forward_raw",
    }
    nodes: list[ast.stmt] = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            nodes.append(node)
            found.add(node.name)
    assert found == wanted
    namespace = {"inspect": inspect, "lru_cache": lru_cache}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_legacy_forward_raw_receives_skip_argument():
    helpers = _load_helpers()

    class Runner:
        def _forward_raw(
            self,
            forward_batch,
            skip_attn_backend_init=False,
            pp_proxy_tensors=None,
            reinit_attn_backend=False,
            split_forward_count=1,
        ):
            self.args = (
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            return "legacy"

    runner = Runner()
    result = helpers["_call_forward_raw"](runner, "batch", True, "proxy", True, 3)
    assert result == "legacy"
    assert runner.args == ("batch", True, "proxy", True, 3)


def test_new_forward_raw_applies_deprecated_flag_to_batch():
    helpers = _load_helpers()

    class Batch:
        def apply_deprecated_skip_attn_backend_init(self, value):
            self.skip_value = value

    class Runner:
        def _forward_raw(
            self,
            forward_batch,
            pp_proxy_tensors=None,
            reinit_attn_backend=False,
            split_forward_count=1,
        ):
            self.args = (
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            return "new"

    batch = Batch()
    runner = Runner()
    result = helpers["_call_forward_raw"](runner, batch, False, "proxy", True, 4)
    assert result == "new"
    assert batch.skip_value is False
    assert runner.args == (batch, "proxy", True, 4)


def test_new_forward_raw_allows_batch_without_compat_method():
    helpers = _load_helpers()

    class Runner:
        def _forward_raw(self, forward_batch, pp_proxy_tensors=None, reinit_attn_backend=False, split_forward_count=1):
            return forward_batch, pp_proxy_tensors, reinit_attn_backend, split_forward_count

    result = helpers["_call_forward_raw"](Runner(), object(), None, "proxy", False, 2)
    assert result[1:] == ("proxy", False, 2)
