"""The SGLang server and client rendezvous on one forward-record file.

The server writes per-forward-pass records there; the client reads them back for
TTFT, per-pass prefill latency, prefill/decode batch size and expert activation.
When the two sides resolved that path differently the client read nothing and the
metrics writer published 0 batch sizes, 0 expert activation and null timings that
were indistinguishable from measurements, with a zero exit code.

These tests guard the properties that can still regress, not the fixed bug:
one shared resolver, the launcher's configured path surviving untouched, and the
refusal to publish when no records came back -- including that the refusal is
narrow enough to leave recoverable runs alone. The method body is executed as it
ships, extracted the way the other client tests do (the module imports torch).
"""

import ast
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

from moe_cap.utils.recorder_paths import RECORDER_DIR_ENV, get_sglang_record_path

REPO_ROOT = Path(__file__).parents[1]
SERVER_SOURCE = REPO_ROOT / "moe_cap" / "systems" / "sglang.py"
CLIENT_SOURCE = REPO_ROOT / "moe_cap" / "runner" / "openai_api_profile.py"

MODEL_PATH = "Qwen/Qwen3-4B"


class ServerRecordsUnavailableError(RuntimeError):
    """Mirror of the client's exception, injected into the extracted namespace."""


class _BackendType:
    SGLANG = "sglang"
    VLLM = "vllm"


def _extract(name):
    """Return the real method, compiled from the source that ships."""

    source = CLIENT_SOURCE.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    block = textwrap.dedent(
        "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    )
    namespace = {
        "os": os,
        "json": __import__("json"),
        "get_sglang_record_path": get_sglang_record_path,
        "RECORDER_DIR_ENV": RECORDER_DIR_ENV,
        "ServerRecordsUnavailableError": ServerRecordsUnavailableError,
        "BackendType": _BackendType,
    }
    exec(compile(ast.parse(block), str(CLIENT_SOURCE), "exec"), namespace)
    return namespace[name]


class _Profiler:
    backend_type = _BackendType.SGLANG
    base_url = "http://127.0.0.1:30000"
    hf_model_name = MODEL_PATH


class _FakeRequests(types.ModuleType):
    """Stands in for ``requests``, which the method imports locally."""

    def __init__(self, error=None):
        super().__init__("requests")
        self.error = error

    def post(self, url, timeout=None):
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(raise_for_status=lambda: None)


class SglangRecordRendezvousTest(unittest.TestCase):
    def setUp(self):
        self.dump = _extract("_dump_batch_recording")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        previous = os.environ.get(RECORDER_DIR_ENV)
        os.environ[RECORDER_DIR_ENV] = self.tmp.name
        if previous is None:
            self.addCleanup(os.environ.pop, RECORDER_DIR_ENV, None)
        else:
            self.addCleanup(os.environ.__setitem__, RECORDER_DIR_ENV, previous)

        self.record_file = get_sglang_record_path(MODEL_PATH)
        self.addCleanup(sys.modules.pop, "requests", None)
        sys.modules["requests"] = _FakeRequests()

    def _write(self, text):
        os.makedirs(os.path.dirname(self.record_file), exist_ok=True)
        with open(self.record_file, "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_no_records_refuses_rather_than_publishing_zeros(self):
        # Absent file = server and client disagreed on the directory.
        # Present but empty = recording never started; both dump paths truncate
        # on open. Downstream the two are the same event: zeroed batch sizes and
        # null timings published as measurements under a zero exit code.
        for case, content in (("absent", None), ("empty", ""), ("blank", "\n  \n")):
            with self.subTest(case):
                if content is None:
                    self.assertFalse(os.path.exists(self.record_file))
                else:
                    self._write(content)
                with self.assertRaises(ServerRecordsUnavailableError):
                    self.dump(_Profiler())

    def test_records_present_are_returned(self):
        # The refusal must not fire on a healthy run.
        self._write('{"forward_mode": "prefill"}\n{"forward_mode": "decode"}\n')
        self.assertEqual(
            self.dump(_Profiler()),
            [{"forward_mode": "prefill"}, {"forward_mode": "decode"}],
        )

    def test_transport_failure_still_degrades_to_empty(self):
        # The refusal must stay narrow: a run that can still fall back is not
        # aborted, so this path keeps its warn-and-continue behaviour.
        sys.modules["requests"] = _FakeRequests(OSError("connection refused"))
        self.assertEqual(self.dump(_Profiler()), [])

    def test_configured_directory_is_honoured_verbatim(self):
        # Every healthy run works because the launcher sets this; the absolute
        # path it sets must survive resolution unchanged.
        configured = "/dev/shm/sglang_expert_distribution_recorder"
        self.assertEqual(
            get_sglang_record_path(MODEL_PATH, {RECORDER_DIR_ENV: configured}),
            os.path.join(configured, MODEL_PATH, "expert_distribution_record.jsonl"),
        )

    def test_both_dump_sites_resolve_through_the_shared_helper(self):
        # The original defect was two private defaults drifting apart. Counting
        # call sites is the only guard available against one side reintroducing
        # its own: the server module needs sglang installed to import.
        def helper_calls(path):
            return [
                n
                for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "get_sglang_record_path"
            ]

        self.assertGreaterEqual(len(helper_calls(SERVER_SOURCE)), 2)
        self.assertGreaterEqual(len(helper_calls(CLIENT_SOURCE)), 1)


if __name__ == "__main__":
    unittest.main()
