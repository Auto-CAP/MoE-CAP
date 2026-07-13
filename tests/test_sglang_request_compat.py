"""Compatibility tests for building `ExpertDistributionReq2` in sglang.py.

Across SGLang releases the base class `BaseReq` changed shape:

* Newer builds (e.g. 0.5.14, image ``rocm/sgl-dev:v0.5.14-rocm720-mi35x-*``)
  define it as ``class BaseReq(msgspec.Struct, tag=True, kw_only=True,
  array_like=True)``. Re-decorating such a class with
  ``@dataclasses.dataclass`` raises ``AttributeError: readonly attribute``.
* Older builds define ``BaseReq`` as a plain dataclass.

These tests exercise the small block in ``moe_cap/systems/sglang.py`` that picks
the right construction mechanism, against both flavours of ``BaseReq``. They use
a fake ``io_struct`` module and exec only the relevant source snippet, so they
never import SGLang.
"""

import dataclasses
import unittest
from pathlib import Path

try:
    import msgspec
except ImportError:  # pragma: no cover - exercised only where msgspec is absent
    msgspec = None

SOURCE_PATH = Path(__file__).parents[1] / "moe_cap" / "systems" / "sglang.py"

_SNIPPET_START = "def _base_req_is_msgspec_struct(base) -> bool:"
_SNIPPET_END = "iostruct.ExpertDistributionReq = ExpertDistributionReq2"


def _compat_snippet():
    """Extract the ExpertDistributionReq2 construction block from sglang.py."""
    lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == _SNIPPET_START)
    end = next(i for i, line in enumerate(lines) if line == _SNIPPET_END)
    assert start < end, (start, end)
    return "\n".join(lines[start : end + 1])


class _FakeIoStruct:
    """Stand-in for ``sglang.srt.managers.io_struct``."""

    class ExpertDistributionReq:  # upstream class we masquerade as
        pass

    # Mimic the real upstream identity so the dunder replacement is observable.
    ExpertDistributionReq.__module__ = "sglang.srt.managers.io_struct"
    ExpertDistributionReq.__qualname__ = "ExpertDistributionReq"
    ExpertDistributionReq.__name__ = "ExpertDistributionReq"


class _ExpertDistributionReqType:
    """Placeholder annotation type; value is irrelevant to construction."""


def _run_snippet(base_req, req_type=_ExpertDistributionReqType):
    iostruct = _FakeIoStruct()
    namespace = {
        "BaseReq": base_req,
        "ExpertDistributionReqType": req_type,
        "iostruct": iostruct,
        "dataclasses": dataclasses,
    }
    exec(_compat_snippet(), namespace)
    return namespace, iostruct


class ExpertDistributionReqCompatTest(unittest.TestCase):
    def _assert_common(self, namespace, iostruct, base_req):
        req2 = namespace["ExpertDistributionReq2"]
        # Subclass of the provided BaseReq and carries the `action` field.
        self.assertTrue(issubclass(req2, base_req))
        self.assertIn("action", getattr(req2, "__annotations__", {}))
        # Dunders are rewritten to impersonate upstream ExpertDistributionReq.
        self.assertEqual(req2.__module__, "sglang.srt.managers.io_struct")
        self.assertEqual(req2.__qualname__, "ExpertDistributionReq")
        self.assertEqual(req2.__name__, "ExpertDistributionReq")
        # The upstream symbol is replaced with our recorder-aware subclass.
        self.assertIs(iostruct.ExpertDistributionReq, req2)

    @unittest.skipIf(msgspec is None, "msgspec not installed")
    def test_msgspec_struct_base(self):
        class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
            pass

        # Use a str field type so the msgpack roundtrip below validates cleanly.
        namespace, iostruct = _run_snippet(BaseReq, req_type=str)
        req2 = namespace["ExpertDistributionReq2"]
        self._assert_common(namespace, iostruct, BaseReq)
        # It is a real msgspec.Struct (not a dataclass) and stays encodable.
        self.assertTrue(issubclass(req2, msgspec.Struct))
        self.assertFalse(dataclasses.is_dataclass(req2))
        # `action` is keyword-only (mirrors upstream `kw_only=True`).
        with self.assertRaises(TypeError):
            req2("start")
        instance = req2(action="start")
        self.assertEqual(instance.action, "start")
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(instance), type=req2
        )
        self.assertEqual(decoded.action, "start")

    @unittest.skipIf(msgspec is None, "msgspec not installed")
    def test_msgspec_struct_base_does_not_raise_readonly(self):
        # Guards against the original bug: @dataclasses.dataclass over a
        # msgspec.Struct base raised "AttributeError: readonly attribute".
        class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
            pass

        with self.assertRaises(AttributeError):

            @dataclasses.dataclass
            class _Bad(BaseReq):
                action: _ExpertDistributionReqType

        # The production snippet, by contrast, builds cleanly.
        _run_snippet(BaseReq)

    def test_legacy_dataclass_base(self):
        @dataclasses.dataclass
        class BaseReq:
            pass

        namespace, iostruct = _run_snippet(BaseReq)
        req2 = namespace["ExpertDistributionReq2"]
        self._assert_common(namespace, iostruct, BaseReq)
        # Legacy path yields a dataclass with a usable `action` field.
        self.assertTrue(dataclasses.is_dataclass(req2))
        instance = req2(action="start")
        self.assertEqual(instance.action, "start")

    def test_detector_ignores_non_struct(self):
        namespace, _ = _run_snippet(dataclasses.make_dataclass("BaseReq", []))
        self.assertFalse(namespace["_base_req_is_msgspec_struct"](object))


if __name__ == "__main__":
    unittest.main()
