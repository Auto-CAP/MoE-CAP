"""Unit tests for the FastAPI >= 0.137 prometheus route-naming compatibility shim.

These tests exercise ``_install_prometheus_routing_compat`` from
``moe_cap/systems/vllm.py`` without importing the real vLLM stack. The function
source is extracted via ``ast`` and executed in an isolated namespace against a
fake ``prometheus_fastapi_instrumentator`` package installed into ``sys.modules``.
"""

import ast
import sys
import types
import unittest
from pathlib import Path

SOURCE_PATH = Path(__file__).parents[1] / "moe_cap" / "systems" / "vllm.py"


def _load_shim():
    """Extract and compile ``_install_prometheus_routing_compat`` in isolation."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    block = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_install_prometheus_routing_compat"
        ):
            block = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            break
    assert block is not None, "shim function not found in vllm.py"
    namespace = {}
    exec(block, namespace)
    return namespace["_install_prometheus_routing_compat"]


class _Request:
    def __init__(self, scope):
        self.scope = scope


class _PromModules:
    """Install a fake prometheus_fastapi_instrumentator package into sys.modules."""

    def __init__(self, get_route_name, with_middleware=True):
        self.original = get_route_name
        self.names = []

        pkg = types.ModuleType("prometheus_fastapi_instrumentator")
        routing = types.ModuleType("prometheus_fastapi_instrumentator.routing")
        routing.get_route_name = get_route_name
        pkg.routing = routing
        self.pkg = pkg
        self.routing = routing
        modules = {
            "prometheus_fastapi_instrumentator": pkg,
            "prometheus_fastapi_instrumentator.routing": routing,
        }
        if with_middleware:
            middleware = types.ModuleType(
                "prometheus_fastapi_instrumentator.middleware"
            )
            # Mimic ``from .routing import get_route_name`` (bound by value).
            middleware.get_route_name = get_route_name
            pkg.middleware = middleware
            self.middleware = middleware
            modules["prometheus_fastapi_instrumentator.middleware"] = middleware
        else:
            self.middleware = None
        self.names = list(modules)
        self._modules = modules

    def __enter__(self):
        self._saved = {n: sys.modules.get(n) for n in self.names}
        sys.modules.update(self._modules)
        return self

    def __exit__(self, *exc):
        for name, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        return False


class PrometheusRoutingCompatTest(unittest.TestCase):
    def setUp(self):
        self.install = _load_shim()

    def test_no_instrumentator_is_noop(self):
        # With no prometheus package installed, the shim must not raise.
        saved = {
            n: sys.modules.pop(n, None)
            for n in list(sys.modules)
            if n.startswith("prometheus_fastapi_instrumentator")
        }
        try:
            self.install()  # should be a silent no-op
        finally:
            for n, m in saved.items():
                if m is not None:
                    sys.modules[n] = m

    def test_non_import_error_on_import_propagates(self):
        # Only ImportError (dependency absent) is tolerated; any other failure
        # while importing the instrumentator must surface, not be swallowed.
        class _Boom(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("broken instrumentator: " + name)

        name = "prometheus_fastapi_instrumentator"
        saved = sys.modules.get(name)
        sys.modules[name] = _Boom(name)
        try:
            with self.assertRaises(RuntimeError):
                self.install()
        finally:
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def test_normal_route_naming_unchanged(self):
        calls = []

        def get_route_name(request, *args, **kwargs):
            calls.append(request)
            return "/v1/models"

        with _PromModules(get_route_name) as env:
            self.install()
            shim = env.routing.get_route_name
            self.assertIsNot(shim, get_route_name)
            result = shim(_Request({"path": "/ignored"}))
            self.assertEqual(result, "/v1/models")
            self.assertEqual(len(calls), 1)

    def test_path_attribute_error_falls_back_to_scope_path(self):
        def get_route_name(request):
            raise AttributeError(
                "'_IncludedRouter' object has no attribute 'path'"
            )

        with _PromModules(get_route_name) as env:
            self.install()
            shim = env.routing.get_route_name
            result = shim(_Request({"path": "/health"}))
            self.assertEqual(result, "/health")

    def test_missing_scope_path_returns_none(self):
        def get_route_name(request):
            raise AttributeError("object has no attribute 'path'")

        with _PromModules(get_route_name) as env:
            self.install()
            shim = env.routing.get_route_name
            self.assertIsNone(shim(_Request({})))

    def test_includedrouter_message_falls_back(self):
        # The exact observed failure text must be recognised as the known case.
        def get_route_name(request):
            raise AttributeError(
                "'_IncludedRouter' object has no attribute 'path'"
            )

        with _PromModules(get_route_name) as env:
            self.install()
            shim = env.routing.get_route_name
            self.assertEqual(shim(_Request({"path": "/v1/models"})), "/v1/models")

    def test_unrelated_attribute_error_propagates(self):
        def get_route_name(request):
            raise AttributeError("'Foo' object has no attribute 'scope'")

        with _PromModules(get_route_name) as env:
            self.install()
            shim = env.routing.get_route_name
            with self.assertRaises(AttributeError):
                shim(_Request({"path": "/health"}))

    def test_path_mentioning_attribute_error_propagates(self):
        # Mentions "path" but is NOT a missing ``.path`` attribute — must not be
        # swallowed by the narrowed match.
        for message in (
            "'Route' object has no attribute 'path_regex'",
            "'Request' object has no attribute 'path_params'",
            "bad path value in route",
        ):
            def get_route_name(request, _m=message):
                raise AttributeError(_m)

            with _PromModules(get_route_name) as env:
                self.install()
                shim = env.routing.get_route_name
                with self.assertRaises(AttributeError):
                    shim(_Request({"path": "/health"}))

    def test_non_attribute_error_propagates(self):
        def get_route_name(request):
            raise ValueError("boom")

        with _PromModules(get_route_name) as env:
            self.install()
            shim = env.routing.get_route_name
            with self.assertRaises(ValueError):
                shim(_Request({"path": "/health"}))

    def test_middleware_reference_is_synced(self):
        def get_route_name(request):
            raise AttributeError(
                "'_IncludedRouter' object has no attribute 'path'"
            )

        with _PromModules(get_route_name) as env:
            self.install()
            # The by-value binding in middleware must now point at the shim.
            self.assertIs(env.middleware.get_route_name, env.routing.get_route_name)
            self.assertTrue(
                getattr(env.middleware.get_route_name, "_moe_cap_route_name_shim", False)
            )
            self.assertEqual(
                env.middleware.get_route_name(_Request({"path": "/metrics"})),
                "/metrics",
            )

    def test_idempotent_does_not_double_wrap(self):
        def get_route_name(request):
            return "/health"

        with _PromModules(get_route_name) as env:
            self.install()
            first = env.routing.get_route_name
            self.install()
            second = env.routing.get_route_name
            self.assertIs(first, second)
            # Original is preserved for unwrapping / introspection.
            self.assertIs(first._moe_cap_wrapped, get_route_name)

    def test_late_middleware_import_is_synced_on_resync(self):
        # Simulates the middleware being imported *after* the first install
        # (the reason patched_build_app re-runs the shim).
        def get_route_name(request):
            raise AttributeError(
                "'_IncludedRouter' object has no attribute 'path'"
            )

        with _PromModules(get_route_name, with_middleware=False) as env:
            self.install()
            shim = env.routing.get_route_name
            # Now a middleware appears binding the ORIGINAL by value.
            middleware = types.ModuleType(
                "prometheus_fastapi_instrumentator.middleware"
            )
            middleware.get_route_name = get_route_name
            sys.modules["prometheus_fastapi_instrumentator.middleware"] = middleware
            try:
                self.install()  # re-sync
                self.assertIs(middleware.get_route_name, shim)
            finally:
                sys.modules.pop(
                    "prometheus_fastapi_instrumentator.middleware", None
                )


class SourceIntegrationTest(unittest.TestCase):
    def test_shim_installed_and_resynced_in_vllm_module(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        # Installed at import time...
        self.assertIn(
            "\n_install_prometheus_routing_compat()", source
        )
        # ...and re-synced when the app (and its middleware) is built.
        self.assertIn(
            "app = original_build_app(args, *extra_args, **extra_kwargs)\n\n"
            "        # Re-sync the prometheus route-naming shim",
            source,
        )
        self.assertIn(
            "_install_prometheus_routing_compat()",
            source.split("def patched_build_app")[1],
        )


if __name__ == "__main__":
    unittest.main()
