"""Regression tests for attempted-prefill and generated-decode accounting."""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple


SOURCE_PATH = Path(__file__).parents[1] / "moe_cap" / "runner" / "openai_api_profile.py"
HELPERS = {"_tokenized_input_length", "_request_token_counts"}


def _load_helpers():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    missing = HELPERS - {node.name for node in nodes}
    if missing:
        raise AssertionError(f"runner helpers not found: {sorted(missing)}")
    namespace = {
        "Any": Any,
        "Callable": Callable,
        "List": List,
        "Tuple": Tuple,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace


_NAMESPACE = _load_helpers()
tokenized_input_length = _NAMESPACE["_tokenized_input_length"]
request_token_counts = _NAMESPACE["_request_token_counts"]


@dataclass
class FakeResult:
    success: bool
    prompt_len: int = 0
    output_len: int = 0
    generated_text: str = ""


class TokenizedInputLengthTest(unittest.TestCase):
    def test_accepts_plain_ids_and_tokenizer_mapping(self):
        self.assertEqual(tokenized_input_length([1, 2, 3]), 3)
        self.assertEqual(
            tokenized_input_length({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}),
            4,
        )

    def test_rejects_batched_chat_prompts(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            tokenized_input_length({"input_ids": [[1, 2], [3, 4]]})


class RequestTokenCountsTest(unittest.TestCase):
    def test_failed_request_keeps_full_prefill_and_zero_decode(self):
        result = FakeResult(
            success=False,
            prompt_len=0,
            output_len=999,
            generated_text="must not count",
        )

        self.assertEqual(
            request_token_counts(result, 46_752, lambda text: [1] * len(text)),
            (46_752, 0),
        )

    def test_success_prefers_server_usage_counts(self):
        result = FakeResult(success=True, prompt_len=123, output_len=45)
        self.assertEqual(
            request_token_counts(result, 999, lambda text: [1]),
            (123, 45),
        )

    def test_success_without_usage_tokenizes_generated_text(self):
        result = FakeResult(
            success=True,
            prompt_len=123,
            output_len=0,
            generated_text="abc",
        )
        self.assertEqual(
            request_token_counts(result, 999, lambda text: list(text)),
            (123, 3),
        )


if __name__ == "__main__":
    unittest.main()
