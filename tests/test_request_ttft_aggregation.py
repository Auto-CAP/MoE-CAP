"""Tests for server-side per-request TTFT aggregation."""

import ast
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SOURCE_PATH = (
    Path(__file__).parents[1] / "moe_cap" / "runner" / "openai_api_profile.py"
)


def _load_helper():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mean_server_request_ttft"
    )
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec("\n".join(lines[node.lineno - 1 : node.end_lineno]), namespace)
    return namespace["_mean_server_request_ttft"]


class RequestTTFTAggregationTest(unittest.TestCase):
    def test_sums_chunked_prefill_and_averages_requests(self):
        mean_ttft = _load_helper()
        records = [
            {
                "forward_mode": "prefill",
                "latency": 0.10,
                "per_req_info": [
                    {"req_pool_idx": 0, "is_last_chunk": False},
                    {"req_pool_idx": 1, "is_last_chunk": True},
                ],
            },
            {
                "forward_mode": "decode",
                "latency": 0.01,
                "req_ids": [1],
            },
            {
                "forward_mode": "prefill",
                "latency": 0.04,
                "per_req_info": [
                    {"req_pool_idx": 0, "is_last_chunk": True},
                ],
            },
        ]
        self.assertAlmostEqual(mean_ttft(records), (0.10 + 0.14) / 2)

    def test_pool_slot_reuse_starts_a_new_request(self):
        mean_ttft = _load_helper()
        records = [
            {
                "forward_mode": "prefill",
                "latency": 0.05,
                "per_req_info": [
                    {"req_pool_idx": 3, "is_last_chunk": True},
                ],
            },
            {
                "forward_mode": "prefill",
                "latency": 0.07,
                "per_req_info": [
                    {"req_pool_idx": 3, "is_last_chunk": True},
                ],
            },
        ]
        self.assertAlmostEqual(mean_ttft(records), 0.06)

    def test_ignores_incomplete_requests_and_missing_annotations(self):
        mean_ttft = _load_helper()
        records = [
            {"forward_mode": "prefill", "latency": 9.0},
            {
                "forward_mode": "prefill",
                "latency": 0.08,
                "per_req_info": [
                    {"req_id": "unfinished", "is_last_chunk": False},
                    {"req_id": "done", "is_last_chunk": True},
                ],
            },
        ]
        self.assertAlmostEqual(mean_ttft(records), 0.08)
        self.assertIsNone(mean_ttft([]))


if __name__ == "__main__":
    unittest.main()
