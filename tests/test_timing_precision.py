"""The profiler must emit timing at full precision.

`e2e_s` was written through ``round(..., 2)``, which on sub-second makespans
is up to ±12.5% quantization — and every downstream quantity priced on wall
clock ($/request, J/request, unnormalized_e2e, request/s) inherits the error.
The wall clock (``total_time``) comes from ``time.perf_counter`` and is exact.

This test pins that specific regression: it walks the profiler's AST and rejects
any ``round()`` call whose argument expression involves ``total_time``. It does
not police other timing fields or indirect rounding through intermediates.
"""

import ast
import unittest
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).parents[1] / "moe_cap" / "runner" / "openai_api_profile.py"
)


class TimingPrecisionTest(unittest.TestCase):
    def test_no_rounding_of_wall_clock_quantities(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "round"
                and node.args
                and any(
                    isinstance(sub, ast.Name) and sub.id == "total_time"
                    for sub in ast.walk(node.args[0])
                )
            ):
                offenders.append(
                    f"line {node.lineno}: {ast.get_source_segment(source, node)}"
                )
        self.assertEqual(
            offenders,
            [],
            "wall-clock quantities must be written at full precision:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
