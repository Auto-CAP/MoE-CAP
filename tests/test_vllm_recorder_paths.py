"""Tests for per-run vLLM recorder filesystem isolation."""

import os
import tempfile
import unittest
from pathlib import Path

from moe_cap.utils.recorder_paths import (
    RECORDER_DIR_ENV,
    get_vllm_recording_dir,
    get_vllm_recording_path,
)


class VllmRecorderPathsTest(unittest.TestCase):
    def test_distinct_run_directories_do_not_share_any_control_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_a = os.path.join(tmp, "run-a")
            run_b = os.path.join(tmp, "run-b")
            filenames = (
                "vllm_batch_recording.flag",
                "vllm_batch_records.jsonl",
                "vllm_expert_distribution_recording.flag",
                "vllm_expert_distribution_auto_start.flag",
            )

            paths_a = {
                get_vllm_recording_path(name, {RECORDER_DIR_ENV: run_a})
                for name in filenames
            }
            paths_b = {
                get_vllm_recording_path(name, {RECORDER_DIR_ENV: run_b})
                for name in filenames
            }

            self.assertTrue(paths_a.isdisjoint(paths_b))
            self.assertTrue(all(path.startswith(run_a + os.sep) for path in paths_a))
            self.assertTrue(all(path.startswith(run_b + os.sep) for path in paths_b))

    def test_configured_directory_is_expanded_and_absolute(self):
        expected = os.path.abspath(os.path.expanduser("~/recorders/run-a"))
        actual = get_vllm_recording_dir(
            {RECORDER_DIR_ENV: "~/recorders/run-a"}
        )
        self.assertEqual(actual, expected)

    def test_unconfigured_directory_keeps_legacy_tmp_fallback(self):
        self.assertEqual(get_vllm_recording_dir({}), tempfile.gettempdir())

    def test_all_vllm_flag_sites_use_the_shared_path_helper(self):
        repo_root = Path(__file__).parents[1]
        system_source = (repo_root / "moe_cap/systems/vllm.py").read_text()
        integration_source = (
            repo_root / "moe_cap/extracted_expert_dist/vllm_integration.py"
        ).read_text()

        self.assertNotIn(
            'tempfile.gettempdir(), "vllm_batch_recording.flag"', system_source
        )
        self.assertNotIn(
            'tempfile.gettempdir(),\n                                "vllm_expert_distribution_auto_start.flag"',
            integration_source,
        )
        self.assertGreaterEqual(
            integration_source.count("get_vllm_recording_path("), 2
        )


if __name__ == "__main__":
    unittest.main()
