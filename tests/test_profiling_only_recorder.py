"""Unit tests for the dense-safe SGLang profiling-only recorder."""

import ast
import json
import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import List

SOURCE_PATH = Path(__file__).parents[1] / "moe_cap" / "systems" / "sglang.py"


def _source_blocks(*names):
    source = SOURCE_PATH.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    blocks = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            blocks[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    assert set(blocks) == set(names), set(names) - set(blocks)
    return blocks


class _RecorderBase:
    @contextmanager
    def disable_this_region(self):
        yield


class _ServerArgs:
    def __init__(self, model_path="Qwen/Qwen3-4B"):
        self.model_path = model_path


def _namespace(save_dir):
    class _EnvVar:
        def get(self):
            return save_dir

    class _Envs:
        SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = _EnvVar()

    namespace = {
        "contextmanager": contextmanager,
        "ExpertDistributionRecorder": _RecorderBase,
        "ServerArgs": _ServerArgs,
        "List": List,
        "_OutputMode": str,
        "logger": logging.getLogger(__name__),
        "envs": _Envs(),
        "os": os,
        "json": json,
    }
    blocks = _source_blocks("_ProfilingOnlyRecorder", "_profiling_only_init_new")
    exec(blocks["_ProfilingOnlyRecorder"], namespace)
    exec(blocks["_profiling_only_init_new"], namespace)
    return namespace


class ProfilingOnlyRecorderTest(unittest.TestCase):
    def test_factory_does_not_require_expert_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = _namespace(tmp)
            recorder = ns["_profiling_only_init_new"](_ServerArgs(), None, 0)
            self.assertIsInstance(recorder, ns["_ProfilingOnlyRecorder"])
            self.assertFalse(recorder.recording)

    def test_start_stop_dump_forward_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = _namespace(tmp)
            recorder = ns["_ProfilingOnlyRecorder"](_ServerArgs(), rank=0)
            recorder.start_record()
            self.assertTrue(recorder.recording)
            recorder.expert_record_list.extend(
                [
                    {
                        "forward_mode": "prefill",
                        "latency": 0.05,
                        "batch_size": 2,
                        "seq_lens_sum": 128,
                    },
                    {
                        "forward_mode": "decode",
                        "latency": 0.01,
                        "batch_size": 2,
                        "seq_lens_sum": 130,
                    },
                ]
            )
            recorder.stop_record()
            self.assertFalse(recorder.recording)
            returned = recorder.dump_record(output_mode="file")
            self.assertEqual(len(returned), 2)
            output = Path(tmp) / "Qwen/Qwen3-4B/expert_distribution_record.jsonl"
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows, returned)
            self.assertEqual([row["forward_mode"] for row in rows], ["prefill", "decode"])

    def test_new_recording_window_clears_stale_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = _namespace(tmp)
            recorder = ns["_ProfilingOnlyRecorder"](_ServerArgs())
            recorder.start_record()
            recorder.expert_record_list.append({"latency": 1.0})
            recorder.stop_record()
            recorder.start_record()
            self.assertEqual(recorder.expert_record_list, [])

    def test_profile_only_patch_is_guarded(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn("if _PROFILING_ONLY:", source)
        self.assertIn(
            "ExpertDistributionRecorder.init_new = staticmethod(_profiling_only_init_new)",
            source,
        )
        self.assertIn("else:\n    ModelRunner.forward = forward_expert_record", source)


if __name__ == "__main__":
    unittest.main()
