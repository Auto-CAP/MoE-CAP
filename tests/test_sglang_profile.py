import os
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from moe_cap.configs import CAPConfig

class FakeModelInfo:
    def __init__(self, config):
        # emulate minimal hf_config and config attr
        self.hf_config = {"model_type": "fake"}
        self.config = SimpleNamespace(hidden_size=512, num_hidden_layers=6, vocab_size=50257)
    def get_moe_info(self):
        return {"ffn_dim": 2048, "num_experts_per_layer": 16, "moe_top_k": 2}
    def get_attention_info(self):
        return {"num_key_value_heads": 8, "head_dim": 64}
    def get_architecture_info(self):
        return {"hidden_size": 512, "num_hidden_layers": 6, "vocab_size": 50257}
    def get_model_precision_bits(self):
        return 2.0

class FakeGSM8KLoader:
    def __init__(self, config):
        self.config = config
    def get_input(self):
        # return a tiny set of questions
        return ["What is 1+1?", "What is 2+3?"]

class DummyResponse:
    def __init__(self, out_dir):
        self.out_dir = out_dir
    def raise_for_status(self):
        return None

def fake_post_factory(out_dir):
    """Return a fake requests.post that writes a dump file when dump_expert_distribution_record is called."""
    def fake_post(url, *args, **kwargs):
        # emulate endpoints: start, stop, dump
        if 'dump_expert_distribution_record' in url:
            # create a minimal jsonl file at out_dir/expert_distribution_record.jsonl
            p = os.path.join(out_dir, 'expert_distribution_record.jsonl')
            with open(p, 'w', encoding='utf-8') as f:
                # write two records that the metrics function can process
                f.write(json.dumps({"expert_activation": 1, "latency": 1.0, "seq_lens_sum": 10, "batch_size": 1, "forward_mode": "prefill"}) + '\n')
                f.write(json.dumps({"expert_activation": 1, "latency": 0.5, "seq_lens_sum": 5, "batch_size": 1, "forward_mode": "decoding"}) + '\n')
        return DummyResponse(out_dir)
    return fake_post

class TestSGLangProfileFlow(unittest.TestCase):
    def test_run_multiple_datasets_with_mocks(self):
        # create temp output dir
        with tempfile.TemporaryDirectory() as tmpdir:
            cap_cfg = CAPConfig(dataset_names=["gsm8k", "gsm8k"], metrics=[], model_id="fake-model", precision="bfloat16")

            # Patch HFModelInfoRetriever; import the runner module and monkeypatch
            with patch('moe_cap.model_loader.HFModelInfoRetriever', FakeModelInfo):
                import importlib
                mod = importlib.import_module('moe_cap.runner.sglang_profile')

                # monkeypatch the registry function and requests.post on the imported module
                mod.get_loader_for_task = lambda task, cfg: (FakeGSM8KLoader(cfg), 256)
                mod.requests.post = fake_post_factory(tmpdir)
                # Prevent RuntimeEndpoint from making network calls and make set_default_backend a no-op
                mod.RuntimeEndpoint = lambda url: SimpleNamespace(url=url)
                mod.set_default_backend = lambda backend: None

                # import analyzer from the already-loaded module
                SGLangMoEActivationAnalyzer = mod.SGLangMoEActivationAnalyzer

                # instantiate analyzer with mocked model info and loader
                analyzer = SGLangMoEActivationAnalyzer(config=cap_cfg, output_dir=tmpdir)

                # replace the run_sgl.run_batch with a dummy that returns quickly
                analyzer.run_sgl = SimpleNamespace(run_batch=lambda *a, **k: [])

                # run should complete without contacting network; it will create per-dataset metrics files
                analyzer.run(port=12345)

                # verify output metrics files exist for each dataset under model dir
                model_dir = os.path.join(tmpdir, analyzer.get_model_simple_name())
                self.assertTrue(os.path.isdir(model_dir))
                for ds in cap_cfg.dataset_names:
                    metrics_path = os.path.join(model_dir, f'cap_metrics_{ds}.json')
                    self.assertTrue(os.path.exists(metrics_path), f"Missing metrics for {ds}")
                    # basic sanity of metrics content
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.assertIn('prefill_smbu', data)

if __name__ == '__main__':
    unittest.main()
