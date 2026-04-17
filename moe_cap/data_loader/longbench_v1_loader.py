import os
import json
import random
import zipfile
from .base_data_loader import DataLoader
from moe_cap.configs.cap_config import CAPConfig
from typing import List

# English QA subsets with non-empty input field
# Excluded: gov_report, multi_news, passage_count, lcc (no question, summarization/code tasks)
_EN_SUBSETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "trec",
    "triviaqa",
    "samsum",
    "passage_retrieval_en",
    "repobench-p",
    "narrativeqa",
    "qmsum",
]

SEED = 42

# System instructions per subset (goes into system prompt)
_SYSTEM_INSTRUCTIONS = {
    "narrativeqa": "Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.",
    "qasper": 'Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question asks about specific methods or terms, list the exact names mentioned in the article. If the question cannot be answered, write "None". If yes/no question, answer "yes", "no", or "None". Do not provide any explanation.',
    "multifieldqa_en": "Only give me the answer and do not output any other words.",
    "hotpotqa": "Only give me the answer and do not output any other words.",
    "2wikimqa": "Only give me the answer and do not output any other words.",
    "musique": "Only give me the answer and do not output any other words.",
    "trec": "Determine the type of the question. Only output the type.",
    "triviaqa": "Only give me the answer and do not output any other words.",
    "samsum": "Summarize the dialogue into a few short sentences.",
    "passage_retrieval_en": "Answer the question based on the given paragraphs.",
    "repobench-p": "Output only the next single line of code, nothing else.",
    "qmsum": "Answer the query in one or more sentences.",
}

_DEFAULT_SYSTEM = "Output the answer directly without description."

# User content templates per subset (context + question only)
_USER_TEMPLATES = {
    "narrativeqa": "Story:\n{context}\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "Article:\n{context}\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "{context}\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "The following are given passages.\n{context}\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "The following are given passages.\n{context}\n\nQuestion: {input}\nAnswer:",
    "musique": "The following are given passages.\n{context}\n\nQuestion: {input}\nAnswer:",
    "trec": "Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "The following are some examples.\n\n{context}\n\n{input}",
    "passage_retrieval_en": "{context}\n\n{input}",
    "repobench-p": "{context}{input}Next line of code:\n",
    "qmsum": "Transcript:\n{context}\n\nQuery: {input}\nAnswer:",
}

_DEFAULT_USER = "Context:\n{context}\n\nQuestion: {input}\nAnswer:"


def _format_prompt(subset: str, context: str, question: str) -> tuple:
    """Returns (system_instruction, user_content) tuple."""
    system = _SYSTEM_INSTRUCTIONS.get(subset, _DEFAULT_SYSTEM)
    user_template = _USER_TEMPLATES.get(subset, _DEFAULT_USER)
    user = user_template.format(context=context, input=question)
    return system, user


def _ensure_data_dir() -> str:
    """Download and extract LongBench v1 data if not present."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "longbench_v1", "data")
    if os.path.exists(cache_dir) and any(
        f.endswith(".jsonl") for f in os.listdir(cache_dir)
    ):
        return cache_dir
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(cache_dir))
    extracted = os.path.join(os.path.dirname(cache_dir), "data")
    if extracted != cache_dir:
        os.rename(extracted, cache_dir)
    return cache_dir


class LongBenchV1Loader(DataLoader):
    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        data_dir = os.environ.get("LONGBENCH_V1_DATA_DIR") or _ensure_data_dir()

        self.prompts = []
        self.system_prompts = []
        self.targets = []

        if config.dataset_subset:
            subsets = [config.dataset_subset]
        else:
            subsets = _EN_SUBSETS

        MAX_WORDS = 33000  # ~50K tokens, skip very long samples

        all_data = []
        for subset in subsets:
            path = os.path.join(data_dir, f"{subset}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("length", 0) > MAX_WORDS:
                        continue
                    all_data.append(obj)

        # Fixed seed shuffle for deterministic sampling
        rng = random.Random(SEED)
        rng.shuffle(all_data)

        for obj in all_data:
            context = obj["context"]
            question = obj["input"]
            subset = obj.get("dataset", "")
            system, user = _format_prompt(subset, context, question)
            self.system_prompts.append(system)
            self.prompts.append(user)
            self.targets.append(
                {
                    "answers": obj.get("answers", []),
                    "dataset": obj.get("dataset", ""),
                    "all_classes": obj.get("all_classes"),
                }
            )

    def get_input(self) -> List[str]:
        return self.prompts

    def get_target(self) -> List[dict]:
        return self.targets
