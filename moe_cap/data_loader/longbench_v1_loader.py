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

# Official prompt templates from THUDM/LongBench (dataset2prompt.json)
_PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "None". If the question is a yes/no question, answer "yes", "no", or "None". Do not provide any explanation.\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "None". If the question is a yes/no question, answer "yes", "no", or "None". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_retrieval_en": 'You are given several paragraphs from Wikipedia. For each, a paragraph ID is given by "Paragraph N:". Answer the question based on the information given.\n\n{context}\n\n{input}',
    "repobench-p": "Please complete the code given below.\n{context}{input}Next line of code:\n",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
}

_DEFAULT_PROMPT = "Answer the question based on the given context. Only give me the answer and do not output any other words.\n\nContext: {context}\n\nQuestion: {input}\nAnswer:"


def _format_prompt(subset: str, context: str, question: str) -> str:
    template = _PROMPTS.get(subset, _DEFAULT_PROMPT)
    return template.format(context=context, input=question)


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
            prompt = _format_prompt(subset, context, question)
            self.prompts.append(prompt)
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
