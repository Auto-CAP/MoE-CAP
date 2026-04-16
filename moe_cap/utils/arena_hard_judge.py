"""
Arena-Hard LLM-as-a-judge evaluation.

Implements pairwise comparison of model outputs against a baseline,
following the official lm-sys/arena-hard-auto methodology.
"""

import asyncio
import aiohttp
import json
import re
from typing import Any, Dict, List, Optional


# Official judge system prompt (from lm-sys/arena-hard-auto)
JUDGE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two "
    "AI assistants to the user prompt displayed below. You will be given assistant A's answer and "
    "assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your "
    "verdicts after providing your explanation. Your final verdict should strictly follow this format: "
    '"[[A>>B]]" if assistant A is much better, '
    '"[[A>B]]" if assistant A is slightly better, '
    '"[[A=B]]" for a tie, '
    '"[[B>A]]" if assistant B is slightly better, '
    '"[[B>>A]]" if assistant B is much better.'
)

# Verdict label to score mapping (from official show_result.py)
# When model is assistant A:
VERDICT_TO_SCORE_A = {
    "A>>B": 1.0,
    "A>B": 1.0,
    "A=B": 0.5,
    "B>A": 0.0,
    "B>>A": 0.0,
}
# Weight for significant wins/losses (>>)
SIGNIFICANT_WEIGHT = 3


def parse_verdict(judge_output: str) -> Optional[str]:
    """Extract verdict like 'A>>B' from judge output containing [[A>>B]]."""
    match = re.search(r"\[\[([AB][>=<]+[AB])\]\]", judge_output)
    if match:
        return match.group(1)
    return None


def verdict_to_score(verdict: Optional[str], model_is_a: bool) -> float:
    """Convert verdict to score for the model being evaluated.

    Args:
        verdict: Parsed verdict string like 'A>B', 'B>>A', etc.
        model_is_a: True if the model under test is assistant A.

    Returns:
        Score between 0.0 and 1.0 for the model.
    """
    if verdict is None:
        return 0.5  # treat parse failures as ties

    score = VERDICT_TO_SCORE_A.get(verdict, 0.5)

    if not model_is_a:
        score = 1.0 - score

    return score


async def call_judge_api(
    question: str,
    answer_a: str,
    answer_b: str,
    judge_api_url: str,
    judge_model: str,
    timeout: int = 300,
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """Call the judge LLM API for a single pairwise comparison with retry."""
    user_prompt = (
        f"<|User Prompt|>\n{question}\n\n"
        f"<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n"
        f"<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
    )

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.post(
                    judge_api_url,
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2**attempt)


async def judge_single_question(
    question: str,
    model_answer: str,
    baseline_answer: str,
    judge_api_url: str,
    judge_model: str,
    api_key: Optional[str] = None,
) -> Dict:
    """Judge a single question with two rounds (swapped order).

    Round 1: baseline=A, model=B
    Round 2: model=A, baseline=B

    Returns dict with verdicts and combined score.
    """
    # Round 1: baseline as A, model as B
    try:
        r1_output = await call_judge_api(
            question,
            baseline_answer,
            model_answer,
            judge_api_url,
            judge_model,
            api_key=api_key,
        )
        r1_verdict = parse_verdict(r1_output)
        r1_score = verdict_to_score(r1_verdict, model_is_a=False)
    except Exception:
        r1_verdict = None
        r1_score = 0.5

    # Round 2: model as A, baseline as B
    try:
        r2_output = await call_judge_api(
            question,
            model_answer,
            baseline_answer,
            judge_api_url,
            judge_model,
            api_key=api_key,
        )
        r2_verdict = parse_verdict(r2_output)
        r2_score = verdict_to_score(r2_verdict, model_is_a=True)
    except Exception:
        r2_verdict = None
        r2_score = 0.5

    combined_score = (r1_score + r2_score) / 2.0

    return {
        "round1_verdict": r1_verdict,
        "round2_verdict": r2_verdict,
        "round1_score": r1_score,
        "round2_score": r2_score,
        "score": combined_score,
    }


async def evaluate_arena_hard(
    questions: List[str],
    model_answers: List[str],
    baseline_answers: List[str],
    judge_api_url: str,
    judge_model: str = "gpt-4.1",
    max_concurrent: int = 10,
    api_key: Optional[str] = None,
) -> Dict:
    """Full Arena-Hard evaluation pipeline.

    Args:
        questions: List of user prompts
        model_answers: Model's generated answers
        baseline_answers: Baseline model's answers
        judge_api_url: OpenAI-compatible chat API endpoint for judge (e.g. https://api.openai.com/v1/chat/completions)
        judge_model: Judge model name
        max_concurrent: Max concurrent judge API calls

    Returns:
        Dict with win_rate, per-question scores, and summary stats
    """
    assert len(questions) == len(model_answers) == len(baseline_answers), (
        f"Length mismatch: questions={len(questions)}, model={len(model_answers)}, baseline={len(baseline_answers)}"
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_with_semaphore(q, m, b):
        async with semaphore:
            return await judge_single_question(
                q,
                m,
                b,
                judge_api_url,
                judge_model,
                api_key=api_key,
            )

    tasks = [
        judge_with_semaphore(q, m, b)
        for q, m, b in zip(questions, model_answers, baseline_answers)
    ]

    print(f"Running Arena-Hard judge ({judge_model}) on {len(tasks)} questions...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = []
    wins = 0
    losses = 0
    ties = 0
    errors = 0

    for r in results:
        if isinstance(r, BaseException):
            scores.append(0.5)
            errors += 1
            continue
        if isinstance(r, dict):
            s = float(r.get("score", 0.5))
        else:
            s = 0.5
            errors += 1
        scores.append(s)
        if s > 0.5:
            wins += 1
        elif s < 0.5:
            losses += 1
        else:
            ties += 1

    win_rate = sum(scores) / len(scores) * 100 if scores else 0

    return {
        "arena_hard_win_rate": round(win_rate, 2),
        "arena_hard_wins": wins,
        "arena_hard_losses": losses,
        "arena_hard_ties": ties,
        "arena_hard_errors": errors,
        "arena_hard_total": len(questions),
        "arena_hard_judge_model": judge_model,
    }


def load_baseline_answers(path: str) -> Dict[str, str]:
    """Load baseline answers from a JSONL file.

    Expected format: each line is a JSON object with at least:
        {"question_id": "...", "choices": [{"turns": [{"content": "..."}]}]}
    OR simpler format:
        {"question_id": "...", "answer": "..."}

    Returns dict mapping question_id -> answer text
    """
    answers = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id", obj.get("uid", obj.get("id", "")))
            # Official arena-hard format: messages array with role/content
            if "messages" in obj:
                for msg in obj["messages"]:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, dict):
                            answers[qid] = content.get("answer", "")
                        else:
                            answers[qid] = content
                        break
            elif "choices" in obj:
                turns = obj["choices"][0].get("turns", [])
                if turns:
                    answers[qid] = turns[0].get("content", "")
            elif "answer" in obj:
                answers[qid] = obj["answer"]
            elif "model_output" in obj:
                answers[qid] = obj["model_output"]
    return answers
