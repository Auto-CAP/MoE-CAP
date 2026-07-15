#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import glob
import json
import os
from pathlib import Path

from moe_cap.configs import CAPConfig
from moe_cap.data_loader.loader_registry import get_loader_for_task
from moe_cap.utils.arena_hard_judge import (
    evaluate_arena_hard,
    is_degenerate_generation,
    judge_single_question,
    load_baseline_answers,
    merge_failures_as_zero,
)


def aggregate(records, judge_model):
    scores = [float(r.get("score", 0.5)) for r in records]
    return {
        "arena_hard_win_rate": round(sum(scores) / len(scores) * 100, 2),
        "arena_hard_wins": sum(s > 0.5 for s in scores),
        "arena_hard_losses": sum(s < 0.5 for s in scores),
        "arena_hard_ties": sum(s == 0.5 for s in scores),
        "arena_hard_errors": sum(
            r.get("round1_verdict") is None or r.get("round2_verdict") is None
            for r in records
        ),
        "arena_hard_total": len(records),
        "arena_hard_judge_model": judge_model,
        "arena_hard_per_question": records,
    }


async def main(args):
    leaf = Path(args.leaf)
    output_file = Path(glob.glob(str(leaf / "output_data_arena-hard_*.jsonl"))[0])
    metrics_file = Path(glob.glob(str(leaf / "metrics_arena-hard_*.json"))[0])
    rows = [json.loads(x) for x in output_file.read_text().splitlines() if x.strip()]
    assert len(rows) == 256, len(rows)

    cfg = CAPConfig(
        dataset_names=["arena-hard"], metrics=[], model_id=args.model_id,
        num_samples=256, profiling_only=True,
    )
    loader, _ = get_loader_for_task("arena-hard", cfg)
    questions = list(loader.get_input())[:256]
    uids = list(loader.get_uids())[:256]
    baseline = load_baseline_answers(args.baseline)
    baseline_answers = [baseline[uid] for uid in uids]
    assert len(questions) == len(uids) == len(baseline_answers) == 256

    success_indices = [
        i
        for i, row in enumerate(rows)
        if row.get("success")
        and not is_degenerate_generation(row.get("output_tokens", ""))
    ]
    failed_indices = [i for i in range(256) if i not in set(success_indices)]
    answers = [rows[i].get("output_tokens", "") for i in success_indices]
    judged = await evaluate_arena_hard(
        questions=[questions[i] for i in success_indices],
        model_answers=answers,
        baseline_answers=[baseline_answers[i] for i in success_indices],
        judge_api_url=args.api_url,
        judge_model=args.judge_model,
        api_key=os.environ["OPENAI_API_KEY"],
        max_concurrent=10,
    )
    records = judged.pop("arena_hard_per_question")

    for repair_round in range(1, 21):
        missing = [
            j for j, r in enumerate(records)
            if r.get("round1_verdict") is None or r.get("round2_verdict") is None
        ]
        if not missing:
            break
        print(f"Repair round {repair_round}: retrying {len(missing)} incomplete judgments", flush=True)
        for j in missing:
            orig = success_indices[j]
            old = records[j]
            repaired = await judge_single_question(
                question=questions[orig],
                model_answer=rows[orig].get("output_tokens", ""),
                baseline_answer=baseline_answers[orig],
                judge_api_url=args.api_url,
                judge_model=args.judge_model,
                api_key=os.environ["OPENAI_API_KEY"],
            )
            for key in ("index", "question", "model_answer", "baseline_answer"):
                if key in old:
                    repaired[key] = old[key]
            records[j] = repaired

    judge_result = aggregate(records, args.judge_model)
    failed_records = [
        {
            "index": i,
            "question": questions[i][:500],
            "uid": uids[i],
            "model_answer": "",
            "baseline_answer": baseline_answers[i][:500],
            "generation_error": (
                "empty or degenerate generation"
                if rows[i].get("success")
                else rows[i].get("error", "generation failed")
            ),
        }
        for i in failed_indices
    ]
    final = merge_failures_as_zero(
        judge_result=judge_result,
        success_indices=success_indices,
        failed_records=failed_records,
        total=256,
    )
    per_question = final.pop("arena_hard_per_question")
    incomplete = sum(
        (not r.get("generation_failed"))
        and (r.get("round1_verdict") is None or r.get("round2_verdict") is None)
        for r in per_question
    )
    if incomplete:
        raise RuntimeError(f"{incomplete} incomplete judge records remain")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    judge_file = leaf / f"arena_hard_judge_arena-hard_{stamp}.jsonl"
    summary_file = leaf / f"arena_hard_judge_summary_arena-hard_{stamp}.json"
    with judge_file.open("w") as f:
        for row in per_question:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "arena_hard_win_rate": final["arena_hard_win_rate"],
        "arena_hard_wins": final["arena_hard_wins"],
        "arena_hard_losses": final["arena_hard_losses"],
        "arena_hard_ties": final["arena_hard_ties"],
        "arena_hard_errors": final.get("arena_hard_errors", 0),
        "arena_hard_total": 256,
        "arena_hard_judge_model": args.judge_model,
        "arena_hard_baseline_model": "gpt-4-0613",
        "run_id": args.run_id,
    }
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")

    metrics = json.loads(metrics_file.read_text())
    metrics["quality"] = {
        "acc": summary["arena_hard_win_rate"] / 100,
        "total": 256,
        "arena_hard_win_rate": summary["arena_hard_win_rate"],
        "arena_hard_wins": summary["arena_hard_wins"],
        "arena_hard_losses": summary["arena_hard_losses"],
        "arena_hard_ties": summary["arena_hard_ties"],
        "arena_hard_errors": summary["arena_hard_errors"],
        "arena_hard_total": 256,
        "arena_hard_judge_model": args.judge_model,
        "arena_hard_baseline_model": "gpt-4-0613",
        "arena_hard_judge_log": judge_file.name,
        "posthoc_eval": True,
        "posthoc_eval_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
    print("VERIFY_COMPLETE", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        + "/chat/completions",
    )
    asyncio.run(main(parser.parse_args()))
