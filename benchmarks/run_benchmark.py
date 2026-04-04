"""Benchmark suite — runs 20 prompts through the agent and reports metrics."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import Settings
from agent.agent import Agent
from agent.schemas import AgentResult


async def run_benchmark() -> None:
    prompts_file = Path(__file__).parent / "prompts.json"
    prompts = json.loads(prompts_file.read_text())

    settings = Settings()
    settings.models.triage = "gemma4"
    settings.models.planner = "gemma4"
    settings.models.executor = "gemma4"
    settings.models.critic_light = "gemma4"
    settings.logging.trace_file = "./logs/benchmark_trace.jsonl"
    settings.recovery.state_file = "./state/benchmark_state.json"

    agent = Agent(settings)
    await agent.initialize()

    results: list[dict] = []
    total_start = time.monotonic()

    print(f"Running {len(prompts)} benchmarks...\n")
    print(f"{'#':>3} {'Cat':<12} {'Time':>6} {'Steps':>5} {'Score':>6} {'Path':<8} Prompt")
    print("-" * 90)

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        category = item["category"]
        start = time.monotonic()

        try:
            result = await agent.run(prompt)
            elapsed = time.monotonic() - start

            scores = [r.score.final_score for r in result.results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            n_steps = len(result.results)

            # Detect path
            path = "simple" if n_steps == 1 and result.results[0].step.id == 0 else "complex"

            entry = {
                "index": i + 1,
                "category": category,
                "prompt": prompt[:80],
                "elapsed_s": round(elapsed, 1),
                "steps": n_steps,
                "avg_score": round(avg_score, 1),
                "scores": scores,
                "path": path,
                "goal": result.goal[:80],
                "success": True,
            }

            print(
                f"{i+1:>3} {category:<12} {elapsed:>5.1f}s {n_steps:>5} {avg_score:>5.1f} {path:<8} {prompt[:50]}"
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            entry = {
                "index": i + 1,
                "category": category,
                "prompt": prompt[:80],
                "elapsed_s": round(elapsed, 1),
                "success": False,
                "error": str(exc)[:100],
            }
            print(
                f"{i+1:>3} {category:<12} {elapsed:>5.1f}s {'FAIL':>5} {'':>6} {'':>8} {prompt[:50]}"
            )

        results.append(entry)

    total_elapsed = time.monotonic() - total_start
    await agent.shutdown()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"Total prompts: {len(results)}")
    print(f"Successes: {len(successes)}/{len(results)}")
    print(f"Failures: {len(failures)}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Avg time per prompt: {total_elapsed / len(results):.1f}s")

    if successes:
        all_scores = [r["avg_score"] for r in successes]
        print(f"Avg critic score: {sum(all_scores) / len(all_scores):.1f}/10")

        simple_count = sum(1 for r in successes if r.get("path") == "simple")
        complex_count = sum(1 for r in successes if r.get("path") == "complex")
        print(f"Simple path: {simple_count}, Complex path: {complex_count}")

        simple_times = [r["elapsed_s"] for r in successes if r.get("path") == "simple"]
        complex_times = [r["elapsed_s"] for r in successes if r.get("path") == "complex"]
        if simple_times:
            print(f"Avg simple latency: {sum(simple_times) / len(simple_times):.1f}s")
        if complex_times:
            print(f"Avg complex latency: {sum(complex_times) / len(complex_times):.1f}s")

    # By category
    print("\nBy category:")
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in successes if r["category"] == cat]
        if cat_results:
            cat_scores = [r["avg_score"] for r in cat_results]
            cat_times = [r["elapsed_s"] for r in cat_results]
            print(
                f"  {cat:<12} {len(cat_results)}/{sum(1 for r in results if r['category'] == cat)} ok, "
                f"avg score {sum(cat_scores)/len(cat_scores):.1f}, "
                f"avg time {sum(cat_times)/len(cat_times):.1f}s"
            )

    # Save results
    output_file = Path(__file__).parent / "results.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_benchmark())
