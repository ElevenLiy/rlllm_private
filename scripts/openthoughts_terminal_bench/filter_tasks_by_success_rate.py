#!/usr/bin/env python3
"""Filter Harbor eval tasks by success rate for training curation."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Filter tasks by success rate from eval results")
    parser.add_argument("eval_results", help="Path to eval_results.json from _validate_agent")
    parser.add_argument("--min-rate", type=float, default=0.25, help="Minimum success rate (inclusive)")
    parser.add_argument("--max-rate", type=float, default=0.75, help="Maximum success rate (inclusive)")
    parser.add_argument("--output", "-o", default="harbor_filtered_tasks.json", help="Output JSON path")
    args = parser.parse_args()

    with open(args.eval_results) as f:
        data = json.load(f)

    per_task = data.get("per_task_rewards", {})
    task_names = data.get("task_names", [])

    results = []
    for uid, info in per_task.items():
        rewards = info.get("all_rewards", [])
        success_rate = info.get("success_rate", sum(1 for r in rewards if r > 0) / max(len(rewards), 1))
        results.append({
            "uid": uid,
            "rewards": rewards,
            "success_rate": success_rate,
            "best_score": info.get("best_score", max(rewards) if rewards else 0),
        })

    filtered = [t for t in results if args.min_rate <= t["success_rate"] <= args.max_rate]
    filtered.sort(key=lambda x: x["success_rate"])

    all_rates = [t["success_rate"] for t in results]
    output = {
        "filter_params": {"min_rate": args.min_rate, "max_rate": args.max_rate},
        "total_tasks": len(results),
        "filtered_count": len(filtered),
        "summary": {
            "all_zero": sum(1 for sr in all_rates if sr == 0),
            "all_pass": sum(1 for sr in all_rates if sr == 1.0),
            "in_range": len(filtered),
            "avg_success_rate": sum(all_rates) / max(len(all_rates), 1),
        },
        "tasks": filtered,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Total tasks evaluated: {len(results)}")
    print(f"  0% success (too hard):     {output['summary']['all_zero']}")
    print(f"  100% success (too easy):   {output['summary']['all_pass']}")
    print(f"  [{args.min_rate*100:.0f}%-{args.max_rate*100:.0f}%] (good for RL): {len(filtered)}")
    print(f"  Avg success rate:          {output['summary']['avg_success_rate']:.1%}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
