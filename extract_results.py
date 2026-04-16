#!/usr/bin/env python3
"""Extract per-instance tau2-bench results from a results.json simulation file.

Reads a tau2-bench results.json, extracts per-task metrics for the agent model,
and merges them into an output JSON file. Can be called repeatedly with different
results.json files to accumulate data for multiple models.

Usage:
    python extract_results.py <results.json> [--output <output.json>] [--model-name <name>]

Examples:
    # Extract from a single run (model name auto-detected from results.json)
    python extract_results.py data/simulations/20260323_164600_.../results.json

    # Accumulate multiple models into one file
    python extract_results.py data/simulations/run1/results.json -o scores.json
    python extract_results.py data/simulations/run2/results.json -o scores.json

    # Override the model name
    python extract_results.py data/simulations/run1/results.json --model-name gpt-5.2
"""

import argparse
import json
import os
from collections import defaultdict


def parse_model_name(info: dict) -> str:
    """Extract a short model name from info.user_info.llm (e.g. 'anthropic/claude-sonnet-4-6' -> 'claude-sonnet-4-6')."""
    raw = info.get("user_info", {}).get("llm", "unknown")
    # Strip provider prefix like 'azure/', 'anthropic/', etc.
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def extract_domain(info: dict) -> str:
    """Extract the domain name from info.environment_info."""
    return info.get("environment_info", {}).get("domain_name", "unknown")


def build_task_lookup(tasks: list) -> dict:
    """Build a dict of task_id -> task metadata."""
    lookup = {}
    for task in tasks:
        tid = str(task["id"])
        desc = task.get("description", {})
        scenario = task.get("user_scenario", {})
        instructions = scenario.get("instructions", {})

        lookup[tid] = {
            "purpose": desc.get("purpose", ""),
            "reason_for_call": instructions.get("reason_for_call", ""),
            "known_info": instructions.get("known_info", ""),
            "task_instructions": instructions.get("task_instructions", ""),
        }
    return lookup


def aggregate_simulations(simulations: list) -> dict:
    """Group simulations by task_id and compute per-task aggregates."""
    by_task = defaultdict(list)
    for sim in simulations:
        by_task[str(sim["task_id"])].append(sim)

    results = {}
    for task_id, sims in by_task.items():
        rewards = [s["reward_info"]["reward"] for s in sims]
        durations = [s["duration"] for s in sims]
        agent_costs = [s["agent_cost"] for s in sims]
        msg_counts = [len(s.get("messages") or []) for s in sims]
        terminations = [s.get("termination_reason", "") for s in sims]

        num_trials = len(sims)
        num_passed = sum(1 for r in rewards if r == 1.0)

        results[task_id] = {
            "resolved": num_passed > num_trials / 2,  # majority vote
            "reward": round(sum(rewards) / num_trials, 4),
            "num_trials": num_trials,
            "pass_count": num_passed,
            "trial_rewards": rewards,
            "avg_duration": round(sum(durations) / num_trials, 2),
            "avg_agent_cost": round(sum(agent_costs) / num_trials, 6),
            "avg_messages": round(sum(msg_counts) / num_trials, 1),
            "termination_reasons": terminations,
        }
    return results


def extract(results_path: str, model_name: str = None) -> tuple:
    """Extract per-instance data from a results.json file.

    Returns (domain, model_name, instances_dict) where instances_dict maps
    task_id -> {"prompt": ..., "purpose": ..., model_name: {...}}
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    info = data["info"]
    if model_name is None:
        model_name = parse_model_name(info)

    domain = extract_domain(info)
    task_lookup = build_task_lookup(data["tasks"])
    task_results = aggregate_simulations(data["simulations"])

    instances = {}
    for task_id in sorted(task_lookup.keys(), key=int):
        meta = task_lookup[task_id]
        instance_key = f"{domain}_{task_id}"

        instance = {
            "prompt": meta["reason_for_call"],
            "purpose": meta["purpose"],
            "task_instructions": meta["task_instructions"],
            "domain": domain,
        }

        if task_id in task_results:
            instance[model_name] = task_results[task_id]

        instances[instance_key] = instance

    return domain, model_name, instances


def merge_into_output(existing: dict, new_instances: dict, model_name: str) -> dict:
    """Merge new model results into existing output, preserving other models' data."""
    for instance_key, new_data in new_instances.items():
        if instance_key not in existing:
            existing[instance_key] = {}

        entry = existing[instance_key]

        # Update/preserve prompt-level fields
        for field in ("prompt", "purpose", "task_instructions", "domain"):
            if field in new_data:
                entry[field] = new_data[field]

        # Add/update model-specific data
        if model_name in new_data:
            entry[model_name] = new_data[model_name]

    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-instance tau2-bench results from results.json"
    )
    parser.add_argument(
        "results_json",
        help="Path to a tau2-bench results.json file",
    )
    parser.add_argument(
        "-o", "--output",
        default="tau2_bench_instances.json",
        help="Output JSON file (default: tau2_bench_instances.json). "
             "Will be created or updated with new model data.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the model name (default: auto-detected from results.json)",
    )
    args = parser.parse_args()

    # Load existing output if present
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = json.load(f)
    else:
        existing = {}

    domain, model_name, instances = extract(args.results_json, args.model_name)

    merged = merge_into_output(existing, instances, model_name)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    # Print summary
    task_results = {k: v[model_name] for k, v in instances.items() if model_name in v}
    total = len(task_results)
    resolved = sum(1 for v in task_results.values() if v["resolved"])
    avg_reward = sum(v["reward"] for v in task_results.values()) / total if total else 0

    print(f"Domain:     {domain}")
    print(f"Model:      {model_name}")
    print(f"Tasks:      {total}")
    print(f"Resolved:   {resolved}/{total} ({resolved/total*100:.1f}%)")
    print(f"Avg Reward: {avg_reward:.3f}")
    print(f"Output:     {args.output}")


if __name__ == "__main__":
    main()
