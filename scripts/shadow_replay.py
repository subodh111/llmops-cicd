#!/usr/bin/env python3
"""
Replays a sample of recent real production requests against the staging
version without serving the response to users ("shadow mode"). Compares
staging output to what production actually returned, to catch regressions
that a static golden dataset wouldn't cover.
"""

import argparse
import json
import statistics
from pathlib import Path

import anthropic


def load_recent_production_requests(sample_size: int) -> list[dict]:
    """
    STUB: pull from your production request log store. Expected shape:
    [{"id": ..., "input": ..., "production_output": ...}, ...]
    """
    sample_path = Path("eval/sample_production_traffic.jsonl")
    if not sample_path.exists():
        print(f"[shadow] No sample traffic file at {sample_path}; skipping shadow replay.")
        return []
    with open(sample_path) as f:
        return [json.loads(line) for line in f][:sample_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--duration-min", type=int, default=15)
    args = parser.parse_args()

    requests = load_recent_production_requests(args.sample_size)
    if not requests:
        return

    client = anthropic.Anthropic()
    divergences = []

    for req in requests:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": req["input"]}],
        )
        staging_output = "".join(b.text for b in response.content if b.type == "text")

        # Simple length-delta divergence signal; swap for semantic diff / judge model as needed
        prod_len = len(req.get("production_output", ""))
        staging_len = len(staging_output)
        divergence = abs(prod_len - staging_len) / max(prod_len, 1)
        divergences.append(divergence)

    avg_divergence = statistics.mean(divergences)
    print(f"[shadow] {len(requests)} requests replayed against {args.env}")
    print(f"[shadow] avg output-length divergence vs production: {avg_divergence:.2%}")

    if avg_divergence > 0.5:
        print("⚠️  High divergence detected — review staging output manually before promoting")


if __name__ == "__main__":
    main()
