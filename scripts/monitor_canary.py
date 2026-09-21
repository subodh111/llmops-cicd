#!/usr/bin/env python3
"""
Monitors a canary deployment's live metrics for a fixed window and triggers
an automatic rollback if any threshold is breached. In production this would
pull real metrics from your observability stack (Datadog, Prometheus,
LangSmith, etc). Here it's stubbed with a pluggable `fetch_live_metrics`
function so you can wire in your actual source.
"""

import argparse
import subprocess
import sys
import time


def fetch_live_metrics(env: str) -> dict:
    """
    STUB: replace with a real call to your monitoring/observability backend.
    Must return: avg_score, avg_cost_usd, p95_latency_ms, error_rate, refusal_rate
    """
    # Example of what a real implementation would do:
    #   resp = requests.get(f"{METRICS_API}/canary/{env}/summary")
    #   return resp.json()
    raise NotImplementedError(
        "Wire fetch_live_metrics() to your metrics backend (Datadog/Prometheus/LangSmith/etc)."
    )


def rollback(env: str):
    print(f"[rollback] Rolling back canary in {env} to previous stable version")
    subprocess.run(["python", "scripts/deploy.py", "--env", env, "--version", "previous-stable", "--traffic-pct", "0"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--duration-min", type=int, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--max-cost", type=float, required=True)
    parser.add_argument("--max-latency-ms", type=float, required=True)
    parser.add_argument("--rollback-on-breach", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--poll-interval-sec", type=int, default=60)
    args = parser.parse_args()

    end_time = time.time() + args.duration_min * 60
    checks = 0

    while time.time() < end_time:
        checks += 1
        try:
            metrics = fetch_live_metrics(args.env)
        except NotImplementedError as e:
            print(f"[monitor] {e}")
            print("[monitor] Skipping live monitoring in this reference implementation.")
            return

        breaches = []
        if metrics["avg_score"] < args.score_threshold:
            breaches.append(f"score {metrics['avg_score']:.3f} < {args.score_threshold}")
        if metrics["avg_cost_usd"] > args.max_cost:
            breaches.append(f"cost ${metrics['avg_cost_usd']:.4f} > ${args.max_cost}")
        if metrics["p95_latency_ms"] > args.max_latency_ms:
            breaches.append(f"p95 latency {metrics['p95_latency_ms']:.0f}ms > {args.max_latency_ms}ms")
        if metrics.get("error_rate", 0) > 0.02:
            breaches.append(f"error rate {metrics['error_rate']:.1%} > 2%")

        print(f"[monitor] check #{checks}: {metrics}")

        if breaches:
            print("❌ Canary breached thresholds:")
            for b in breaches:
                print(f"  - {b}")
            if args.rollback_on_breach:
                rollback(args.env)
                sys.exit(1)

        time.sleep(args.poll_interval_sec)

    print(f"✅ Canary healthy for full {args.duration_min}min window")


if __name__ == "__main__":
    main()
