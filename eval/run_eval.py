```python
#!/usr/bin/env python3
"""
LLMOps evaluation harness.

Runs a golden dataset against a prompt/model version, scores outputs on
correctness, safety, cost, and latency, compares against a stored baseline,
and exits non-zero if any hard gate fails. Designed to be called from CI.

Usage:
    python run_eval.py \
        --dataset eval/golden_dataset.jsonl \
        --prompt prompts/current_prompt.yaml \
        --model claude-sonnet-4-6 \
        --baseline eval/baseline_scores.json \
        --score-threshold 0.85 \
        --max-regression 0.05 \
        --max-cost 0.05 \
        --max-latency-ms 4000 \
        --output eval/results/latest_run.json
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import yaml


# --- Simple denylist for a safety gate. Replace with a real classifier /
# moderation model in production; this is illustrative only. ---
UNSAFE_MARKERS = [
    "i cannot verify but here is",
]


@dataclass
class CaseResult:
    id: str
    input: str
    expected: str
    actual: str
    score: float
    cost_usd: float
    latency_ms: float
    safety_flag: bool


def load_dataset(path: str) -> list[dict]:
    """Load evaluation cases from a JSONL dataset."""
    cases = []

    with open(path) as file:
        for line in file:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def load_prompt(path: str) -> dict:
    """Load prompt configuration from YAML."""
    with open(path) as file:
        return yaml.safe_load(file)


def render_prompt(template: str, variables: dict) -> str:
    """Render a user prompt template with case variables."""
    return template.format(**variables)


def call_model(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user_msg: str,
) -> tuple[str, float, float]:
    """Return response text, estimated cost, and latency in milliseconds."""
    start = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    latency_ms = (time.time() - start) * 1000

    text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    # Rough cost estimate. Replace rates with current model pricing.
    input_rate_per_mtok = 3.00
    output_rate_per_mtok = 15.00

    cost_usd = (
        response.usage.input_tokens / 1_000_000 * input_rate_per_mtok
        + response.usage.output_tokens / 1_000_000 * output_rate_per_mtok
    )

    return text, cost_usd, latency_ms


def score_output(
    expected: str,
    actual: str,
    judge_client: anthropic.Anthropic,
    model: str,
) -> float:
    """
    Score semantic similarity between expected and actual answers.

    Uses an LLM-as-judge approach and falls back to a simple containment
    heuristic if the judge response cannot be parsed or the API call fails.
    """
    judge_prompt = f"""Rate how well the ACTUAL answer matches the EXPECTED answer on a 0.0-1.0 scale.
Consider semantic correctness, not exact wording. Reply with only a number.

EXPECTED: {expected}
ACTUAL: {actual}"""

    try:
        response = judge_client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": judge_prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        return max(0.0, min(1.0, float(text)))

    except (ValueError, TypeError, anthropic.APIError):
        return 1.0 if expected.lower() in actual.lower() else 0.0


def check_safety(text: str) -> bool:
    """Return True if a safety violation is detected."""
    lowered = text.lower()
    return any(marker in lowered for marker in UNSAFE_MARKERS)


def run_evaluation(
    dataset_path: str,
    prompt_path: str,
    model: str,
) -> list[CaseResult]:
    """Run the complete evaluation against the golden dataset."""
    client = anthropic.Anthropic()

    prompt_cfg = load_prompt(prompt_path)
    system_prompt = prompt_cfg["system"]
    template = prompt_cfg["user_template"]

    results = []

    for case in load_dataset(dataset_path):
        user_msg = render_prompt(template, case["variables"])

        actual, cost, latency = call_model(
            client,
            model,
            system_prompt,
            user_msg,
        )

        score = score_output(
            case["expected"],
            actual,
            client,
            model,
        )

        safety_flag = check_safety(actual)

        results.append(
            CaseResult(
                id=case["id"],
                input=user_msg,
                expected=case["expected"],
                actual=actual,
                score=score,
                cost_usd=cost,
                latency_ms=latency,
                safety_flag=safety_flag,
            )
        )

    return results


def summarize(
    results: list[CaseResult],
    baseline_path: str,
    thresholds: dict,
) -> dict:
    """Calculate evaluation metrics and compare them against the baseline."""
    overall_score = statistics.mean(r.score for r in results)
    avg_cost = statistics.mean(r.cost_usd for r in results)

    latencies = sorted(r.latency_ms for r in results)

    p95_latency = (
        latencies[int(len(latencies) * 0.95) - 1]
        if latencies
        else 0
    )

    safety_violations = sum(
        1 for result in results if result.safety_flag
    )

    # Default to the current score when no baseline exists.
    baseline_score = overall_score

    if Path(baseline_path).exists():
        with open(baseline_path) as file:
            baseline = json.load(file)
            baseline_score = baseline.get(
                "overall_score",
                overall_score,
            )

    regression_delta = overall_score - baseline_score

    return {
        "overall_score": overall_score,
        "baseline_score": baseline_score,
        "regression_delta": regression_delta,
        "avg_cost_usd": avg_cost,
        "p95_latency_ms": p95_latency,
        "safety_violations": safety_violations,
        "num_cases": len(results),
        "thresholds": thresholds,
        "cases": [asdict(result) for result in results],
    }


def evaluate_gates(summary: dict) -> list[str]:
    """Return failure reasons; an empty list means all gates passed."""
    failures = []
    thresholds = summary["thresholds"]

    if summary["overall_score"] < thresholds["score_threshold"]:
        failures.append(
            f"Overall score {summary['overall_score']:.3f} "
            f"below threshold {thresholds['score_threshold']}"
        )

    if summary["regression_delta"] < -thresholds["max_regression"]:
        failures.append(
            f"Regression of {-summary['regression_delta']:.3f} "
            f"exceeds max allowed {thresholds['max_regression']}"
        )

    if summary["avg_cost_usd"] > thresholds["max_cost"]:
        failures.append(
            f"Avg cost ${summary['avg_cost_usd']:.4f} "
            f"exceeds max ${thresholds['max_cost']}"
        )

    if summary["p95_latency_ms"] > thresholds["max_latency_ms"]:
        failures.append(
            f"P95 latency {summary['p95_latency_ms']:.0f}ms "
            f"exceeds max {thresholds['max_latency_ms']}ms"
        )

    if summary["safety_violations"] > 0:
        failures.append(
            f"{summary['safety_violations']} safety violation(s) detected"
        )

    return failures


def main() -> None:
    """Run the evaluation harness."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--max-regression", type=float, required=True)
    parser.add_argument("--max-cost", type=float, required=True)
    parser.add_argument("--max-latency-ms", type=float, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    thresholds = {
        "score_threshold": args.score_threshold,
        "max_regression": args.max_regression,
        "max_cost": args.max_cost,
        "max_latency_ms": args.max_latency_ms,
    }

    print(
        f"Running eval: model={args.model} "
        f"dataset={args.dataset}"
    )

    results = run_evaluation(
        args.dataset,
        args.prompt,
        args.model,
    )

    summary = summarize(
        results,
        args.baseline,
        thresholds,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(summary, file, indent=2)

    print(f"Overall score: {summary['overall_score']:.3f}")
    print(
        f"Regression vs baseline: "
        f"{summary['regression_delta']:.3f}"
    )
    print(
        f"Avg cost/request: "
        f"${summary['avg_cost_usd']:.4f}"
    )
    print(
        f"P95 latency: "
        f"{summary['p95_latency_ms']:.0f}ms"
    )
    print(
        f"Safety violations: "
        f"{summary['safety_violations']}"
    )

    failures = evaluate_gates(summary)

    if failures:
        print("\n❌ EVAL GATE FAILED:")

        for failure in failures:
            print(f"  - {failure}")

        sys.exit(1)

    print("\n✅ All eval gates passed")


if __name__ == "__main__":
    main()
```
