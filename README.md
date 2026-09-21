# LLMOps CI/CD Reference Implementation

A working GitHub Actions pipeline for shipping LLM-powered features safely:
lint → automated eval gate → staging shadow traffic → manual approval →
canary rollout with auto-rollback → full production promotion.

## Structure

```
.github/workflows/llmops-cicd.yml   # the pipeline itself
prompts/current_prompt.yaml         # versioned prompt template
eval/
  golden_dataset.jsonl              # fixed regression test cases
  baseline_scores.json              # last known-good scores to diff against
  run_eval.py                       # scoring + gating logic
  validate_prompts.py               # schema/lint check for prompt files
scripts/
  deploy.py                         # writes routing config (traffic %) per env
  shadow_replay.py                  # replays real traffic against staging
  monitor_canary.py                 # watches live metrics, auto-rolls back
```

## Local usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Validate prompt files
python eval/validate_prompts.py --prompts-dir prompts/

# Run the eval gate locally (same command CI runs)
python eval/run_eval.py \
  --dataset eval/golden_dataset.jsonl \
  --prompt prompts/current_prompt.yaml \
  --model claude-sonnet-4-6 \
  --baseline eval/baseline_scores.json \
  --score-threshold 0.85 \
  --max-regression 0.05 \
  --max-cost 0.05 \
  --max-latency-ms 4000 \
  --output eval/results/latest_run.json
```

Exit code is non-zero if any gate fails — this is what CI checks.

## What you need to wire up before this is production-ready

1. **`ANTHROPIC_API_KEY`** as a GitHub Actions secret.
2. **`scripts/deploy.py`** — replace the JSON file stub with your actual
   feature-flag service, config store, or traffic router (e.g. LaunchDarkly,
   a k8s ConfigMap, an API gateway weighted route).
3. **`scripts/monitor_canary.py` → `fetch_live_metrics()`** — connect to your
   real observability backend (Datadog, Prometheus, LangSmith, etc.) instead
   of the `NotImplementedError` stub.
4. **`scripts/shadow_replay.py` → `load_recent_production_requests()`** —
   point at your production request log store instead of the sample file.
5. **`eval/golden_dataset.jsonl`** — replace the 5 sample cases with a real,
   curated set (start with 50-200; grow it from production failures over
   time).
6. **Baseline refresh** — after each successful prod deploy, update
   `eval/baseline_scores.json` with the new scores so future PRs diff
   against current reality, not a stale baseline.
7. **Safety gate** — `UNSAFE_MARKERS` in `run_eval.py` is a placeholder.
   Swap in a real moderation/safety classifier.

## Threshold tuning

All thresholds are environment variables at the top of the workflow file:

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_SCORE_THRESHOLD` | 0.85 | Minimum acceptable golden-dataset score |
| `MAX_REGRESSION_DROP` | 0.05 | Max allowed drop vs. baseline before blocking |
| `MAX_COST_PER_REQUEST_USD` | 0.05 | Cost ceiling per request |
| `MAX_P95_LATENCY_MS` | 4000 | Latency ceiling |

Start conservative and loosen once you trust the eval set's signal quality.
