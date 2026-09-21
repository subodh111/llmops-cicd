#!/usr/bin/env python3
"""
Validates prompt YAML files before they enter the eval pipeline.
Checks: required keys present, template variables are well-formed,
system prompt isn't empty, no accidental hardcoded secrets/keys.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

REQUIRED_KEYS = {"system", "user_template", "version"}
SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def validate_file(path: Path) -> list[str]:
    errors = []
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"{path}: invalid YAML — {e}"]

    if not isinstance(cfg, dict):
        return [f"{path}: root must be a mapping"]

    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        errors.append(f"{path}: missing required keys {missing}")

    if "system" in cfg and not cfg["system"].strip():
        errors.append(f"{path}: system prompt is empty")

    if "user_template" in cfg:
        # Ensure every {var} placeholder is a valid identifier
        placeholders = re.findall(r"\{(\w+)\}", cfg["user_template"])
        if not placeholders:
            errors.append(f"{path}: user_template has no variable placeholders — check it's not static")

    full_text = str(cfg)
    for pattern in SECRET_PATTERNS:
        if pattern.search(full_text):
            errors.append(f"{path}: possible hardcoded secret detected — remove and use env vars")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", required=True)
    args = parser.parse_args()

    prompt_files = list(Path(args.prompts_dir).glob("*.yaml"))
    if not prompt_files:
        print(f"No prompt files found in {args.prompts_dir}")
        sys.exit(1)

    all_errors = []
    for path in prompt_files:
        all_errors.extend(validate_file(path))

    if all_errors:
        print("❌ Prompt validation failed:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"✅ Validated {len(prompt_files)} prompt file(s)")


if __name__ == "__main__":
    main()
