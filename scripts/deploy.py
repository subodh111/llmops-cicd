#!/usr/bin/env python3
"""
Deploys a prompt/model version to an environment, optionally at a partial
traffic percentage (canary). This is a reference implementation — swap the
`_write_routing_config` internals for your actual infra (feature flag
service, config store, k8s configmap, etc).
"""

import argparse
import json
import time
from pathlib import Path

ROUTING_CONFIG_DIR = Path("deploy_state")


def _write_routing_config(env: str, version: str, traffic_pct: int):
    ROUTING_CONFIG_DIR.mkdir(exist_ok=True)
    config_path = ROUTING_CONFIG_DIR / f"{env}_routing.json"

    config = {"env": env, "deployed_at": time.time(), "versions": []}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Replace or add this version's traffic allocation
    config["versions"] = [v for v in config.get("versions", []) if v["version"] != version]
    config["versions"].append({"version": version, "traffic_pct": traffic_pct})
    config["deployed_at"] = time.time()

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[deploy] {env}: version={version} traffic={traffic_pct}%")
    print(f"[deploy] routing config written to {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["staging", "production"])
    parser.add_argument("--version", required=True, help="git sha or prompt/model version tag")
    parser.add_argument("--traffic-pct", type=int, default=100)
    args = parser.parse_args()

    _write_routing_config(args.env, args.version, args.traffic_pct)


if __name__ == "__main__":
    main()
