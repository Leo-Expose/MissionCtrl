"""Operational helpers for Hugging Face Space validation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx


def load_runtime_metadata() -> Dict[str, str]:
    """Normalize runtime metadata from local/HF environments."""
    return {
        "service": "missionctrl",
        "space_id": os.getenv("SPACE_ID", "unknown"),
        "container_id": os.getenv("HOSTNAME", "unknown"),
        "build_id": os.getenv("BUILD_ID", os.getenv("SPACE_ID", "unknown")),
        "git_sha": os.getenv("GIT_SHA", os.getenv("HF_SPACE_COMMIT_SHA", "unknown")),
    }


def smoke_check(base_url: str | None = None, timeout_s: float = 10.0) -> Dict[str, Any]:
    """Validate that health-critical endpoints respond with HTTP 200."""
    root = (base_url or os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")).rstrip("/")
    endpoints = ["/", "/health", "/state", "/logs"]
    results: Dict[str, Any] = {"base_url": root, "ok": True, "checks": []}

    with httpx.Client(timeout=timeout_s) as client:
        for endpoint in endpoints:
            url = f"{root}{endpoint}"
            response = client.get(url)
            is_ok = response.status_code == 200
            results["checks"].append({
                "endpoint": endpoint,
                "status_code": response.status_code,
                "ok": is_ok,
            })
            if not is_ok:
                results["ok"] = False
    return results


if __name__ == "__main__":
    report = {
        "runtime": load_runtime_metadata(),
        "smoke_check": smoke_check(),
    }
    print(json.dumps(report, indent=2))
