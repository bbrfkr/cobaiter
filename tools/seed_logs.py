"""LIVE: send the eval prompts to a running cobaiter server.

Each prompt is a fresh single-user-message conversation, so every one produces
an initial ``classifier-select`` decision that is written to the decision log —
useful for accumulating a difficulty/domain-balanced sample for
``cobaiter-calibrate``. Prints the routed model per prompt (from the
x-cobaiter-* response headers). ``max_tokens`` is kept small because only the
routing decision matters, not the generated answer.

    python -m tools.seed_logs [--base-url http://host:8080] [--max-tokens 16]

--base-url defaults to COBAITER_TOOLS_BASE_URL or http://localhost:8080.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import httpx

from ._common import cloud_models, default_base_url, load_eval_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=default_base_url())
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--model", default="auto", help="virtual model name")
    args = ap.parse_args()

    prompts = load_eval_prompts()
    cloud = cloud_models()
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for i, (domain, level, text) in enumerate(prompts, 1):
            body = {"model": args.model, "stream": False, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": text}]}
            try:
                r = client.post(f"{args.base_url}/v1/chat/completions", json=body)
                model = r.headers.get("x-cobaiter-model", "?")
                route = r.headers.get("x-cobaiter-route", "?")
                status = r.status_code
            except Exception as exc:  # noqa: BLE001
                model, route, status = f"ERR:{exc}", "-", 0
            rows.append((domain, level, model))
            tag = "  <== CLOUD" if model in cloud else ""
            print(f"[{i:2}/{len(prompts)}] {domain:7} {level:6} -> {model:28} ({route}) "
                  f"[{status}]{tag}")
            time.sleep(0.4)

    print("\n=== routing by level ===")
    for lvl in ("easy", "medium", "hard"):
        c = Counter(m for d, l, m in rows if l == lvl)
        print(f"{lvl:6}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
