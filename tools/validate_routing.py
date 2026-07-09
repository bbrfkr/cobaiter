"""LIVE: validate which model a running cobaiter server routes each eval prompt to.

Conversation state is sticky, so re-sending an identical prompt would return
PINNED to the previously-chosen model. This tool first DELETEs each prompt's
conversation state (via the admin API, using the server's own fingerprint), then
POSTs it fresh — forcing a re-classification under the CURRENT server config —
and reads the routed model from the x-cobaiter-* headers. Use it after a config
change (``docker compose up -d --force-recreate cobaiter``) to confirm the
predicted routing from tools.sim_router.

    python -m tools.validate_routing [--base-url http://host:8080]

--base-url defaults to COBAITER_TOOLS_BASE_URL or http://localhost:8080.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import httpx

from cobaiter.features import conversation_key

from ._common import as_request, cloud_models, default_base_url, load_eval_prompts


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
            # Use the server's own key derivation so the delete targets the right state.
            key = conversation_key(as_request(text), None)
            try:
                client.delete(f"{args.base_url}/admin/conversations/{key}")
            except Exception:  # noqa: BLE001
                pass
            body = {"model": args.model, "stream": False, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": text}]}
            try:
                r = client.post(f"{args.base_url}/v1/chat/completions", json=body)
                model = r.headers.get("x-cobaiter-model", "?")
                route = r.headers.get("x-cobaiter-route", "?")
            except Exception as exc:  # noqa: BLE001
                model, route = f"ERR:{exc}", "-"
            rows.append((domain, level, model))
            tag = "  <== CLOUD" if model in cloud else ""
            print(f"[{i:2}/{len(prompts)}] {domain:7} {level:6} -> {model:28} ({route}){tag}")
            time.sleep(0.4)

    print("\n=== routing by level ===")
    for lvl in ("easy", "medium", "hard"):
        c = Counter(m for d, l, m in rows if l == lvl)
        print(f"{lvl:6}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    n_hard = sum(1 for _, l, _ in rows if l == "hard")
    n_hard_cloud = sum(1 for _, l, m in rows if l == "hard" and m in cloud)
    n_false = sum(1 for _, l, m in rows if l in ("easy", "medium") and m in cloud)
    print(f"\nhard escalated to cloud: {n_hard_cloud}/{n_hard}")
    print(f"easy+medium wrongly escalated: {n_false}")


if __name__ == "__main__":
    main()
