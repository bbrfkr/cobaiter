"""OFFLINE: predict the server's model selection per eval prompt.

Runs the REAL RouteEngine scoring (``_rank`` -> ``_select_best``) with real
embeddings under a chosen (anchors, capability_curve, cost_bias) config, so you
can sweep those knobs and see which prompts escalate to a cloud model WITHOUT
recreating the container each time. Matches live routing to within ~1-2 prompts
(embedding jitter); use it for DIRECTION, then confirm with tools.validate_routing.

    python -m tools.sim_router \
        --easy-anchor 0.412 --hard-anchor 0.702 --curve 1.2 --cost-bias 0.3,0.2,0.1

Each --cost-bias value is simulated in turn (embeddings are cached, so the sweep
is cheap). Anchors/curve default to your .env / Settings. Run from the repo root
with COBAITER_LITELLM_* pointing at a reachable embedding gateway.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from cobaiter.classifier import EmbeddingClassifier
from cobaiter.config import get_settings
from cobaiter.litellm_client import LiteLLMClient
from cobaiter.router import RouteEngine

from ._common import as_request, cloud_models, load_eval_prompts, registry_specs


async def _run(easy: float | None, hard: float | None, curve: float | None,
               cost_biases: list[float]) -> None:
    settings = get_settings()
    if easy is not None:
        settings.difficulty_easy_anchor = easy
    if hard is not None:
        settings.difficulty_hard_anchor = hard
    if curve is not None:
        settings.capability_curve = curve
    client = LiteLLMClient.create(settings)
    specs = registry_specs()
    cloud = cloud_models()
    clf = EmbeddingClassifier(client, settings)  # uses in-code default exemplars
    engine = RouteEngine(None, client, clf, settings)  # store unused by _rank/_select_best
    prompts = load_eval_prompts()

    try:
        for cb in cost_biases:
            settings.cost_bias = cb
            selected = []
            for domain, level, text in prompts:
                res = await engine._rank(as_request(text), specs)
                best = engine._select_best(res, specs)
                selected.append((domain, level, best.model))
            _report(settings, cb, selected, cloud)
    finally:
        await client.close()


def _report(settings, cb, selected, cloud) -> None:
    hard = [(d, m) for d, l, m in selected if l == "hard"]
    n_cloud = sum(1 for _, m in hard if m in cloud)
    gh = sum(1 for d, l, m in selected if d == "general" and l == "hard" and m in cloud)
    ch = sum(1 for d, l, m in selected if d == "coding" and l == "hard" and m in cloud)
    false_esc = sum(1 for d, l, m in selected if l in ("easy", "medium") and m in cloud)
    print(f"\n===== anchors {settings.difficulty_easy_anchor}/"
          f"{settings.difficulty_hard_anchor}  curve={settings.capability_curve}  "
          f"cost_bias={cb} =====")
    print(f"  hard->cloud: {n_cloud}/{len(hard)}  (general {gh}/7, coding {ch}/6)  "
          f"easy+medium wrongly escalated: {false_esc}")
    for lvl in ("easy", "medium", "hard"):
        c = Counter(m for _, l, m in selected if l == lvl)
        print(f"  {lvl:6}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--easy-anchor", type=float, default=None)
    ap.add_argument("--hard-anchor", type=float, default=None)
    ap.add_argument("--curve", type=float, default=None)
    ap.add_argument("--cost-bias", type=_floats, default=None,
                    help="comma-separated values to sweep, e.g. 0.3,0.2,0.1")
    args = ap.parse_args()
    cbs = args.cost_bias if args.cost_bias else [get_settings().cost_bias]
    asyncio.run(_run(args.easy_anchor, args.hard_anchor, args.curve, cbs))


if __name__ == "__main__":
    main()
