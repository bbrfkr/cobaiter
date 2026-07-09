"""OFFLINE: measure task difficulty per eval prompt via the real classifier.

Reuses cobaiter's EmbeddingClassifier + the live embedding gateway (from your
.env) to print difficulty / sim_easy / sim_hard / ratio for every eval prompt,
so you can see WHERE each difficulty lands before touching the server. Difficulty
depends on the anchors + exemplars only (cost_bias/curve don't affect it).

    python -m tools.measure_difficulty [--easy-anchor X --hard-anchor Y]

Anchors default to the values in your .env / Settings; override to preview a
different mapping. Needs COBAITER_LITELLM_* pointing at a reachable embedding
gateway. Run from the repo root.
"""

from __future__ import annotations

import argparse
import asyncio

from cobaiter.classifier import EmbeddingClassifier
from cobaiter.config import get_settings
from cobaiter.litellm_client import LiteLLMClient

from ._common import as_request, load_eval_prompts, registry_specs


async def _run(easy_anchor: float | None, hard_anchor: float | None) -> None:
    settings = get_settings()
    if easy_anchor is not None:
        settings.difficulty_easy_anchor = easy_anchor
    if hard_anchor is not None:
        settings.difficulty_hard_anchor = hard_anchor
    client = LiteLLMClient.create(settings)
    specs = registry_specs()
    clf = EmbeddingClassifier(client, settings)

    print(
        f"anchors {settings.difficulty_easy_anchor}/{settings.difficulty_hard_anchor}  "
        f"embedding={settings.embedding_model}\n"
    )
    print(f"{'lvl':6} {'dom':7} {'diff':>5} {'sEasy':>6} {'sHard':>6} {'ratio':>6}  prompt")
    print("-" * 92)
    agg: dict[tuple[str, str], list[float]] = {}
    try:
        for domain, level, text in load_eval_prompts():
            res = await clf.score(as_request(text), specs)
            raw = res.raw
            se = raw.sim_easy if raw and raw.sim_easy is not None else float("nan")
            sh = raw.sim_hard if raw and raw.sim_hard is not None else float("nan")
            ratio = sh / (sh + se) if (sh + se) > 0 else float("nan")
            agg.setdefault((domain, level), []).append(res.difficulty)
            first = text.splitlines()[0]
            print(f"{level:6} {domain:7} {res.difficulty:5.2f} {se:6.3f} {sh:6.3f} "
                  f"{ratio:6.3f}  {first[:44]}")
    finally:
        await client.close()

    print("\n=== mean difficulty by (domain, level) ===")
    for k in sorted(agg):
        v = agg[k]
        print(f"  {k[0]:7} {k[1]:6}: mean={sum(v)/len(v):.3f} "
              f"[{min(v):.3f}..{max(v):.3f}]  n={len(v)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--easy-anchor", type=float, default=None)
    ap.add_argument("--hard-anchor", type=float, default=None)
    args = ap.parse_args()
    asyncio.run(_run(args.easy_anchor, args.hard_anchor))


if __name__ == "__main__":
    main()
