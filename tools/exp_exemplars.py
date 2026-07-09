"""OFFLINE: compare a CANDIDATE difficulty-exemplar set against the current one.

Measures the difficulty ratio (sim_hard / (sim_hard + sim_easy)) for every eval
prompt under two exemplar sets and reports the mean ratio per (domain, level)
plus the hard-vs-medium separation — the signal that decides whether hard tasks
pull away from medium. Use it to tune ``difficulty_exemplars.yaml`` before
deploying: a good change RAISES hard-medium separation without lifting medium.

    python -m tools.exp_exemplars [--candidate path/to/exemplars.yaml]

--candidate defaults to the repo's difficulty_exemplars.yaml. "Baseline" is
whatever is currently baked into cobaiter/classifier.py. Run from the repo root
with COBAITER_LITELLM_* pointing at a reachable embedding gateway.
"""

from __future__ import annotations

import argparse
import asyncio

import cobaiter.classifier as clf_mod
from cobaiter.classifier import EmbeddingClassifier
from cobaiter.config import get_settings
from cobaiter.exemplars import load_difficulty_exemplars
from cobaiter.litellm_client import LiteLLMClient

from ._common import as_request, load_eval_prompts, registry_specs


async def _measure(client, specs, settings, easy, hard, label) -> None:
    clf = EmbeddingClassifier(client, settings, easy_exemplars=easy, hard_exemplars=hard)
    agg: dict[tuple[str, str], list[float]] = {}
    for domain, level, text in load_eval_prompts():
        res = await clf.score(as_request(text), specs)
        raw = res.raw
        se = raw.sim_easy if raw and raw.sim_easy is not None else 0.0
        sh = raw.sim_hard if raw and raw.sim_hard is not None else 0.0
        ratio = sh / (sh + se) if (sh + se) > 0 else 0.0
        agg.setdefault((domain, level), []).append(ratio)

    print(f"\n########## {label} (easy={len(easy)}, hard={len(hard)}) ##########")
    for k in sorted(agg):
        v = agg[k]
        print(f"  {k[0]:7} {k[1]:6}: mean_ratio={sum(v)/len(v):.3f} "
              f"[{min(v):.3f}..{max(v):.3f}]  n={len(v)}")

    def mean(dom, lvl):
        v = [r for (d, l), rs in agg.items() if d == dom and l == lvl for r in rs]
        return sum(v) / len(v) if v else float("nan")

    print(f"  general hard-medium separation: {mean('general','hard')-mean('general','medium'):+.3f}")
    print(f"  coding  hard-medium separation: {mean('coding','hard')-mean('coding','medium'):+.3f}")


async def _run(candidate: str | None) -> None:
    settings = get_settings()
    client = LiteLLMClient.create(settings)
    specs = registry_specs()
    base_easy = list(clf_mod._EASY_EXEMPLARS)
    base_hard = list(clf_mod._HARD_EXEMPLARS)
    cand_easy, cand_hard = load_difficulty_exemplars(candidate) if candidate \
        else (base_easy, base_hard)
    try:
        await _measure(client, specs, settings, base_easy, base_hard,
                       "BASELINE (in-code defaults)")
        await _measure(client, specs, settings, cand_easy, cand_hard,
                       f"CANDIDATE ({candidate or 'same as baseline'})")
    finally:
        await client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="difficulty_exemplars.yaml",
                    help="candidate exemplars YAML (easy:/hard:); default: repo file")
    args = ap.parse_args()
    asyncio.run(_run(args.candidate))


if __name__ == "__main__":
    main()
