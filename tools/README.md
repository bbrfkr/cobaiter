# Routing-tuning tools

Dev-only helpers for tuning cobaiter's difficulty/escalation routing — deciding
which tasks stay on the light local model, use the local "think" model, or
escalate to a pricier cloud model. They are **not** part of the built package
(the wheel only ships `cobaiter/`).

All tools share one evaluation set, [`eval_prompts.yaml`](./eval_prompts.yaml):
a *domain* (general / coding) × *difficulty* (easy / medium / hard) grid with
varied phrasing (deliberately **not** copies of the classifier's difficulty
exemplars) so it measures real generalization, not exemplar memorization. Extend
it freely — keep the `domain`/`level` labels so the summaries stay meaningful.

Run everything as a module **from the repo root**:

```bash
python -m tools.<name> [options]
```

## The two kinds of tool

### Offline (predict, no server round-trip)

Reuse cobaiter's real classifier/router + the live **embedding gateway** (from
your `.env`, so `COBAITER_LITELLM_*` must reach a real gateway). Embeddings are
cached within a run, so sweeps are cheap.

| Tool | What it answers |
| --- | --- |
| `tools.measure_difficulty` | Where does each prompt's **difficulty / ratio** land? Prints difficulty, `sim_easy`, `sim_hard`, ratio per prompt + means. Difficulty depends only on anchors + exemplars. |
| `tools.exp_exemplars` | Does a **candidate `difficulty_exemplars.yaml`** separate hard from medium better than the current one? Reports mean ratio per (domain, level) + hard−medium separation for baseline vs candidate. |
| `tools.sim_router` | Under a given **(anchors, curve, cost_bias)**, which prompts **escalate to cloud**? Runs the real `RouteEngine._rank` → `_select_best`. Sweep `--cost-bias 0.3,0.2,0.1`. |

Matches live routing to within ~1–2 prompts (embedding jitter) — use it for
**direction**, then confirm on the server.

### Live (validate against a running server)

Talk to a cobaiter server over HTTP. Target it with `--base-url` or
`COBAITER_TOOLS_BASE_URL` (default `http://localhost:8080`). "Cloud" models are
read from `models.yaml` (`is_local: false`), never hardcoded.

| Tool | What it does |
| --- | --- |
| `tools.seed_logs` | Sends each eval prompt as a fresh conversation so every one is logged as an initial `classifier-select` decision — accumulates a balanced sample for `cobaiter-calibrate`. |
| `tools.validate_routing` | Deletes each prompt's conversation state (so it re-classifies), re-sends it, and reports the routed model per level + `hard→cloud` / false-escalation counts. Run after a config change. |

## The tuning workflow

```
                 ┌─────────────────────────────────────────────┐
   tune a knob   │ 1. predict offline                          │
   (anchors,     │    python -m tools.sim_router \             │
    curve,       │        --easy-anchor .. --hard-anchor .. \  │
    cost_bias,   │        --curve .. --cost-bias 0.3,0.2,0.1   │
    exemplars)   │    (exemplars: tools.exp_exemplars first)   │
                 └───────────────────┬─────────────────────────┘
                                     │ looks good?
                 ┌───────────────────▼─────────────────────────┐
                 │ 2. apply on the server                      │
                 │    edit .env / difficulty_exemplars.yaml,   │
                 │    then RECREATE the container:             │
                 │    docker compose up -d --force-recreate \  │
                 │        cobaiter                             │
                 │    (a plain `restart` does NOT reload the   │
                 │     env_file — a classic footgun)           │
                 └───────────────────┬─────────────────────────┘
                 ┌───────────────────▼─────────────────────────┐
                 │ 3. validate live                            │
                 │    python -m tools.validate_routing \       │
                 │        --base-url http://SERVER:8080        │
                 └─────────────────────────────────────────────┘
```

### Which knob does what

- **Difficulty anchors** (`COBAITER_DIFFICULTY_EASY_ANCHOR` / `_HARD_ANCHOR`) —
  linearly map the raw `sim_hard / (sim_hard + sim_easy)` ratio to difficulty.
  Re-fit from real traffic with `cobaiter-calibrate` (feed it a balanced sample
  via `tools.seed_logs`). A linear rescale can't fix a weak signal — check the
  ratio separation with `tools.exp_exemplars` if anchors alone won't separate.
- **`COBAITER_DIFFICULTY_EXEMPLARS_CONFIG`** → `difficulty_exemplars.yaml` — the
  easy/hard exemplar sets. Adding hard exemplars for under-covered expert domains
  raises those tasks' `sim_hard`. Difficulty uses MAX cosine, so keep hard
  exemplars clearly "expert" or they lift medium too. Tune with `exp_exemplars`.
- **`COBAITER_CAPABILITY_CURVE`** — moves BOTH the no-think→think boundary and
  the think→cloud boundary. Lower (→1.0) escalates more hard tasks but leaks
  trivial tasks onto the slower "think" model; higher (→1.2) keeps trivial tasks
  on no-think but escalates fewer hard tasks.
- **`COBAITER_COST_BIAS`** — the independent lever that shifts think→cloud
  without touching the easy boundary. Weak at higher `capability_curve` (a
  0.3→0.2 change can be a no-op); check its real effect with `sim_router`.

## Prerequisites

- Run from the repo root with the project venv (`cobaiter` importable).
- **Offline** tools need `.env` with `COBAITER_LITELLM_BASE_URL` /
  `COBAITER_LITELLM_API_KEY` / `COBAITER_EMBEDDING_MODEL` reaching a real
  embedding gateway.
- **Live** tools need a reachable cobaiter server (`--base-url`).
