# cobaiter — Context Based AI Router

`cobaiter` は、AI エージェントと LLM ゲートウェイ（**LiteLLM** 前提）の間に立つ
**OpenAI 互換プロキシ**です。呼び出し側はどのモデルが使われるかを意識せず、仮想モデル名
`cobaiter-auto` に投げるだけ。cobaiter が会話のコンテキストを見て最適なモデルを自動選択し、
**1 つの会話では基本同じモデルに固定**して回答の一貫性を保ちます（会話が違えば違う AI と話す体験）。

## 特徴

- **モデル非依存の自動選択**: エージェントは `cobaiter-auto` を呼ぶだけ。
- **会話スティッキー**: 会話ごとにモデルを固定（`route: pinned`）。一貫性を担保。
- **ハイブリッド判定**: ハード制約フィルタ（モデルレジストリ）で候補を絞り、複数残るときだけ
  **embedding 類似度（relevance）＋ヒューリスティック（difficulty）**で 1 つを選択。
  生成系 LLM を同期パスで呼ばないため、ルーティングの固定レイテンシーは数十 ms 程度。
- **ヒステリシス付き再ルーティング**: 文脈が実質的に変化したら会話途中でも切替（`route: context-switch`）。
  `min_dwell_turns` と `switch_margin` で過剰なフラッピングを抑制。
- **フェイルオーバー**: コンテキスト長超過 / レート制限 / クォータ枯渇 / 障害を検出し、
  フォールバックチェーンの次段へ切替（`route: failover`）。切替は単方向（元へ戻さない）。
- **永続化**: 会話状態・モデルレジストリを **Valkey**（AOF）に保存。

## アーキテクチャ

```
AIエージェント ──(cobaiter-auto, OpenAI互換)──▶ cobaiter ──(具体モデル名)──▶ LiteLLM ──▶ Claude / GLM / Ollama ...
                                                  │                               ▲
                ┌─────────────────────────────────┼──────────────┐               │ budget/spend 参照
   会話状態(Valkey)            モデルレジストリ(Valkey)        embeddingモデル      │ (クレジット余力)
   conv:<key>→{model,...}     capabilities/窓/cost/tier/     (LiteLLM経由)─────────┘
                              fallback_chain                 ※複数候補時だけ
```

### ルーティング判定（ユーザーターン単位の統合フロー）

1. 会話キーを決定（明示 ID → `metadata.conversation_id` → `user` → 会話先頭の指紋）。
2. **ユーザーターン境界かどうかを判定**: リクエスト中の `user` ロールメッセージ数が前回判定時より
   増えていれば「新しいユーザーターン」、そうでなければ「1 指示の途中（エージェントのツール往復・再送）」。
   - **指示の途中**: 原則 `pinned` で固定（ソフト再評価・文脈スイッチは行わない）。固定モデルが
     制約違反/不可用/クレジット枯渇になった場合のみ強制 `failover`（留まる選択肢がないため）。
   - これにより、**エージェントが 1 指示で何十回も API を叩いても、その指示の中ではモデルが切り替わらない**。
3. ハード制約を再計算（画像/ツール/プライバシー/トークン数 vs 窓）＋各モデルの可用性・クレジット余力。
4. **継続会話（新しいユーザーターン）**: 固定モデルが有効なら `pinned`。制約違反/不可用/クレジット枯渇なら即 `failover`。
   実質的な文脈変化があれば（2 段ゲート: 安価トリガ→再分類→マージン判定）`context-switch`。
5. **新規会話**: 制約フィルタ後、候補 1 つなら `rule`、複数なら relevance/difficulty スコアリングで
   `classifier-select`、0 なら `default`。

> **ターンの定義**: ヒステリシス（`min_dwell_turns` / `soft_recheck_every`）が数える「ターン」は
> **ユーザーの発話 1 回**であり、ダウンストリームの API コール回数ではありません。エージェンティック
> ループのツール往復はターンを消費しないため、ドウェル／再評価の窓が指示途中で空回りしません。

詳細は実装計画（`~/.claude/plans/…`）を参照。

## クイックスタート（Docker Compose）

```bash
cp .env.example .env
# .env と litellm_config.yaml を編集（ANTHROPIC_API_KEY などを設定）
export ANTHROPIC_API_KEY=sk-...
docker compose up -d            # valkey + litellm + cobaiter
curl localhost:8000/healthz
```

### ローカル開発（Valkey のみコンテナ）

```bash
uv sync
docker compose up -d valkey
COBAITER_VALKEY_URL=redis://localhost:6379/0 uv run python -m cobaiter
```

## 使い方

### モデルレジストリ（設定ファイルで外部注入）

どのモデルをどの `cost`／`tier`／能力／フォールバックに分類するかは、**外部の設定ファイルで手動管理**します
（ハードコードしません）。パスは `COBAITER_MODELS_CONFIG`（既定では未指定＝内蔵デフォルト）で指定し、
compose では `models.yaml` をコンテナにマウントします。**このファイルが真実の源**で、起動時にレジストリは
ファイルの内容ちょうどに同期されます（ファイルに無いモデルは削除）。

```yaml
# models.yaml
models:
  - model: bbrfkr-llm-general          # LiteLLM が公開する実モデル名
    description: 汎用の対話・推論・文章作成向け  # このモデルの「用途」のみ（task_examples 未設定時のフォールバック）
    task_examples:                     # 代表的なタスク文（設定時はこちらが relevance のスコアリング対象）
      - この文章を要約してほしい
      - 旅行のプランについて相談したい
    cost: 0                            # 相対コスト（USD/Mtok 目安、ローカル=0）
    tier: 2                            # 能力レベル（整数・大きいほど高性能/低速）
    context_window: 262144
    multimodal: true
    supports_tools: true
    is_local: true
    fallback_chain: [bbrfkr-llm-general-no-think]
  # ... 以降、運用するモデルを列挙
```

**関心の分離**がこのスキーマの肝です。`description`/`task_examples` は各モデルの**用途のみ**を表す自由文で、
ルーティング時に 2 軸のスコアを**LLM 生成なし**で算出します:

- **`relevance`（0..1）** … 会話ダイジェストの embedding と各候補の参照テキストの embedding の
  **top-2-mean cosine 類似度**。参照テキストは `task_examples` が設定されていればその全件、未設定なら
  `description` を1件のリストとして扱う（`task_examples` が空の場合は旧来の単一ベクトル比較と完全に
  同じ挙動）。複数 example の上位2件の平均を取ることで、1本の example の言い回しのブレ（埋め込みモデルは
  短い日本語文の些細な違いに敏感 — 句点の有無だけで類似度が有意にずれることを実測済み）を緩和しつつ、
  「example 数が多いドメインほど有利になる」比較上のバイアスも抑える。最良候補を 1.0 に固定し、類似度が
  `COBAITER_EMBEDDING_REL_BAND`（既定 0.10）下回るごとに 0.0 へ落とすコントラスト正規化を行う（生 cosine は
  圧縮帯域に張り付くため）。参照テキストのベクトルはプロセス内キャッシュされ、定常状態では**リクエストあたり
  embedding 1 コール（ダイジェストのみ）**で済む。`task_examples` を新規導入・大幅追加すると raw cosine の
  分布が変わりうるため、band の再較正（後述）も併せて回すことを推奨する。
- **`difficulty`（0..1）** … タイトル生成・要約・翻訳・抽出などの**メタタスク語（LOW_INTENT キーワード）が
  指示部（system 冒頭＋最新 user メッセージの先頭/末尾）にあれば、埋め込まれた本文がどれだけ難しくても
  低難度（0.15）に決定的に固定**（最優先でチェック）。それ以外は embedding ベース: 会話ダイジェストが
  「易しいタスク」「難しいタスク」の2つの固定 exemplar 集合（数学・コーディング・科学・法律など複数ドメイン
  にまたがる）とどれだけ似ているか（`sim_hard / (sim_hard + sim_easy)`）を計算し、
  `COBAITER_DIFFICULTY_EASY_ANCHOR`/`_HARD_ANCHOR` で 0.15〜0.85 にrescale。エラー・スタックトレース痕跡で
  `+0.05`、code fence で `+0.03` の小さな加点が乗る。トークン数だけを見る粗いフォールバック
  （`_fallback_difficulty`）は、embedding 呼び出し自体が失敗した場合の最終手段としてのみ使われる。

続く連続値の算術は**ルーターのコード**が決定的に合成します。まず能力適合を作り、続いて cost/tier を再ランキング:

```
capability_fit = 1 − max(0, difficulty − tier/maxTier)   # 力不足のときだけ減点、過剰能力は満点
suitability    = relevance × capability_fit
effective      = suitability − (cost_bias×(cost/maxCost) + tier_bias×(tier/maxTier)) × (1 − difficulty)
```

これにより「**難しいタスクは高 tier を、簡単なタスクは無料/ローカル（cost=0）をしっかり使い**、有料モデルは明確に
優位なときだけ選ぶ」挙動になります。校正済みフロートをコード側で作るので、スコアが 0/1 に潰れず 0..1 に分布します。

cost/tier ペナルティは `(1 − difficulty)` でスケールします。**簡単なタスクほどコスト/tier を満額で効かせ**（過剰投資を
避け、最安・最軽量の十分なモデルを選ぶ）、**難しいタスクほどペナルティを緩め**（能力に対価を払う価値があるので、明確に
高能力な有料モデルが勝てる）ようにします。つまり difficulty が「コストと能力のトレードオフ」を一括で握るノブです。
**各モデル自身の suitability ではなく difficulty でスケールする**のが要点で、`(1 − suitability)` でスケールすると分類器が
「ドメイン的にドンピシャ（suitability=1.0）」と判定した瞬間にコストペナルティが 0 になり、同じく十分な無料ローカル
モデルがあっても有料クラウドが必ず勝ってしまう（コスト選好が無効化される）ためです。difficulty が無い場合は
ペナルティを満額適用し、最安・最軽量へ素直に倒します（embedding が失敗しても relevance が全候補一律 0.5 に
なるだけで、difficulty ヒューリスティックは常に算出されます）。

`capability_fit` の `maxTier` は**実際に競合している候補**（relevance がトップの `COBAITER_CAPABILITY_REL_FRACTION`
＝既定 0.5 以上）だけで取ります。relevance ~0 の畑違いモデル（例: 非コーディングタスクにおける高 tier の coding
モデル）が `maxTier` を吊り上げてドメイン内モデルの fit を不当に下げ、no-think→think の境界を下げすぎる（タイトル生成の
ような自明なタスクまで think に上がる）のを防ぐためです。

**`tier` の役割分担**（重要）: 「**難しいタスクで高 tier を選ぶ**」は `capability_fit`（力不足だけ減点）が担います。
一方 `effective` の `tier` 項は**ペナルティ**で、「**足りているなら、より軽い（低 tier）モデルを選ぶ**」＝簡単なタスクで
重いモデルを過剰に使わない（over-provision を避ける）ための項です。つまり `cost` も `tier` も**ペナルティ**で対称的に
「安く・軽く、ただし十分なものを」を表現します（同点時のタイブレークも local → 安い → 低 tier の順）。重み
`cost_bias`／`tier_bias` は `COBAITER_COST_BIAS`／`COBAITER_TIER_BIAS` で調整できます（`cost_bias > tier_bias` が原則）。

実行中の一時的な上書きは `PUT /admin/models` でも可能ですが、再起動すると設定ファイルに再同期されます。
能力値（窓・マルチモーダル・ツール対応）はバックエンドに合わせて手動調整してください
（LiteLLM 側に未設定なことが多いため）。embedding モデル（`COBAITER_EMBEDDING_MODEL`）はルーティング対象外
なのでレジストリには含めません（LiteLLM 側の `model_list` には `/v1/embeddings` 対応エントリとして登録が必要）。

### difficulty/relevance の再キャリブレーション（ログ + LLM judge）

`difficulty_easy_anchor` / `difficulty_hard_anchor` / `embedding_rel_band` は、当初は手作業の少数
キャリブレーションセットで一度だけ決めた定数です。実運用のトラフィックで妥当性を検証・再調整できるよう、
分類器が実際に判断した回（`route: classifier-select` / `context-switch`）の生シグナルを Valkey の
Stream（`cobaiter:decisions`）に記録し、オフラインの `cobaiter-calibrate` コマンドで再キャリブレーションを
支援します。

- **何を記録するか**: 会話キー・ターン・route・選定モデル・difficulty、各候補への raw cosine 類似度
  （`candidate_sims`）、各候補が実際に比較に使った**解決済み参照テキスト**（`candidate_refs` — `task_examples`
  またはそのフォールバックの `description`。レジストリのメタデータであり会話本文ではないため、privacy会話でも
  redact されない）、difficulty exemplar への類似度（`sim_easy`/`sim_hard`）。プライバシー会話（`needs_local`）
  ではタスク本文（`task_text`）のみ必ず `None` に落とします。`COBAITER_DECISION_LOG_ENABLED=false` で無効化、
  `COBAITER_DECISION_LOG_MAXLEN` で Stream の上限（近似トリム）を指定できます。記録は best-effort —
  Valkey 書き込みに失敗してもルーティング自体は失敗しません。`task_examples` を広く使うようになった場合、
  1エントリのサイズが伸びるため `COBAITER_DECISION_LOG_MAXLEN` の引き下げも検討してください。
- **再キャリブレーション**: `COBAITER_CALIBRATION_JUDGE_MODEL` に judge 用モデルを設定した上で

  ```bash
  uv run cobaiter-calibrate
  # あるいは: uv run python -m cobaiter.calibrate
  ```

  を実行すると、ログから最大 `COBAITER_CALIBRATION_SAMPLE_SIZE` 件をサンプリングし、以下の2軸を再較正します:

  - **difficulty**: 各タスク本文を judge モデルに投げて 0..1 の難易度ラベルを取得し、既存の
    `sim_easy`/`sim_hard` 比率に対して最小二乗で線形回帰、`difficulty_easy_anchor`/`hard_anchor` の推奨値と
    RMSE（現行 vs 推奨）をレポートします。
  - **relevance band**: `candidate_refs` が完全一致する候補を1つの「ドメイン」としてグルーピングし
    （think/no-think のような同一ドメイン内の tier 違いを judge に区別させないため）、judge にタスク本文と
    各ドメインの参照テキストを見せてどのドメインが正しいか判定させます。ただし `embedding_rel_band` は
    `relevance_from_sims` の性質上、**raw cosine が最大の候補を常に relevance 1.0 にする**ため、band の値を
    変えても「どのドメインが勝つか」は変わりません（band で直せるのは relevance の誤判定ではなく、
    description/task_examples の書き方の問題です）。band が実際にコントロールするのは「正解ドメインが
    既に raw top を取れているときに、他のドメインをどれだけ確実に抑え込めるか」なので、再較正は
    「judge が正解だと言ったドメインが raw top と一致した」サンプルだけを対象に、**全サンプルで
    誤ドメインの suitability 相当値が 0.5（分類器のニュートラル値）以下に収まる、最大の band 値**を探索します
    （difficulty のような RMSE 回帰ではなく、band は raw top 自体を変えないため grid search + 安全率が
    適切な指標です）。レポートには band 非依存の `raw top-pick accuracy`（relevance がそもそも正しいドメインを
    最上位に選べている割合）も出るので、これが低い場合は band ではなく `description`/`task_examples` の
    見直しが必要というシグナルになります。あわせて、候補間の raw cosine 類似度差が
    `COBAITER_EMBEDDING_REL_BAND` 未満だった「際どい」ルーティング判断の一覧に、judge の判定
    （どのドメインが正解でルーターと一致したか）を付記して表示します。
- **意図的に自動適用しない**: judge 呼び出しは同期 LLM 生成そのもの（分類器が embedding 方式に置き換えた
  レイテンシコスト）なので、このツールはリクエストのホットパスには一切乗らず、常にオフラインでバッチ実行
  します。またルーター自身の誤判定がそのまま「正解」として再学習に混入するフィードバックループを避けるため、
  `.env` の値は**人間がレポートを見て手動で書き換える**運用とし、自動書き込みは行いません。

### チャット（OpenAI 互換）

```bash
curl -s -D- localhost:8000/v1/chat/completions \
  -H 'x-cobaiter-conversation-id: conv-123' \
  -d '{"model":"cobaiter-auto","messages":[{"role":"user","content":"こんにちは"}]}'
# レスポンスヘッダ:
#   x-cobaiter-model: 実際に使われたモデル
#   x-cobaiter-route: pinned|rule|classifier-select|context-switch|failover|default
#   x-cobaiter-conversation: 会話キー
```

会話の固定状態は `GET /admin/conversations/<key>` で確認、`DELETE` でリセットできます。

## エンドポイント

| Method | Path | 説明 |
| --- | --- | --- |
| POST | `/v1/chat/completions` | メイン。ルーティングして LiteLLM へ中継 |
| GET | `/v1/models` | 仮想モデル＋レジストリのモデル一覧 |
| GET | `/healthz` | ヘルスチェック（Valkey 接続含む） |
| GET/PUT | `/admin/models` | レジストリ一覧 / 追加・更新 |
| DELETE | `/admin/models/{model}` | レジストリ削除 |
| GET/DELETE | `/admin/conversations/{key}` | 会話束縛の確認 / リセット |

## 設定（環境変数）

すべての設定は環境変数（接頭辞 `COBAITER_`）または `.env` で与えます。雛形は `.env.example` を参照。
以下が全項目です（既定値は `cobaiter/config.py`）。

### ダウンストリーム LiteLLM ゲートウェイ

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_LITELLM_BASE_URL` | `http://localhost:4000` | 中継先 LiteLLM のベース URL |
| `COBAITER_LITELLM_API_KEY` | （空） | LiteLLM 呼び出し用 API キー（master key 等） |

### Valkey（会話状態・モデルレジストリの永続化）

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_VALKEY_URL` | `redis://localhost:6379/0` | Valkey/Redis 接続 URL。到達不能時は状態系 API が 503 |

### モデルレジストリ

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_MODELS_CONFIG` | （空） | 外部レジストリファイル（YAML/JSON）のパス。**空＝内蔵デフォルトシード**。指定時は起動毎にこのファイルへ完全同期（未記載モデルは削除） |

### ルーティング

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_VIRTUAL_MODEL` | `cobaiter-auto` | エージェントが呼ぶ仮想モデル名。これ宛のリクエストをルーティング対象とする |
| `COBAITER_EMBEDDING_MODEL` | `text-embedding-3-small` | relevance スコアリングに使う embedding モデル（LiteLLM の `/v1/embeddings` 経由）。**レジストリには含めない**（ルーティング対象外） |
| `COBAITER_EMBEDDING_REL_BAND` | `0.10` | relevance のコントラスト帯域。最良候補との cosine 類似度差がこの値に達すると relevance 0.0。小さいほどドメイン分離が鋭くなる |
| `COBAITER_CLASSIFIER_DIGEST_CHARS` | `400` | embedding するタスクダイジェスト（会話の先頭＋末尾）の最大文字数 |
| `COBAITER_DEFAULT_MODEL` | `claude-haiku-4-5` | どの候補も制約を満たさないときの安全なフォールバック先 |

### 会話スティッキー / ヒステリシス

「ターン」は**ユーザーの発話 1 回**で数えます（API コール回数ではない。前述の「ターンの定義」参照）。

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_CONV_TTL_SECONDS` | `604800`（7 日） | 会話束縛（固定モデル等）を Valkey に保持する TTL |
| `COBAITER_MIN_DWELL_TURNS` | `3` | ソフト切替（品質駆動の `context-switch`）を許すまで現モデルに留まる最小ユーザーターン数。フラッピング抑制 |
| `COBAITER_SWITCH_MARGIN` | `0.15` | 切替に必要な「最良候補スコア − 固定モデルスコア」の優位差。小さいほど切り替わりやすい |
| `COBAITER_SOFT_RECHECK_EVERY` | `4` | 変化トリガが無くても N ユーザーターン毎に分類器で定期再評価する周期 |
| `COBAITER_SCORE_EMA_ALPHA` | `0.5` | 固定モデルスコアの EMA 平滑係数（0..1、大きいほど直近に反応） |

### ログ / 再キャリブレーション

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_DECISION_LOG_ENABLED` | `true` | 分類器が実行された判断を `cobaiter:decisions`（Valkey Stream）へ記録するか |
| `COBAITER_DECISION_LOG_MAXLEN` | `20000` | 上記 Stream の上限（`XADD MAXLEN ~`、近似トリム） |
| `COBAITER_CALIBRATION_JUDGE_MODEL` | （空） | `cobaiter-calibrate` が difficulty のゴールドラベル生成に使う judge モデル。未設定だと実行時エラー |
| `COBAITER_CALIBRATION_SAMPLE_SIZE` | `200` | 1 回の再キャリブレーションでログから抽出しジャッジに投げる件数の上限 |

### クレジット / 可用性（LiteLLM の budget/spend を参照）

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_CREDIT_FLOOR` | `0.0` | 残クレジット余力（USD）がこの値を下回るモデルは不可用として候補から除外 |
| `COBAITER_CREDIT_CACHE_TTL` | `30` | LiteLLM の budget/spend 参照結果のキャッシュ TTL（秒） |

### HTTP サーバ / クライアント

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `COBAITER_HOST` | `0.0.0.0` | 待ち受けホスト |
| `COBAITER_PORT` | `8000` | 待ち受けポート |
| `COBAITER_REQUEST_TIMEOUT` | `600.0` | ダウンストリーム HTTP リクエストのタイムアウト（秒） |

## テスト

```bash
uv run pytest
```

`tests/test_classifier.py` 等は決定的なフェイク embedding 空間を使うため、実際の embedding モデルとは
無関係に高速・オフラインで実行できます。実際の `models.yaml`（本番レジストリ）と実 embedding モデルに対する
回帰テスト（golden set、`tests/fixtures/routing_cases.yaml`）は別途用意されており、デフォルトでは skip されます:

```bash
docker compose up -d valkey litellm   # または COBAITER_LITELLM_BASE_URL を既存環境に向ける
COBAITER_RUN_GOLDEN=1 uv run pytest -m golden -v
```

`models.yaml` の description/task_examples・embedding モデル・`COBAITER_EMBEDDING_REL_BAND` や difficulty
アンカーを変更したときは、このgolden setを実行してルーティング結果が壊れていないか確認してください。

## 留意点

- クレジット余力・予算・spend は LiteLLM に委譲（cobaiter は参照のみ）。
- LiteLLM 自体のサイレントなフォールバックは使わず、使用モデルは常に cobaiter が決定する。
- ストリーミングはストリーム**開始前**の失敗のみ自動フェイルオーバー。開始後の失敗は伝播。
- **1 つのユーザー指示の途中ではモデルを切り替えない**。ユーザーターン境界（新しい `user` メッセージ）でのみ
  品質駆動の再ルーティング（`context-switch`）を検討し、指示途中のツール往復は固定したまま。例外は強制
  フェイルオーバーのみ（可用性・制約違反は留まれないため切替）。
- 依存障害は 500 にせず明示化: **LiteLLM 到達不能 → 502**、**Valkey（状態ストア）到達不能 → 503**。
  `/healthz` は Valkey 障害時 `degraded` を返す。
- relevance/difficulty の算出に生成系 LLM は使わない（embedding 1 コール＋決定的ヒューリスティックのみ）。
  embedding 呼び出しが失敗した場合は全候補を一律 relevance 0.5 とみなして安全にフォールバックし
  （あとはルーターの cost/tier 再ランクが最安・最軽量を選ぶ）、ログに embedding モデル名・スコア・採否を出力する。
- API キー等の機密は `.env` で管理し、ログ・レスポンスに出さない。
