# cobaiter: relevance/calibrationの強化（task_examples多点化 + decision log拡張 + golden test + 自動band較正）

## Context

現状のrelevanceスコアリングは「候補ごとにdescription 1本」を埋め込んで比較する設計で、
`cobaiter/schemas.py`のModelSpecがそれを裏付けている。ユーザーはcodex(別AI)のレビューを踏まえ、
以下の5点を今回のスコープとして選定した:

1. `task_examples`による複数プロトタイプ化（relevanceの中核強化）
2. decision logへの候補description/task_examplesスナップショット保存
3. golden set回帰テスト基盤
4. READMEのdiff正化（difficulty説明が実装とズレている）
5. relevanceのband自動較正（現状はdifficultyのみOLS自動フィット、relevanceは目視レビューのみ）

**明示的にスコープ外**（codexの提案のうち今回は見送り）: `negative_examples`/`domains`/`strengths`/
`weaknesses`フィールド、モデル信頼性（エラー率）ベースのペナルティ、複数ターン連結によるrelevance入力拡張、
cheap task-type feature追加。理由: 変更範囲を1回のPRで検証可能な単位に留めるため。将来やる場合は
memory（`cobaiter-multi-domain-routing`: ドメインは将来増える前提で設計する）を踏まえて別途計画する。

すでに`cobaiter/router.py`のdocstringに明記されている設計原則（最新userメッセージのみでdifficulty/relevanceを
判定し、過去ターンを混ぜない — 実測で混ぜるとdifficultyがドリフトすることを確認済み）は**変更しない**。
今回の変更はいずれも「候補側（レジストリ）の表現力」と「オフライン較正の自動化」の強化であり、
オンラインの推論ロジック（router.pyのhysteresis/cost-tier再ランキング等）には手を入れない。

## 1. `ModelSpec`にtask_examplesを追加し、relevanceを複数プロトタイプ化

**`cobaiter/schemas.py`**
- `ModelSpec`に `task_examples: list[str] = Field(default_factory=list)` を追加。
  docコメントで解決順序（`task_examples`があればそれ、空なら`[description]`にフォールバック）と
  複数プロトタイプscoring（top-2-mean、下記）を明記。`description`は維持（フォールバック先 + 人間可読な要約）。
- `ClassifierDiagnostics`に `candidate_refs: dict[str, list[str]] = Field(default_factory=dict)` を追加
  （§2で使用）。registryのメタデータであり会話本文ではないので、privacy(`needs_local`)でも redact 不要、
  とdocコメントに明記。

**`cobaiter/classifier.py`**
- キャッシュ `self._desc_vecs` を `self._ref_vecs`（description/task_examples共通の text->vector キャッシュ）に汎化。
- `_resolve_refs(spec: ModelSpec) -> list[str]`: `spec.task_examples or ([spec.description] if spec.description.strip() else [])`。
- `_multi_sim(task_vec, vecs: list[list[float]]) -> float`: **top-2-mean**（`_RELEVANCE_TOP_K = 2`）。
  `sorted(cosine(...) for v in vecs, reverse=True)[:min(2,len(vecs))]` の平均。
  理由: `task_examples`が1件（=フォールバック時）ならtop-1-mean=maxと完全一致し**既存挙動と完全後方互換**。
  かつ、このファイル自身が既に文書化している埋め込みモデルの短文brittleness（`_EDGE_PUNCT`のコメント:
  「こんにちは」と「こんにちは。」で類似度が有意に変わる実測）に対する具体的な緩和になる — 1本のexample
  文言の揺れにmax()だと丸ごと引きずられるが、2本平均なら緩和される。副次的に「example数が多いほどmaxが
  統計的に吊り上がる」比較上のバイアスも抑える。難易度側の`sim_easy`/`sim_hard`（固定・均等サイズではない
  exemplar集合に対するmax）は据え置き — あちらは登録者が自由に数を変える対象ではなく、比較対象間の公平性の
  問題が構造的に異なるため。
- `_embed_and_score_relevance`: `descs`ベースの単一文字列ロジックを`refs_per_candidate = [_resolve_refs(c) for c in candidates]`
  に置き換え、未キャッシュのref文字列を候補横断でdedupしてバッチembedding。戻り値を
  `(task_vec, relevance, raw_sims, refs_per_candidate)`の4-tupleに拡張。全候補のrefsが空なら
  従来通り早期return（`relevance=None`）。
- `relevance_from_sims`という名前で`_spread`を**public化**（`difficulty_from_ratio`が`cobaiter.calibrate`
  再利用のために既にpublicにされている前例に倣う）。docstringに「`cobaiter.calibrate`がband値をシミュレート
  するために再利用する」旨を明記。呼び出し箇所を更新。
- `score()`: `candidate_refs`をログ用に切り詰めて（`_REF_LOG_CHARS = 200`、decision logの肥大化対策 — §2参照）
  `candidate_sims`と同じフィルタ条件（sim is not None）で`ClassifierDiagnostics.candidate_refs`に格納。

**`models.yaml`**
- 各モデルに`task_examples`を3〜6件、日本語で具体的タスク文を追加（既存のdescriptionを分解する形。
  例: general系なら雑談/要約/翻訳/数学/旅行相談など個別文に分割、coding系ならバグ修正/レビュー/設計/
  CI調査/API相談など）。**STATE POSITIVELY**（既存description運用ルールを踏襲）。
- ヘッダコメントに`task_examples`フィールドの説明・解決順序・「ドメイン間でexample数を大きく偏らせない
  （目安3〜6件）」ガイダンスを追記。

**`cobaiter/registry.py`**
- モジュールdocstringの YAML サンプルに`task_examples:`を追記（コード変更は不要、バリデーションは汎用的）。

## 2. decision logへの候補description/task_examplesスナップショット保存

`store.py`/`registry.py`/`app.py`はModelSpec/ClassifierDiagnosticsを汎用的に扱っているため**コード変更不要**
（pydanticの`model_dump_json`/`model_validate_json`が新フィールドを自動的に扱う）。

サイズ対策: `candidate_refs`をdiagnosticsに積む際、各ref文字列を`_REF_LOG_CHARS=200`文字で切り詰める
（embeddingには全文を使うが、ログへのスナップショットのみ切り詰め）。ドメイン数が増えた将来を見積もると
1エントリ数KB→最大十数KBまで伸びうるので、README/config.pyのコメントに
「`task_examples`を広く使うなら`COBAITER_DECISION_LOG_MAXLEN`の引き下げも検討」と明記する
（デフォルト値そのものは今回変更しない）。

## 3. golden set回帰テスト基盤

**新規 `tests/fixtures/routing_cases.yaml`**: `{name, message, acceptable_models: [...], avoid_models: [...]}`
のリスト。現行registryの2ドメイン（coding/general）に加え、ドメイン境界に近いケース（例: 「このAPIの料金体系を
調べて」のような一般寄りだが技術語を含む文）を含め、relevance分離の脆さを継続検証できるようにする。
`models.yaml`のモデル集合が変わったら手動更新が必要である旨をファイル冒頭にコメントで明記。

**新規 `tests/test_routing_golden.py`**:
- 実LiteLLMゲートウェイの`/v1/embeddings`が必要なため、デフォルトでは**スキップ**。
  `pyproject.toml`の`[tool.pytest.ini_options]`に
  `markers = ["golden: requires a live LiteLLM/embedding backend; opt in with COBAITER_RUN_GOLDEN=1"]`を追加。
  テストモジュール冒頭で
  `pytestmark = [pytest.mark.golden, pytest.mark.skipif(os.environ.get("COBAITER_RUN_GOLDEN") != "1", reason=...)]`。
  通常の`uv run pytest`ではskip表示され、設定変更時は`COBAITER_RUN_GOLDEN=1 uv run pytest -m golden -v`で
  明示的に実行する運用。
- `get_settings()`ベースの実`LiteLLMClient`/`EmbeddingClassifier`を構築しつつ、registryは
  `settings.models_config`（未設定/コンテナパス想定）に頼らず、リポジトリルートの`models.yaml`を
  直接`load_model_registry()`で読み込む（`Path(__file__).resolve().parents[1] / "models.yaml"`）。
  Store は fakeredis で十分（会話状態は実Valkey不要、embeddingコールのみ実物を叩く）。
- `routing_cases.yaml`をパラメトライズし、各ケースで`await engine.decide(user_req(case.message), header_id=case.name)`
  を呼び、`decision.model in acceptable_models` / `not in avoid_models` を検証（RouteEngineフル経路を通す
  — constraint filter→classifier→capability-fit→cost/tier再ランキングまで含めて検証する）。
- fixtureのteardownで`client.close()`/`store.close()`を確実に呼ぶ。

## 4. READMEのdiff正化

- difficulty説明（現行「入力トークン数を基底に、意図キーワードで加点」という記述）を実装に合わせて修正:
  LOW_INTENTキーワードによる決定的オーバーライド（0.15固定）が最初にチェックされ、それ以外は
  embedding exemplar比（`sim_hard/(sim_hard+sim_easy)`、`difficulty_easy_anchor`/`hard_anchor`で
  0.15〜0.85にrescale）が主軸、そこに`+0.05`（エラーマーカー）/`+0.03`（code fence）の小さな加点が乗る。
  トークン数ベースの推定（`_fallback_difficulty`）はembeddingが完全に使えない場合のみのフォールバック。
- relevanceの節に`task_examples`とtop-2-mean多点化の説明を追加し、フォールバック規則
  （`task_examples`空なら`description`を1件のexampleとして扱う）を明記。
  `task_examples`導入でraw cosineのスケールが変わりうる（1本の最良一致→2本平均で分布が変わる）ため、
  band再較正（§5）を併せて回す運用を明記。
- `models.yaml`サンプルスニペットに`task_examples:`を追記。
- 再キャリブレーション節に、band自動グリッドサーチ（§5）と`candidate_refs`ベースのグルーピング、
  judge付きの曖昧relevanceフラグ一覧を追記。
- テスト節にgolden testと`COBAITER_RUN_GOLDEN=1`の使い方を追記。

## 5. relevanceのband自動較正（`cobaiter/calibrate.py`拡張）

現状`_find_ambiguous_relevance`は近接マージンの目視レビュー一覧のみ。difficultyのOLS自動フィットに相当する
仕組みをrelevanceにも用意する。band(`embedding_rel_band`)の効きはargmaxに対して閾値的（区分線形）なので、
**RMSE回帰ではなくグリッドサーチ+正解率（accuracy）**を採用する（既にすべてembedding済みの`candidate_sims`
に対する後処理のみなのでLLM呼び出しは追加不要 — グリッドは`[0.01, 0.02, ..., 0.40]`の40点で十分細かく取れる）。

- `_JUDGE_RELEVANCE_SYSTEM` + `_judge_relevance(client, model, task_text, groups: list[list[str]]) -> int | None`:
  `_judge_difficulty`と同じJSON抽出パターンで`{"domain_index": N}`をパース（`-1`=「該当なし」として
  正解率の分母から除外）。
- `_group_candidates(diagnostics) -> list[tuple[list[str], list[str]]]`: `candidate_refs`が完全一致する
  モデルを1グループにまとめる（think/no-thinkペアのような同一ドメインのtier違いをjudgeに区別させない
  ため）。最小モデル名でソートし安定した順序を保証。
- `RelevanceCalibration`データクラス（n, current_band, current_accuracy, suggested_band, suggested_accuracy, note）、
  `RelevanceFlag`に`judge_domain: int | None`/`agreed_with_router: bool | None`を追加。
- `_calibrate_relevance(entries, client, settings) -> tuple[RelevanceCalibration, dict[tuple[str,int], int]]`:
  `candidate_refs`+`candidate_sims`(>=2グループ)+`task_text`を持つエントリをサンプリングし、
  各エントリを1回だけjudgeにかけてグループindexを得る（`(conversation_key, turn) -> judge_group_index`の
  辞書を構築）。現行bandとグリッド各点でのaccuracyを計算し、同点ならより小さいband（＝より鋭いドメイン
  分離、READMEの既存方針と一致）を採用して提案する。
- `_find_ambiguous_relevance`のシグネチャにjudge-labelの辞書（省略可、デフォルト`{}`）を追加し、
  判定済みエントリには`judge_domain`/`agreed_with_router`を付与（同じjudge呼び出しをflag一覧とband較正の
  両方で使い回し、二重にLLMを叩かない）。
- `run_calibration`: `_calibrate_relevance`を呼び、judge-label辞書を`_find_ambiguous_relevance`に渡す。
  `CalibrationReport`に`relevance: RelevanceCalibration`を追加。
- `format_report`: `== relevance band (COBAITER_EMBEDDING_REL_BAND) ==`セクションを追加
  （current vs suggested band + accuracy）。曖昧フラグの出力行に`judge=<index|n/a> agreed=<bool|n/a>`を追記。
- `classifier.py`の`relevance_from_sims`をimportして再利用（§1で public化済み）。

新規Settingsフィールドは不要（既存の`calibration_judge_model`/`calibration_sample_size`を再利用。
最小サンプル数は`calibrate.py`内モジュール定数`_MIN_RELEVANCE_SAMPLES = 20`を追加するのみ）。

## テスト計画（既存テストは破壊しない — grep確認済み: `_spread`/`_desc_vecs`はテストコードから直接参照されていない）

**`tests/test_classifier.py`（追加）**
- 複数`task_examples`のtop-2-mean選択を確認するテスト（3例中2例が近い場合に近い2つの平均になること）
- `task_examples`未設定時に`description`単体へフォールバックすることを明示的にpinするテスト
- `task_examples`が個別にembedding・キャッシュされること（`test_shared_description_is_deduped_and_cached`と
  同型パターン）
- example数が多い候補が「数の多さだけ」で勝たないことを示す回帰テスト（1例 vs 5例中1例が近い、で
  1例側が不当に負けないこと）
- `result.raw.candidate_refs`が解決済みrefsと一致すること（フォールバック/task_examples両パターン）

**`tests/test_decision_logging.py`（追加）**
- privacy会話で`candidate_refs`はredactされず、`task_text`のみredactされることを確認するテスト
  （`_diagnostics()`ヘルパーに`candidate_refs`引数を追加）

**`tests/test_store.py`（追加）**
- `candidate_refs`を含むDecisionLogEntryがValkey streamをラウンドトリップすることの確認

**`tests/test_calibrate.py`（追加）**
- `_group_candidates`のグルーピング単体テスト
- `_judge_relevance`のパース系テスト（`_judge_difficulty`のテスト群と同型: 正常/プロース混じり/範囲外/不正）
- `test_calibrate_relevance_grid_search_recovers_known_optimal_band`:
  既知の最適band値になるよう合成した`candidate_sims`/`candidate_refs`とscripted judge応答で、
  `run_calibration`後の`report.relevance.suggested_band`が既知値（グリッド刻み以内）に収束し
  `suggested_accuracy == 1.0`になることを確認（既存`test_run_calibration_suggests_anchors_matching_judge_labels`
  と同型のend-to-end検証）
- 同点accuracy時に小さい方のbandが選ばれることの確認
- 曖昧flagにjudge verdictが付与される/されない（サンプル外）の確認

**`tests/test_registry.py`**: 既存`_YAML`（description-onlyフィクスチャ）はそのまま通る
（`task_examples`はデフォルト空リスト）。`task_examples`を含むYAMLの読み込みを確認するテストを追加。

## 検証方法

1. `uv run pytest` — 既存＋新規の単体/統合テストが全て通ること（golden testは環境変数未設定でskip表示になる）。
2. `COBAITER_RUN_GOLDEN=1 uv run pytest -m golden -v`（ローカルでLiteLLMゲートウェイ+embeddingモデルが
   立ち上がっている前提 — `docker compose up -d valkey litellm` 等）でgolden setが実際のembeddingモデルに
   対して妥当なルーティングになることを確認。
3. `uv run python -m cobaiter.calibrate`（`COBAITER_CALIBRATION_JUDGE_MODEL`設定要）を実データ/テストデータに
   対して実行し、新しい`relevance band`セクションがレポートに出力されること、既存のdifficultyセクションが
   壊れていないことを目視確認。
4. README記載のコマンド例・設定説明が実装と一致していることを再読して確認。
