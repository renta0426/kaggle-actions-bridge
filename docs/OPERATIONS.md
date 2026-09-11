# Operations Guide

## Operating principle

このrepositoryはGitHub-hosted runnerからKaggleへ**事前定義された操作だけ**を送る制御ブリッジです。一般目的shell runnerとして運用しません。

2026-09-11以降の新規・変更Kaggle実行workflowは [`EXECUTION_POLICY_V2.md`](EXECUTION_POLICY_V2.md) を正本とします。特に、Kaggle remote capacity/session/quotaのschedulerをbridge側で再実装しません。

実runから判明した失敗パターンは [`OPERATIONAL_LESSONS.md`](OPERATIONAL_LESSONS.md) に記録し、同じ原因を再調査しないことを運用要件とします。

## Phases

### Phase 0: Bootstrap / credential-free validation

- Secretなし
- Kaggle writeなし
- resource computeなし
- request parse、Python compile、Notebook materialization、hash、title/slug、dependency lock、synthetic compatibility testを可能な限り完了する
- policy-v2 launchではbridge-local capacity gateがないことを検査する

### Phase 1: Read-only authenticated identity checks

- 保護Environment名は **`kaggle-readonry`**。綴りを推測して変更しない
- Kaggle credentialはこのEnvironmentからのみ受け取る
- Environment reviewer承認後だけ認証付き操作を行う
- exact target/input version/status等、approved operationに必要なlive identityだけを確認する

### Phase 2: Notebook / Dataset / Model write or run-start

- approved operationにつきresource-starting/write callは最大1回
- Kaggle remote capacityはKaggleへ委ねる
- active session数、remaining quota、unrelated accelerator metadataをlaunch gateにしない
- write後はread-only reconciliationでside effectを確認する

### Phase 3: Submission / destructive / public operations

Submission、Dataset/Model/Notebook公開、delete等は通常run-startと分離し、別の明示承認を要求します。

## Allowed operation model

requestは任意コマンドではなく固定schemaを使用します。

```json
{
  "schema_version": 1,
  "request_id": "20260911-example-001",
  "execution_policy": "kaggle_native_capacity_v2",
  "operation": "save_kernel_once",
  "competition": "example-slug",
  "target": "owner/example-notebook",
  "resource": {
    "accelerator": "gpu",
    "machine_shape": "NvidiaTeslaT4",
    "expected_runtime_minutes": 100,
    "hard_timeout_minutes": 180
  },
  "side_effects": ["create one private Notebook version"],
  "automatic_compute_retries": 0
}
```

実装時の最低条件:

- `operation`はcode内allowlistに一致
- `request_id`は一意
- slug/refは厳格な形式
- requestから任意shell/Python式/URL/packageを受け取らない
- request内容を未検証でshell展開しない
- policy-v2 resourceは実行環境の指定であり、bridge capacity admissionではない

新規policy-v2 requestで`max_active_runs`や`min_remaining_quota_hours`等のbridge-local capacity fieldを使いません。

## Static validation versus protected execution

### Credential-free PR validationで完了するもの

- request JSON/schema
- Python source compile
- deterministic Notebook materialization
- source/blob/Notebook SHA-256
- title/slug/ref consistency
- frozen scientific config/formulas
- exact callable/signature/schema compatibilityのsynthetic check
- dependency lock / serializer version
- launch workflowにbridge-local capacity gateがないこと

PR validationの標準checkerは `scripts/kaggle_launch_policy_v2.py` です。

### Protected jobでのみ確認するもの

- repository / actor / event / workflow identity
- approved payload/hash
- Kaggle credential
- current target version/state（operation上必要な場合）
- required Kaggle input current version/status（科学条件上必要な場合）
- requested Notebook metadata
- single approved write/run-start

static assertionの大半をprotected jobに複製しません。これによりEnvironment承認後の「Kaggleへ到達する前のdeterministic failure」を減らします。

## Kaggle-native capacity policy

Kaggle側のCPU/GPU/TPU availability、remaining quota、同時実行可否はKaggleをauthorityとします。

標準launch pathでは次を実行可否判定に使いません。

- `quota_view()` remaining-time threshold
- `kernels_list()`によるunrelated active-run counting
- active NotebookのCPU/GPU/TPU分類
- unknown accelerator metadataのfail-closed refusal
- bridge独自concurrency limit

診断目的でcapacity metadataを読む場合はobservation-onlyです。取得失敗もlaunchを止めません。

Kaggleがcapacity不足等でwriteを拒否した場合、その応答を記録し、exact read-only reconciliationでnew side effectがないか確認します。

## One-shot write and reconciliation

- 1 request executionからwrite/run-start callは最大1回
- automatic compute retryは0
- response error/timeoutでも同じwriteを即再送しない
- expected version/resourceの存在をread-onlyに確認する

分類:

- expected side effectを確認 → write observed
- Kaggle rejection + side effectなし確認 → `platform_rejected_no_side_effect`
- side effectの有無を確定できない → `ambiguous_write`
- Kaggle run作成後Notebook内部失敗 → `resource_consumed_runtime_failure`

`platform_rejected_no_side_effect`がcapacity/quota/temporary availability由来なら、repair PRや新slugを作らず、同じimmutable requestを後でfresh Environment approvalで再実行できます。

`ambiguous_write`は再送前に必ずreconcileします。

## Private research input policy

public bridgeのprotected jobはprivate research repositoryを実行時に読めることを前提にしません。

- bridge commit + approved public/Kaggle inputsで自己完結させる
- public sourceから同一artifactを再構築できる場合はその方法を使う
- source revision/hashをPR validationで固定する
- private repository accessを増やしてpreflight failureを解決しない

## Trigger / runner policy

Secret付きworkflowで禁止するevent:

- `pull_request`
- `pull_request_target`
- `issues`
- `issue_comment`
- `workflow_run`
- forkから制御可能なevent

Runner baseline:

```yaml
runs-on: ubuntu-24.04
permissions: {}
timeout-minutes: <bounded>
```

- self-hosted runnerは禁止
- Secret付きjobはowner/main由来の固定boundaryを検証する
- local PCでworkflowを実行しない
- external Actionは原則不使用。例外はfull commit SHA固定・監査済みに限る

## Live Competition checks

Rules/Code Requirementsはrequest authoring時に確認します。

ただしgeneric Notebook launchのprotected jobで、操作に影響しないmutable HTML/page textを毎回parseしてhard gateにしません。live page/API checkがblockingでよいのは、現在値を確認しないとunauthorized submissionや明確なrule violationを起こす場合だけです。

## Notebook working/output contract

`/kaggle/working`はexport surfaceです。

- clone/source checkout/temporary dataset/cacheは`/tmp`
- successful completion時はdeclared final outputsだけを`/kaggle/working`へ残す
- `.env`、credential、Git metadata、private source treeを残さない
- final outputは必要に応じてname/max bytes/hash/schemaをrequestで固定

## Current-version Notebook output read

historical `scriptVersionId` / version-specific output取得はproduction capabilityではありません。

current output readの順序:

1. exact kernel identityを確認
2. terminal statusを確認
3. `current_version_number == expected_version`を確認
4. 一致した場合だけcurrent-output readerを1回実行
5. 不一致ならlatestへ黙って置換せず停止

標準helperは `scripts/kaggle_current_output_read.py` です。official `kaggle kernels output`を使うfallbackではstdout/stderrをcaptureし、allowlist/byte limit/unconditional cleanupを適用します。

## API / polling

- unbounded loop / unlimited paginationは禁止
- HTTP 429を高頻度retryしない
- writeをidempotency確認なしに再送しない
- pollingはboundedでkeep-alive化しない
- short startup failureを検知できる初期intervalを使い、その後backoffしてよい
- immutable metadataを同一request内で不必要に反復取得しない

capacityを予測するためのactive-session pollingは行いません。

## Data handling

- private Notebook output / Competition dataをGitHub log/cache/artifactへ保存しない
- runner-local dataはjob終了時に削除
- large dataは可能ならKaggle-sideで直接使用
- current-output fallbackはbroad file listをpublic logへ流さない
- Team外へのprivate sharingは禁止

## Failure repair procedure

失敗後に記録するもの:

1. exact failing step
2. Kaggle writeが発生したか
3. Kaggle computeが開始したか
4. failure class
5. prior request/run ID
6. established root cause

対応はfailure classで分けます。

### `static_validation_failure`

PR CIで修正します。Environment approvalやKaggle executionへ進めません。

### `prewrite_identity_or_authorization_failure`

approved target/input/authorizationが現在状態と一致しないため停止したものです。原因を確認し、必要なcontract変更だけを行います。

### `platform_rejected_no_side_effect`

Kaggle側capacity/quota/temporary availability等で拒否され、side effectなしをread-onlyに確認済みならコード修正しません。同じimmutable requestを後でfresh approvalでmanual re-executionできます。

### `ambiguous_write`

target/submission/dataset/model stateをexact read-only reconciliationしてから次の操作を決めます。未確認再送は禁止です。

### `resource_consumed_runtime_failure`

Notebook/runtime/science entry pointを診断します。既にcomputeを消費したためblind rerunはしません。

### `readout_failure`

computeを再実行せず、current exact output/status retrievalのみrepairします。

## Change procedure

1. [`OPERATIONAL_LESSONS.md`](OPERATIONAL_LESSONS.md) と [`EXECUTION_POLICY_V2.md`](EXECUTION_POLICY_V2.md) を読む
2. request/payload/workflowを作る
3. SecretなしCIでdeterministic failureを潰す
4. security/trigger/network/dependency impactを確認
5. PR review
6. main merge後にprotected Environment approval
7. single Kaggle operation
8. side effect reconciliation

## Secrets lifecycle

- Kaggle credentialは`kaggle-readonry` Environmentへ直接登録
- Chat/commit/Issue/PR/fileへ値を貼らない
-漏洩疑い時は即失効・rotate
- GitHub PAT/SSH key/cloud long-lived keyをEnvironmentへ追加しない

## Logs

公開logへ出してよいもの:

- success/failure class
- HTTP status class
- request ID
- counts/bytes/checksums
- public resource class
- public runner/actor/repository metadata

出してはいけないもの:

- credentials
- Authorization header/cookie/session
- private Notebook/data output本文
- broad output download file list
- private source/content

## Emergency stop

異常時はGitHub Actions cancel/disable、Kaggle credential失効、Environment Secret削除、run/audit確認、[`INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md)の順で対応します。
