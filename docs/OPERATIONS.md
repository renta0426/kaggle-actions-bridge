# Operations Guide

## Operating principle

このrepositoryは、GitHub-hosted runnerからKaggleへ**事前定義された操作だけ**を要求する制御ブリッジです。一般目的のshell runnerとして運用しません。

実行済みrunから判明した失敗パターンは `OPERATIONAL_LESSONS.md` に記録し、同じ原因を再調査しないことを運用要件とします。

## Phases

### Phase 0: Bootstrap diagnostics

- Secretなし
- 外部Actionなし
- package installなし
- checkoutなし
- Kaggle、PyPI、GitHubへの認証なし到達性だけを確認
- actor、repository、event、ref、runner種別を記録

### Phase 1: Read-only authentication

- 現在の保護Environment名は **`kaggle-readonry`**。綴りを修正・推測して別名へ置換しない
- Kaggle credentialはこのEnvironmentからのみ受け取る
- Environment reviewerの承認後だけ実行
- 許可operationは認証確認、Competition一覧、metadata、file一覧などに限定
- write操作は別request/workflowに分離する

### Phase 2: Controlled downloads and notebook execution

Phase 1の監査後に必要な操作だけを追加します。Competition dataはrunnerの一時領域で扱い、Git、cache、artifactには保存しません。大容量・長時間処理はActions内で完結させず、Kaggle Notebookを起動して別の短いjobで状態を確認します。

### Phase 3: Write operations

Submission、Dataset/Model作成、Notebook公開などは別workflow、別承認フローへ分離します。各実行にrequest ID、重複防止、上限回数を必須とします。

## Allowed operation model

requestは任意コマンドではなく、次のような固定schemaを使用します。

```json
{
  "schema_version": 1,
  "request_id": "20260902-001",
  "operation": "competition_info",
  "competition": "example-slug"
}
```

実装時の最低条件:

- `operation`はcode内のallowlistに一致すること
- `request_id`は一意であること
- slug/refは厳格な正規表現に一致すること
- 未知fieldを拒否すること
- URL、shell、Python式、package名を入力として受け取らないこと
- request内容をshellへ展開しないこと

## Private research input policy

public bridgeのprotected jobは、private research repositoryを実行時にmaterializeできることを前提にしてはいけません。

- protected Kaggle jobはbridge commit + approved public/Kaggle inputsで自己完結させる
- public sourceから同一cohort/artifactを再構築できる場合は、credential追加ではなく再構築を選ぶ
- hash/revision/commitをSecret露出前に固定・検証する
- private repository accessが本当に不可欠なら、専用の承認済みmechanismを先に設計し、既存Kaggle Environmentへ広いGitHub権限を足して解決しない

## Trigger policy

Bootstrapでは、workflow file自身に対するownerの`push`のみを使います。本運用のtriggerは診断後にactor IDとrepository IDを固定してから決定します。

Secret付きworkflowで禁止するevent:

- `pull_request`
- `pull_request_target`
- `issues`
- `issue_comment`
- `workflow_run`
- `repository_dispatch`
- forkから制御可能なその他event

## Runner policy

```yaml
runs-on: ubuntu-24.04
permissions: {}
timeout-minutes: 5
```

- `self-hosted` labelは禁止
- `ubuntu-latest`ではなく固定versionを使用
- `concurrency`で重複実行を抑止
- local PCでworkflowやscriptを実行しない

## Resource admission

remote computeのconcurrencyはaccount全体の単一slotではなくresource class別に扱います。

- recent kernelsはbounded件数だけ調べる
- `RUNNING` / `QUEUED` / `PENDING`だけをactive扱いする
- exact metadataでCPU/GPU/TPUを分類する
- unknown resourceはfail closed
- requestの`max_active_runs`を要求resource classへ適用する
- preflight時と、実際のwrite直前の2回確認する
- blockerの診断が必要な場合はprivate refを公開logへ出さずSHA-256 identityを使う
- admission defer後の自動poll/retryは禁止。fresh run + fresh approvalとする

## Notebook working/output contract

**`/kaggle/working`はexport surfaceでありscratch spaceではありません。**

新規Notebookは次を満たします。

- Git clone、source checkout、temporary dataset、scratch cache、download cacheは `/tmp` に置く
- resumable shardが必要なら実行中だけ保持し、final consolidation後に削除する。ただしrequestで明示的outputとしたshardは除く
- successful completion時の `/kaggle/working` はrequestで宣言したfinal outputsだけにする
- `.env`、credential、Git metadata、private source treeを `/kaggle/working` に置かない
- final outputは名前、最大bytes、必要ならhash/schemaをrequestで固定する

これにより、current-output fallbackでNotebookのsaved working directory全体が取得されても、不要なmaterialをbridgeへ運ばない設計にします。

## Current-version Notebook output read

private Notebookのhistorical `scriptVersionId` / version-specific output取得はproduction capabilityではありません。

current outputを読む場合のみ、次の順序を固定します。

1. exact kernelを1件だけdiscoverする
2. terminal statusを確認する
3. metadataの `current_version_number` がrequestのexpected versionと**完全一致**することを確認する
4. 一致した場合だけcurrent-output readerを1回実行する
5. expected versionとcurrent versionが異なる場合は停止し、latestへ黙って置換しない

標準helperは `scripts/kaggle_current_output_read.py` です。このhelperはofficial `kaggle kernels output` のstdout/stderrをcaptureし、公開logへdownload file listを流しません。また、allowlist外のsaved filesを検出した新規workflowはfail closedします。

`kaggle kernels output` はnamed-file APIではなくsaved working directory全体をdownloadするため、上記Notebook working/output contractとセットでのみ通常運用します。既にworking directoryが汚れているlegacy kernelを読む必要がある場合は、そのfull-output fallbackをrequestに明記し、専用workflowでboundedに扱います。

## Secrets lifecycle

1. Bootstrap完了まではSecretを作らない
2. 旧credentialは失効させる
3. 新credentialをKaggle側で発行する
4. `kaggle-readonry` Environmentへ直接登録する
5. Chat、commit、Issue、PR、fileへ値を貼らない
6. 漏洩疑いがあれば即時失効し、新credentialへrotateする

## Logs and outputs

公開logへ出してよい情報:

- success/failure
- HTTP status class
- request ID
- 件数
- checksum/hash
- resource class
- runner、actor、repositoryの公開metadata

出してはいけない情報:

- credential値またはその一部
- Authorization header
- cookie、session情報
- 環境変数一覧
- Kaggle非公開dataやNotebook output本文
- broad CLI downloadのfile list
- 個人情報を含むAPI response

private operational identityが必要な場合はplaintextではなくhashを利用します。cache/artifactをprivate Kaggle materialの保管場所にしません。runner-local dataは`always()` cleanupで削除します。

## Failure repair procedure

失敗後は即rerunせず、以下を記録します。

1. exact failing step
2. Kaggle writeが発生したか
3. resource computeが開始したか
4. failure class: `pre-write` / `ambiguous-write` / `resource-consumed` / `read-only`
5. prior request ID / run ID
6. 新しく確認できたroot cause
7. 次requestで変更するmechanismを1つに限定できるか

その後、Secretなしvalidation → PR → main → fresh Environment approvalの順で再実行します。

write timeout/network errorなど結果が曖昧な場合は、target resourceの存在確認を先に行い、未確認のままwriteを再送しません。

## Change procedure

1. 変更の目的と必要権限を確認する
2. `docs/OPERATIONAL_LESSONS.md` を読み、既知failureを再実装していないか確認する
3. trigger、Secrets、network destination、dependencyへの影響を確認する
4. `SECURITY.md`と`THREAT_MODEL.md`の不変条件に違反しないか確認する
5. Secretなしのvalidationを先に実行する
6. write操作はread-only workflowへ混在させない
7. upstream library/OSSの不具合を発見した場合は、事象と再現条件を記録して報告する

## Emergency stop

異常時は次の順で停止します。

1. GitHub Actionsをrepository settingsで無効化
2. 実行中workflowをcancel
3. Kaggle credentialをKaggle側で失効
4. EnvironmentからSecretを削除
5. 不審なworkflow、commit、run logを確認
6. [Incident Response](INCIDENT_RESPONSE.md)へ移行
