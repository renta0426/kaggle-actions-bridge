# Operations Guide

## Operating principle

このrepositoryは、GitHub-hosted runnerからKaggleへ**事前定義された操作だけ**を要求する制御ブリッジです。一般目的のshell runnerとして運用しません。

## Phases

### Phase 0: Bootstrap diagnostics

- Secretなし
- 外部Actionなし
- package installなし
- checkoutなし
- Kaggle、PyPI、GitHubへの認証なし到達性だけを確認
- actor、repository、event、ref、runner種別を記録

### Phase 1: Read-only authentication

- `kaggle-readonly` Environmentを使用
- `KAGGLE_API_TOKEN`をEnvironment Secretとして登録
- Environment reviewerの承認後だけ実行
- 許可operationは認証確認、Competition一覧、metadata、file一覧などに限定
- download、Notebook push、submission、dataset/model変更は禁止

### Phase 2: Controlled downloads and notebook execution

Phase 1の監査後に必要な操作だけを追加します。Competition dataはrunnerの一時領域で扱い、Git、cache、artifactには保存しません。大容量・長時間処理はActions内で完結させず、Kaggle Notebookを起動して別の短いjobで状態を確認します。

### Phase 3: Write operations

Submission、Dataset/Model作成、Notebook公開などは別workflow、別Environment、別承認フローへ分離します。各実行にdry-run、request ID、重複防止、上限回数を必須とします。

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

## Secrets lifecycle

1. Bootstrap完了まではSecretを作らない
2. 旧tokenは失効させる
3. 新tokenをKaggle側で発行する
4. `kaggle-readonly` Environment Secretへ直接登録する
5. Chat、commit、Issue、PR、fileへ値を貼らない
6. 漏洩疑いがあれば即時失効し、新tokenへrotateする

## Logs and outputs

公開logへ出してよい情報:

- success/failure
- HTTP status class
- request ID
- 件数
- runner、actor、repositoryの公開metadata

出してはいけない情報:

- token値またはその一部
- Authorization header
- cookie、session、XSRF token
- 環境変数一覧
- Kaggle非公開dataやNotebook output
- 個人情報を含むAPI response

Bootstrap期間はcacheとartifactを使用しません。結果は必要最小限のlogだけにします。

## Change procedure

1. 変更の目的と必要権限を確認する
2. trigger、Secrets、network destination、dependencyへの影響を確認する
3. `SECURITY.md`と`THREAT_MODEL.md`の不変条件に違反しないか確認する
4. Secretなしのvalidationを先に実行する
5. write操作はread-only workflowへ混在させない
6. upstream library/OSSの不具合を発見した場合は、事象と再現条件を記録して報告する

## Emergency stop

異常時は次の順で停止します。

1. GitHub Actionsをrepository settingsで無効化
2. 実行中workflowをcancel
3. Kaggle tokenをKaggle側で失効
4. EnvironmentからSecretを削除
5. 不審なworkflow、commit、run logを確認
6. [Incident Response](INCIDENT_RESPONSE.md)へ移行
