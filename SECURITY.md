# Security Policy

## Scope

このポリシーは、`main`上のGitHub Actions workflow、request schema、Kaggle API client、運用文書に適用します。

## Security invariants

次の条件は常に維持します。

1. `runs-on`はGitHub-hosted runnerの固定ラベルだけを使用する。
2. workflowの権限は、必要性が証明されない限り`permissions: {}`とする。
3. GitHub PAT、SSH秘密鍵、deploy key、クラウドの長期資格情報をSecretsへ登録しない。
4. Kaggle tokenはRepository Secretではなく、保護されたEnvironment Secretとしてのみ登録する。
5. Secret付きjobは、外部Pull Request、fork、Issue、コメント、`pull_request_target`、`workflow_run`から起動しない。
6. requestから任意shell、Python式、URL、package、workflow pathを受け取らない。
7. Kaggle data、Notebook出力、モデル、tokenをGit履歴、log、cache、artifactへ保存しない。
8. 外部Actionは原則不使用とし、例外時は完全なcommit SHAへ固定し、Actionとその動的依存を監査する。
9. dependencyの自動更新、floating tag、`curl | sh`、未固定の`pip install -U`を禁止する。
10. Competition submission、Dataset/Model公開、Notebook公開などの外部変更操作は、read-only操作と分離し、明示承認なしでは実行しない。

## Credential policy

許可候補は次だけです。

- `KAGGLE_API_TOKEN`: 承認付きEnvironment Secret

禁止対象には以下を含みます。

- Classic/Fine-grained GitHub PAT
- 個人または仕事用SSH秘密鍵
- `gh auth`やGit Credential Managerから抽出したtoken
- AWS、Azure、Google Cloudの長期access key
- 他repositoryを読めるdeploy keyまたはGitHub App秘密鍵

Secret値をworkflow、request、Issue、Pull Request、commit、log、artifactへ貼り付けてはいけません。

## Workflow policy

- `set -x`を使用しない。
- 環境変数一覧を出力しない。
- HTTP Authorization headerを出力しない。
- 認証付きresponse bodyを無条件に出力しない。
- `ACTIONS_STEP_DEBUG`と`ACTIONS_RUNNER_DEBUG`を有効化しない。
- `timeout-minutes`と`concurrency`を必ず設定する。
- checkoutが不要なjobでは`actions/checkout`を使用しない。
- cacheとartifactは、機密性とKaggle利用規約の検証が完了するまで使用しない。

## Dependency policy

外部dependencyを導入する場合は、次を満たす変更だけを受け入れます。

- 正確なversionを固定する。
- Python packageは可能な限りwheelだけを許可する。
- lockfileにSHA-256 hashを記録し、`--require-hashes`を使う。
- install時script、dynamic download、transitive dependencyを確認する。
- upstreamのsecurity advisory、release provenance、既知issueを確認する。
- 異常やOSS側の不具合を発見した場合、再現条件を記録してownerへ報告する。

## Vulnerability reporting

公開Issueへ認証情報や未修正の脆弱性詳細を投稿しないでください。GitHubのPrivate vulnerability reportingが利用可能な場合はそれを使用し、利用できない場合はrepository ownerへ非公開経路で連絡してください。

## Incident handling

資格情報漏洩、予期しないworkflow起動、未知の外向き通信、workflow改変、Kaggle上の予期しない操作が確認された場合は、[Incident Response](docs/INCIDENT_RESPONSE.md)に従います。
