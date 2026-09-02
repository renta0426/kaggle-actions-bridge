# kaggle-actions-bridge

GitHub-hosted runnerからKaggle APIを限定的に操作するための、公開・専用ブリッジです。

## 現在の状態

**Bootstrap diagnostic: PASS**

Secretなしで、runner種別、実行主体、権限、外向き通信を確認しました。結果は[Bootstrap Diagnostic Result](docs/BOOTSTRAP_RESULT.md)に記録しています。保護されたEnvironmentと新しいKaggle tokenが設定されるまで、認証付きKaggle操作は有効化しません。

## セキュリティ境界

このリポジトリは次の原則で運用します。

- GitHub-hosted runnerのみを使用する。self-hosted runnerは禁止する。
- ローカルPCでは、このリポジトリのコードやworkflowを実行しない。
- GitHub Personal Access Token、SSH秘密鍵、クラウド資格情報を登録しない。
- workflowの設定可能な`GITHUB_TOKEN`権限は原則として空にする。GitHubが残す`metadata: read`以外を付与しない。
- Kaggle認証情報は、承認付きEnvironment Secretとしてのみ保持する。
- 外部Pull Request、Issue、コメント、fork由来のコードをSecret付きjobで実行しない。
- `pull_request_target`、任意shell入力、任意URL取得、任意package指定を禁止する。
- Kaggle competition data、Notebook出力、モデル、認証情報をcommit・cache・artifactへ保存しない。
- 外部Actionを必要最小限にし、利用時は完全なcommit SHAへ固定して内容を監査する。

詳細は[SECURITY.md](SECURITY.md)と[THREAT_MODEL.md](THREAT_MODEL.md)を参照してください。

## 想定アーキテクチャ

```text
ChatGPT / GitHub Connector
        |
        | 許可された定型requestだけをcommit
        v
Public GitHub repository
        |
        | GitHub-hosted runner / permissions: {}
        v
Kaggle API
```

GitHub Actionsは任意コマンド実行基盤として使用しません。許可されたoperationをschemaで列挙し、入力値を検証してから固定実装を呼び出します。

## Bootstrap診断

`.github/workflows/00-bootstrap-diagnostic.yml`は次だけを行います。

- repository、actor、event、refの確認
- GitHub-hosted runnerであることの確認
- 危険な資格情報用環境変数が存在しないことの確認
- Kaggle、PyPI、GitHubへの認証なしHTTPS到達性確認
- Python、Git、pipのバージョン確認

外部Action、checkout、package install、Secret、cache、artifactは使用しません。

## 運用文書

- [Bootstrap result](docs/BOOTSTRAP_RESULT.md)
- [Operations](docs/OPERATIONS.md)
- [Incident response](docs/INCIDENT_RESPONSE.md)
- [Security policy](SECURITY.md)
- [Threat model](THREAT_MODEL.md)

## 非目標

このリポジトリは、一般用途のCI、任意コードの実行、Kaggleデータの保管、private repositoryへのアクセス、ローカルPCの遠隔操作を目的としません。
