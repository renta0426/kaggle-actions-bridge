# Bootstrap Diagnostic Result

## Result

- Workflow run: [Bootstrap diagnostic #2](https://github.com/renta0426/kaggle-actions-bridge/actions/runs/33596206511)
- Commit: `61a7c6518cb0257100232f87d1464a4ef72475a0`
- Conclusion: **PASS**
- Executed at: 2026-09-02 05:49 UTC / 2026-09-02 14:49 JST

## Verified identity and boundary

| Field | Observed value |
|---|---|
| Repository | `renta0426/kaggle-actions-bridge` |
| Repository ID | `1354356687` |
| Actor | `renta0426` |
| Actor ID | `71638068` |
| Triggering actor | `renta0426` |
| Event | `push` |
| Ref | `refs/heads/main` |
| Runner environment | `github-hosted` |
| Runner OS / architecture | `Linux` / `X64` |
| Runner image | `ubuntu-24.04` |

`@GitHub` Connectorによるcommitは、Actions上ではrepository owner本人のactorとして処理されました。このためEnvironmentでowner本人をrequired reviewerにする場合、`Prevent self-review`を有効にすると承認できません。単独運用では同設定を無効にします。

## Effective GitHub token permissions

Workflowには`permissions: {}`を設定しました。runner log上の実効権限は以下です。

```text
Metadata: read
```

GitHubが残すrepository metadataのread権限以外に、contents、actions、issues、pull requests、packages、OIDC等の権限は表示されませんでした。`GITHUB_TOKEN`をscriptへ明示的に渡していません。

## Credential check

次の資格情報用環境変数がjobへ注入されていないことを確認しました。

- Kaggle token / legacy username-key
- GitHub PAT / `GH_TOKEN`
- SSH private key
- AWS credentials
- Azure client secret
- Google application credentials

この検査は環境変数への注入有無を確認するもので、GitHub Settings内のSecret名一覧を列挙するものではありません。

## Toolchain

| Tool | Observed version |
|---|---|
| Python | `3.12.3` |
| Git | `2.55.0` |
| pip | `24.0` |

## Network reachability

すべてDNS解決、TLS、HTTPS応答まで成功しました。

| Target | HTTP status | Interpretation |
|---|---:|---|
| GitHub public API | `200` | 到達可能 |
| Kaggle home | `200` | 到達可能 |
| Kaggle public API | `401` | 到達可能。認証なしのため拒否される想定結果 |
| PyPI Kaggle metadata | `200` | 到達可能 |

これにより、GitHub-hosted runnerをKaggle API bridgeとして利用できること、およびPyPIから公式Kaggle CLIを取得できることが確認できました。Kaggle tokenの有効性はまだ試験していません。

## Bootstrap issue and correction

初回workflowは、`runner` contextをjob-level `env`で参照したためGitHubの定義検証で拒否され、jobは一度も開始されませんでした。`runner` contextを許可されるstep-level `env`へ移し、2回目のrunで全stepが成功しました。Secretは両runとも使用していません。

## Next gate

認証付き試験の前に、次が必要です。

1. `kaggle-readonly` Environmentを手動作成
2. owner reviewer、`main`限定、administrator bypass禁止を設定
3. 新しいKaggle tokenを同Environment Secretへ登録
4. `main` rulesetを有効化
5. read-only API smoke testをPR経由で追加

認証付きworkflowはEnvironmentが存在するまで作成しません。存在しないEnvironment名をworkflowから参照すると、無保護Environmentが自動作成される可能性を避けるためです。
