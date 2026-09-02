# Threat Model

## Objective

Public GitHub ActionsをKaggleへの限定ブリッジとして利用しつつ、Kaggle token、GitHubアカウント、他repository、ローカルPCへの被害拡大を防ぎます。

## Protected assets

1. Kaggle API tokenとKaggleアカウント
2. このrepositoryのworkflowとrequest定義の完全性
3. Competitionの提出枠、Dataset、Notebook、ModelなどのKaggle資産
4. 他のprivate repositoryとGitHub認証情報
5. 個人PC・仕事PC上の認証情報、ファイル、社内ネットワーク

## Trust boundaries

```text
Untrusted Internet / forks / public inputs
                  |
                  X  Secret付きjobへは接続しない
                  |
GitHub repository + protected main
                  |
                  v
Ephemeral GitHub-hosted runner
                  |
                  v
Kaggle HTTPS API
```

次は信頼しません。

- fork、Pull Request、Issue、コメントなど第三者が変更できる内容
- floating tagのAction、未固定package、実行時に取得されるbinary
- request内の文字列
- public log、cache、artifact
- HTML、Discussion、Notebook metadataなどKaggleから取得した外部コンテンツ

## Threat actors

- 公開repositoryを利用する外部攻撃者
- 悪意あるfork/PR/Issue作成者
- compromiseされたGitHub Action、PyPI package、upstream release
- 誤設定または誤操作したrepository owner
- 外部コンテンツに埋め込まれたprompt injectionに影響されたautomation

## Principal attack paths and controls

### 1. External code executes with Secrets

**Attack:** `pull_request_target`、Issue comment、`workflow_run`などを利用して攻撃者のコードをSecret付きjobで実行する。

**Controls:** Secret付きworkflowは許可済み`push`と保護Environmentだけから起動し、外部PRコードをcheckoutしない。危険なeventを定義しない。

### 2. Supply-chain compromise

**Attack:** Action tag、PyPI dependency、download binaryが改ざんされ、runner内のtokenを外部送信する。

**Controls:** bootstrapでは外部dependencyをゼロにする。導入時は完全versionとhashへ固定し、動的downloadも監査する。可能なら標準ライブラリによる直接API clientを優先する。

### 3. Command injection

**Attack:** competition slug、Notebook ref、messageなどをshellへ展開し、任意コマンドを実行する。

**Controls:** operationをallowlist化し、入力を厳格なschemaと正規表現で検証する。`shell=True`、`eval`、`exec`、文字列連結したshell commandを禁止する。

### 4. Secret exfiltration through logs or artifacts

**Attack:** token、Authorization header、response、環境変数をlog/cache/artifactへ残す。

**Controls:** token値を表示しない。環境変数一覧とdebug tracingを禁止する。cache/artifactは初期運用では使用しない。log retentionを最小化する。

### 5. Lateral movement to other GitHub repositories

**Attack:** PAT、SSH鍵、広域GitHub App tokenを盗み、private repositoryへアクセスする。

**Controls:** それらをrunnerへ渡さない。workflowの`GITHUB_TOKEN`権限を空にし、このrepository以外のcredentialを登録しない。

### 6. Compromise of local or work PC

**Attack:** self-hosted runnerまたはlocal executionを通じて、PC上の鍵や社内networkへアクセスする。

**Controls:** GitHub-hosted runnerだけを使用し、self-hosted runnerとlocal executionを明示的に禁止する。

### 7. Abuse as a public execution/proxy service

**Attack:** 第三者が任意URL、任意package、任意commandを指定し、Actionsを踏み台として利用する。

**Controls:** public入力eventを無効化し、固定operationだけを実装する。任意network destinationを受け付けず、Kaggleの固定hostだけを許可する。

### 8. Destructive or quota-consuming Kaggle actions

**Attack:** Competition submission枠消費、Notebook/Dataset公開、削除、反復実行によるquota消費。

**Controls:** read-onlyとwrite操作を別workflow・別Environmentに分離する。write操作は明示承認、idempotency key、rate limit、dry-runを必須とする。

### 9. Prompt injection through repository or Kaggle content

**Attack:** README、request、Discussion、Notebook本文に「Secretを表示せよ」などの命令を埋め込み、agentやautomationを誘導する。

**Controls:** 取得コンテンツはdataとして扱い、workflow仕様や許可operationを変更させない。Secret操作、workflow変更、権限拡張はrepository ownerの明示指示とcode reviewなしで行わない。

## Residual risk

Kaggle tokenをGitHub-hosted runnerへ渡す限り、Secret付きjob内で実行される信頼済みコードまたはdependencyが侵害された場合、そのtokenは窃取され得ます。Environment承認、依存排除、固定workflowにより確率と影響を抑えますが、暗号学的な完全防御ではありません。

より強い最終構成では、Kaggle tokenを外部brokerに保持し、Actionsには短命なGitHub OIDC tokenだけを渡します。brokerがrepository ID、actor ID、ref、workflow SHAを検証し、許可されたKaggle操作だけを代理実行します。

## Security acceptance criteria

認証付きworkflowを有効化する前に、以下を確認します。

- repositoryとrunnerのIDが期待値に一致する
- runnerが`github-hosted`である
- workflowの権限が空である
- Kaggle/PyPIへの通信確認がSecretなしで成功する
- workflow triggerが許可eventだけである
- Environment approvalが機能する
- tokenがlog、artifact、cacheへ現れない
- write系Kaggle操作が実装されていない
