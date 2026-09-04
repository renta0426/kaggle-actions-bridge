# kaggle-actions-bridge

GitHub-hosted runnerからKaggleを**限定的・監査可能・人間承認付き**で操作するための公開ブリッジです。

このリポジトリの目的は、AIエージェントにKaggleの一般目的shellを与えることではありません。KaggleのCompetition Rules、Terms of Use、Acceptable Use Policy、Community Guidelines、APIのrate limit、Notebook quotaを守りながら、事前定義したML・データサイエンス操作だけを再現可能に実行することです。

> [!CAUTION]
> Kaggleの無料CPU/GPU/TPU、Notebook、Dataset、Model、API、storageを、汎用計算、サーバーファーム、ジョブファーム、無料ストレージ、クローラ、回避用プロキシとして使用してはいけません。Kaggleの上限値は「使い切る権利」ではなく、プラットフォーム側の最大値です。

## 現在の状態

次の経路をGitHub-hosted Ubuntu runner上で確認済みです。

- Bootstrap diagnostic
- `KAGGLE_API_TOKEN`による認証
- SHA-256 lock付きKaggle公式CLI 2.2.4
- commit固定したNVIDIA `nvidia-kaggle` skillのread-only操作
- private Datasetの作成とファイル存在確認
- 参加済みCompetitionの指定ファイルdownload
- Competition Discussion一覧とthread/commentの取得
- 所有するprivate Notebookのlatest versionのpull
- exact target/revision/resourceを固定したprivate Notebook push/run
- resource class別のCPU/GPU/TPU admissionとpre-write再確認
- current versionがapproved versionと一致する場合のprivate Notebook output read
- output read後のrunner-local aggregate evaluationと無条件cleanup

private Notebookの`scriptVersionId`を指定したhistorical version取得は、latest/current pullとは別機能です。現時点ではNVIDIAのarchive経路とKaggle内部API経路で成功していないため、production capabilityとして扱いません。要求されたversionを無視してlatestへ黙って置換することも禁止します。

current version outputについては、metadataで`current_version_number == expected_version`を確認した場合に限り、`scripts/kaggle_current_output_read.py`のcurrent-only contractを使用できます。official `kaggle kernels output` はsaved working directory全体を取得するため、Notebook側のoutput hygieneとセットで運用します。

Notebook push/run、Competition submission、Model操作、削除、公開範囲変更は、個別の検証と承認フローを持つrequestだけで使用します。汎用operationとしては有効化しません。

現在の保護Environment名は、既存設定に合わせて次の文字列です。**綴りを推測して修正しないでください。**

```text
kaggle-readonry
```

## 既知failureから確定した運用ルール

2026-09-04までの実runで、private-repository materialization、GPU admission、private Notebook output取得に複数の失敗経路がありました。再発防止の詳細は [`docs/OPERATIONAL_LESSONS.md`](docs/OPERATIONAL_LESSONS.md) に固定しています。

特に次は新規workflowの必須条件です。

- protected Kaggle jobをprivate research repositoryの実行時readに依存させない。可能ならpinned public sourceからSecret露出前に再構築する
- resource-consuming runはrequested accelerator class内でadmissionし、unknown resourceはfail closed、write直前に再確認する
- admission blockerのprivate refをpublic logへ出さず、必要ならSHA-256 identityだけを出す
- `/kaggle/working`をscratch spaceにしない。clone、source checkout、intermediate cacheは`/tmp`へ置き、successful completion時はdeclared final outputsだけを残す
- historical-version outputは未対応。current versionとapproved versionが完全一致する場合だけcurrent-output readを行う
- `kaggle kernels output`のstdout/stderrをpublic logへstreamしない。download file listを含むためcaptureする
- failure後はroot cause、write有無、resource消費有無を確定し、1つのrepairだけを入れてfresh approvalを取る。blind retryしない

## ルールの優先順位

AIエージェントは、実行のたびに次の順で最新情報を確認します。READMEの過去の記載や以前の成功runを、現在の許可根拠にしてはいけません。

1. 対象Competitionの`Rules`
2. Competitionの`Overview`、`Code Requirements`、`Data`、Host/Kaggle Staffの公式告知
3. [Kaggle Terms of Use](https://www.kaggle.com/terms)
4. [Kaggle Acceptable Use Policy](https://www.kaggle.com/aup)
5. [Kaggle Community Guidelines](https://www.kaggle.com/community-guidelines)
6. [Kaggle Public API documentation](https://www.kaggle.com/docs/api)
7. [Kaggle Notebook documentation](https://www.kaggle.com/docs/notebooks)
8. このリポジトリのポリシー

上位の公式ルールがこのリポジトリより厳しい場合は公式ルールを適用します。このリポジトリの方が厳しい場合は、このリポジトリの制約を維持します。ルールを取得できない、内容が曖昧、Host告知とRulesが矛盾する、または対象操作への適用が判断できない場合は**fail closed**とし、実行しません。

## Biohub Competitionで確認した例

2026-09-02に、[Biohub - Cell Tracking During DevelopmentのRules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)とCompetition pageを確認しました。これは確認方法の例であり、他Competitionへ転用する共通ルールではありません。

Competition固有の主な条件は次のとおりです。

- 1日最大5 Submissions、Final Submissionsは最大2件
- Teamは最大5名
- Team外でのCompetition code/dataのprivate sharingは禁止
- SubmissionはKaggle Notebook経由
- CPU NotebookとGPU Notebookはそれぞれ12時間以内
- Submission rerunではInternet accessを無効化
- 出力ファイル名は`submission.csv`
- External data/modelは、公開され、全参加者が実質的に利用可能で、過度な費用を要しないこと
- 勝者は、training/inference code、計算環境、必要resourceを含む再現可能な説明を求められる

RulesにはCompetition Dataの利用条件としてCC0が記載される一方、未同意者への無断提供を防ぐ合理的措置も要求されています。この公開ブリッジでは、Competition data、hidden-test関連情報、private Notebook、Notebook outputをGit、public log、cache、artifactへ保存しません。

リソース面で特に重要なのは、Competitionの「12時間以内」は提出Notebookの適格性に関する上限であり、連続起動、quotaの意図的な消化、複数アカウント、汎用計算、storage abuseを許可する条項ではないことです。Competition Rulesに書かれていない利用態様にも、KaggleのAUP、Terms、Community Guidelinesが適用されます。

## Kaggleのリソース関連ポリシー

KaggleのAUPは、提供resourceを使ったcryptomining、DDoS、server farming、malware/hacking/circumvention、ML・データサイエンスと無関係な活動、過剰なcontent crawlingを禁止しています。

Community Guidelinesでは、duplicate accountや、free storageなどのkernel resource abuseが即時banの対象になり得ることが明記されています。Termsでも、複数のactive Kaggle accountを保有・支配・運用することは禁止されています。

Public APIにはdynamic rate limitingがあります。HTTP 429または`Too many requests`を受けた場合、即時再試行や別endpointへの迂回をせず、処理を停止して、意図しないloop・重複call・過剰paginationがないか確認します。

## 絶対禁止事項

このリポジトリを利用する全エージェントは、次を実行してはいけません。

- 複数Kaggle account、代理account、別account tokenを使ってquota、submission limit、concurrency limit、banを回避する
- Notebookをworker node、常駐server、generic batch farm、無料GPU farmとして運用する
- cryptomining、DDoS、port scan、credential testing、malware、hacking tool、回避・難読化toolを実行する
- ML・データサイエンスと無関係な計算やcontent生成をKaggle resourceへ載せる
- keep-alive、再接続、無限loop、自動再起動によりsession/runtime制限を実質的に延長する
- quotaを消費するためだけにCPU/GPU/TPUを起動する、または不要なacceleratorを選択する
- Kaggle Dataset/Model/Notebook outputを一般的なbackup、artifact store、file relay、分割archive置場として使う
- 未使用・無関係なDatasetやModelをNotebookへattachする
- Kaggle contentを高頻度でcrawl、mirror、全件反復downloadする
- Competitionのsubmission上限まで自動的に連続提出する
- Competition data、private code、private Notebook、credentialをpublic repository、Actions log、cache、artifactへ出す
- 外部PR、Issue、comment、fork、任意URL、任意shell文字列をSecret付きjobで実行する
- exact version、file、dataset、competition等の要求を、成功させるために別resourceへ黙って置換する
- CIを緑にする目的で、要求条件、検証条件、security check、resource guardrailを弱める

## AIエージェント実行契約

### 1. 実行前のRule Preflight

認証付き操作やresource消費操作の前に、エージェントは最低限次を確認します。

- Competition slugと対象resourceの完全な識別子
- Rules、Overview/Code Requirements、Data、関連する公式告知の取得日時
- Submission/day、Final Submission、Team、code/data sharing、external data、Internet、runtimeの各制約
- Kaggle Terms、AUP、Community Guidelines上の禁止事項
- 現在のactive session/runと、利用可能なCPU/GPU/TPU quota
- download対象のファイル名、version、概算size
- API call上限、pagination上限、poll interval
- side effect、rollback、cleanup方法
- 同じ`request_id`が未実行であること

確認結果は、token、cookie、private contentを含まない短いmanifestとしてPRまたはrun logに残します。公式ページが取得できなければ、過去のcacheだけで実行してはいけません。

### 2. Request Manifest

resourceを消費する操作には、自由文ではなく検証可能なmanifestを使います。

```json
{
  "schema_version": 1,
  "request_id": "20260902-example-001",
  "competition": "competition-slug",
  "operation": "kernel_run",
  "target": "owner/notebook-slug",
  "resource": {
    "accelerator": "gpu",
    "expected_runtime_minutes": 180,
    "hard_timeout_minutes": 210,
    "max_active_runs": 1
  },
  "api_budget": {
    "max_calls": 50,
    "poll_interval_seconds": 300,
    "max_pages": 5
  },
  "side_effects": [
    "create one private notebook version"
  ],
  "automatic_compute_retries": 0,
  "rules_checked_at_utc": "YYYY-MM-DDTHH:MM:SSZ"
}
```

未知field、自由なshell/Python、任意URL、任意package、未検証slug、上限のない整数を拒否します。manifestの承認後に内容を変更した場合は、承認を取り直します。

### 3. 人間承認

Environment approvalは、**そのcommit、そのmanifest、その1回のrun**だけを許可します。将来のrunへの包括承認ではありません。

- public metadataのread-only確認を除き、認証付き操作は保護Environmentを通す
- Dataset/Model/Notebookの作成・更新、download、resource起動、submission、delete、公開範囲変更は事前承認を必須とする
- resourceを消費するrunの自動retryは禁止し、新しい原因確認と新しい承認を必要とする
- submissionとFinal Submission選択は、Notebook実行承認と分離する
- destructive operationとpublic化は、通常のwrite operationと分離する

### 4. Resource concurrency

複数エージェントが同じKaggle accountを同時に操作しても、安全性とquotaは共有されます。ただし、Kaggleのremote computeをaccount全体の単一slotとはみなしません。

- Notebook作成・更新、run起動などのbridge側admission/write操作は、共通のglobal concurrency groupまたは排他的leaseで直列化する
- remote runの同時実行可否はCPU・GPU・TPUのresource classごとに判断する
- manifestの`resource.max_active_runs`は、特記がなければ要求したaccelerator class内の上限として扱う
- CPU requestとGPU/TPU requestなど、異なるresource classのrunは、Competition Rules、Kaggleのlive制限、quota、個別承認が許す範囲で同時実行できる
- 同一resource classがmanifest上限に達している場合、またはactive runのresource classを確認できない場合は、writeせずdeferする
- defer時に自動pollingや自動再実行は行わず、再実行にはfresh Environment approvalを要求する
- 起動前にKaggle側のactive session/runを再確認し、resource class別件数だけを非機密metadataとして記録する
- blocker identityが診断に必要な場合はprivate refではなくhashを記録する
- 1 requestからKaggle runを開始するAPI callは1回だけにする
- timeout、network error、5xxで結果が不明な場合、再送前にKaggle側で作成済みか確認する

Kaggleまたは対象Competitionがより厳しい同時実行上限を示す場合は、その上限を適用します。並列性は作業上必要な範囲に限定し、quota消化の目標にしてはいけません。

### 5. Notebook run

- CPUで足りる処理にGPU/TPUを割り当てない
- accelerator type、expected runtime、hard timeout、Internet設定をmanifestで固定する
- exact logits/ranksなどbackend fidelityが科学条件に含まれる場合、CUDA実験をTPU/XLAへ黙って置換しない
- hard timeoutはCompetition/platformの上限以下とし、上限ぎりぎりを意図的な通常運用にしない
- `enable_internet`はCompetitionの提出条件に従う
- 無限学習、無限探索、無期限待機、外部job worker化を禁止する
- progressがない、入力が欠ける、output pathが想定外、quota情報が矛盾する場合は早期終了する
- Git clone、temporary source/data/cacheは`/tmp`へ置き、`/kaggle/working`にはdeclared final outputsだけを残す
- 失敗後にparameterを変えて自動再実行しない
- 完了後は不要なactive sessionが残っていないことを確認する

GitHub Actionsは長時間Notebookを監視し続けません。原則として、1回だけKaggle runを開始して終了し、状態確認は間隔を空けた別のbounded operationで行います。

### 6. API、polling、crawl

- unbounded loopと無制限paginationを禁止する
- API call総数と最大page数をmanifestに明記する
- 同じimmutable metadata/fileを1 request内で繰り返し取得しない
- status pollingは既定300秒以上、明示的理由がある場合でも60秒未満にしない
- HTTP 429では最低15分停止し、同じrun内で連打しない
- 403、401、404を別endpoint・別account・header偽装で迂回しない
- 5xx retryは指数backoff付きの少数回に限定し、write requestはidempotency確認なしに再送しない
- Discussion、Notebook、Datasetの全件取得は、目的、page上限、保存先、cleanupを明示する
- Web page scrapingよりKaggle公式CLI/APIを優先する

### 7. Downloadとdata handling

- 先にfile listとsizeを確認し、必要なnamed fileだけを取得する
- private Notebook current outputは、approved expected versionとcurrent versionが一致する場合だけ読む
- historical-version outputをlatest/currentへ黙って置換しない
- official CLIのcurrent-output fallbackを使う場合、stdout/stderrをcaptureし、broad file listをpublic logへ出さない
- current-output fallbackはallowlist、per-file/total byte limit、unexpected-file rejection、unconditional cleanupを持つ
- full Competition datasetの反復downloadを避ける
- 大容量dataをGitHub Actions経由で往復させず、可能ならKaggle Notebook内で直接使用する
- public Actions logへresponse body、Notebook source、Discussion本文、Competition dataを出さない
- GitHub cache/artifactをCompetition dataの保管場所にしない
- runner-local dataはjob終了時に削除する
- Team外へのprivate sharingを行わない
- licenseが公開可能に見えても、対象CompetitionのData Security条項を毎回確認する

### 8. Dataset、Model、storage

Kaggleへのuploadは、対象ML作業に直接必要なartifactだけに限定します。

- 既定はprivate
- title、owner、用途、source、license、作成request IDを記録する
- file countと合計sizeに明示的上限を置く
- generic backup、checkpoint倉庫、temporary relay、重複versionの量産を禁止する
- 作成後に存在、privacy、file一覧を確認する
- 不要になったresourceの削除は、別のdestructive approvalで行う
- synthetic dataはKaggleの表示要件に従って明示する

### 9. Submission

- Competitionの当日上限とremaining slotsを直前に再確認する
- 1承認につき最大1 Submission
- build/test成功を理由に自動submitしない
- network timeout後に重複submitしない。submission historyを先に確認する
- message、Notebook version、output file、checksumを固定する
- Final Submissionの選択は別の明示承認とする
- 複数accountやTeam mergeでsubmission上限を迂回しない

### 10. Auditと失敗時の扱い

run logへ残してよいのは、request ID、対象の公開識別子、operation、HTTP status class、件数、byte数、checksum、開始・終了時刻、resource種別、成功/失敗などの非機密metadataだけです。

次を検知したら直ちに停止します。

- 未承認または同一resource classの上限を超えるactive run、duplicate run、quota急減
- HTTP 429、ban/suspension警告、abuse警告
- 予期しない外向き通信
- credential、cookie、session情報、private contentのlog出力
- manifest外のDataset/Model/Notebook/submission作成
- cleanup失敗

失敗後は、exact failing step、write有無、resource消費有無、failure class、prior run/request、root causeを記録します。原因が異なる限り別repairとして扱い、blind rerunしません。詳細は [`docs/OPERATIONAL_LESSONS.md`](docs/OPERATIONAL_LESSONS.md) を参照してください。

停止後は、実行中runのcancel、GitHub Actionsの無効化、Kaggle credentialの失効、Environment Secret削除、run/audit確認の順で対応します。詳細は[Incident Response](docs/INCIDENT_RESPONSE.md)を参照してください。

## 標準実行フロー

```text
0. OPERATIONAL_LESSONSで既知failureを確認
1. 対象Competitionと操作を確定
2. 最新Rules/AUP/Guidelinesを取得
3. resource・API・side effectをmanifest化
4. Secretなしvalidation
5. PRでimmutable workflow/requestを確認
6. mainへmerge
7. Environmentで人間がその1回を承認
8. 1つの定型operationだけ実行
9. side effect、quota、active runを確認
10. runner-local dataを削除し、最小metadataを記録
```

## CLIとNVIDIA skillの参照先

CLI commandやNVIDIA skillの操作手順は、このREADMEへ複製しません。実際のcommand、metadata形式、Notebook/Dataset/Submission workflowは、利用するcommitに固定したupstream documentationを参照します。

- [NVIDIA/nvidia-kaggle](https://github.com/NVIDIA/nvidia-kaggle)
- [固定して検証したcommitのskill documentation](https://github.com/NVIDIA/nvidia-kaggle/tree/2b78cf29f5f30680764292a6592de8d53d4147a8/skills/nvidia-kaggle-skill)
- [Kaggle公式CLI](https://github.com/Kaggle/kaggle-cli)
- [Kaggle Public API documentation](https://www.kaggle.com/docs/api)

upstreamの手順は「どう操作するか」を定義し、このREADMEは「その操作をこのブリッジから実行してよい条件」を定義します。両方を満たさない操作は実行しません。

## セキュリティ境界

- GitHub-hosted runnerのみを使用し、self-hosted runnerは禁止する
- ローカルPCではこのリポジトリのworkflowを実行しない
- GitHub PAT、SSH秘密鍵、deploy key、クラウド長期資格情報を登録しない
- `permissions: {}`を既定とする
- Kaggle tokenは承認付きEnvironment Secretとしてのみ保持する
- protected Kaggle jobをprivate research repositoryのruntime accessへ依存させない
- 外部PR、fork、Issue、comment、`pull_request_target`、`workflow_run`からSecret付きjobを起動しない
- 外部Actionは原則不使用とし、必要時は完全なcommit SHAへ固定して動的依存まで監査する
- dependencyはversionとSHA-256を固定する

詳細は[SECURITY.md](SECURITY.md)と[THREAT_MODEL.md](THREAT_MODEL.md)を参照してください。

## 想定アーキテクチャ

```text
AI agent / GitHub Connector
        |
        | allowlist済みmanifestをPR
        v
Public GitHub repository
        |
        | protected main + Environment approval
        v
GitHub-hosted runner / permissions: {}
        |
        | bounded official CLI/API operation
        v
Kaggle
```

## 運用文書

- [Bootstrap result](docs/BOOTSTRAP_RESULT.md)
- [Operations](docs/OPERATIONS.md)
- [Operational lessons](docs/OPERATIONAL_LESSONS.md)
- [Incident response](docs/INCIDENT_RESPONSE.md)
- [Security policy](SECURITY.md)
- [Threat model](THREAT_MODEL.md)

## 非目標

このリポジトリは、一般用途CI、任意コード実行、Kaggle resourceの最大消費、Kaggle dataの保管、private repositoryへのアクセス、ローカルPCの遠隔操作、複数account管理、resource limitの回避を目的としません。
