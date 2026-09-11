# kaggle-actions-bridge

GitHub-hosted runnerからKaggleを**限定的・監査可能・人間承認付き**で操作するための公開ブリッジです。

このrepositoryの目的は、AIエージェントにKaggleの一般目的shellを与えることではありません。KaggleのCompetition Rules、Terms of Use、Acceptable Use Policy、Community Guidelinesを守りながら、事前定義したML・データサイエンス操作を再現可能に実行することです。

> [!CAUTION]
> KaggleのCPU/GPU/TPU、Notebook、Dataset、Model、API、storageを、汎用計算、サーバーファーム、ジョブファーム、無料ストレージ、クローラ、回避用プロキシとして使用してはいけません。

## 最重要: Execution Policy v2

2026-09-11以降の新規・変更Kaggle実行workflowは [`docs/EXECUTION_POLICY_V2.md`](docs/EXECUTION_POLICY_V2.md) を正本とします。

**Kaggle側のCPU/GPU/TPU session capacity、account quota、concurrent Notebook availabilityはKaggle自身を判定主体とします。** bridgeが独自にactive session数や残quotaを推定して、承認済みNotebook launchを止めてはいけません。

したがって、新規・変更workflowでは次をlaunch gateにしません。

- remaining CPU/GPU/TPU quotaのbridge独自閾値
- unrelated `RUNNING` / `QUEUED` / `PENDING` Notebook数
- unrelated active NotebookのCPU/GPU/TPU分類
- bridge独自の同時実行上限
- unrelated active Notebookのaccelerator metadataを取得できないこと
- Kaggleがcapacity不足で拒否するだろう、というbridge側の予測

GPU枠等が実際に利用できない場合は、承認済みの1回のKaggle operationをKaggleまで到達させ、Kaggle側の応答を事実として扱います。

これはCompetition Rules、Kaggle Terms/AUP、security boundary、exact target identity、明示的なユーザー制約を弱める変更ではありません。

## 現在確認済みの経路

GitHub-hosted Ubuntu runner上で次を確認済みです。

- Bootstrap diagnostic
- `KAGGLE_API_TOKEN`による認証
- SHA-256 lock付きKaggle公式CLI 2.2.4
- commit固定したNVIDIA `nvidia-kaggle` skillのread-only操作
- private Datasetの作成とfile存在確認
- 参加済みCompetitionの指定file download
- Competition Discussion一覧とthread/comment取得
- 所有するprivate Notebookのlatest/current version read
- exact target/revision/resourceを固定したprivate Notebook push/run
- current versionがapproved versionと一致する場合のprivate Notebook output read
- output read後のrunner-local evaluationと無条件cleanup

private Notebookのhistorical `scriptVersionId`指定output取得はproduction capabilityとして扱いません。要求versionを無視してlatest/currentへ黙って置換することは禁止します。

current version outputは、metadataで`current_version_number == expected_version`を確認した場合に限り、`scripts/kaggle_current_output_read.py`のcurrent-only contractを使用できます。official `kaggle kernels output`はsaved working directory全体を取得するため、Notebook側のoutput hygieneとセットで運用します。

## 保護Environment

現在のEnvironment名は既存設定に合わせた歴史的な綴りです。**推測して修正しないでください。**

```text
kaggle-readonry
```

認証付きのwrite/run-start/download等は原則としてこの保護Environmentを通し、Environment reviewerの人間承認後にだけ実行します。

## ルールの優先順位

実行時は次の順で適用します。

1. 対象CompetitionのRules / Code Requirements / Host・Kaggle Staffの公式告知
2. Kaggle Terms of Use / Acceptable Use Policy / Community Guidelines
3. Kaggle公式API・Notebook仕様
4. 明示的なユーザー指示
5. このrepositoryのexecution/security policy

上位の公式ルールがより厳しい場合は公式ルールを適用します。bridge独自の利便性・quota節約・session管理のために、公式ルールにもユーザー指示にもないremote-capacity制約を追加しません。

## 絶対禁止事項

- 複数Kaggle accountや代理accountでquota、submission limit、concurrency limit、banを回避する
- Notebookをworker node、常駐server、generic batch farm、無料GPU farmとして運用する
- cryptomining、DDoS、port scan、credential testing、malware、hacking、circumventionを行う
- ML・データサイエンスと無関係な計算をKaggle resourceへ載せる
- keep-alive、無限loop、自動再起動でsession/runtime制限を実質的に延長する
- Kaggle Dataset/Model/Notebook outputを一般backupやfile relayとして利用する
- Competition data、private code、private Notebook、credentialをpublic repository、Actions log、cache、artifactへ出す
- 外部PR、Issue、comment、fork、任意URL、任意shell文字列をSecret付きjobで実行する
- exact version/file/dataset/competition等を、成功させるために別resourceへ黙って置換する
- ambiguous writeを未確認のまま再送する

## AIエージェント実行契約

### 1. Request manifest

resourceを消費する新規requestは自由文ではなく検証可能なmanifestを使い、policy v2を明示します。

```json
{
  "schema_version": 1,
  "request_id": "20260911-example-001",
  "execution_policy": "kaggle_native_capacity_v2",
  "competition": "competition-slug",
  "operation": "save_kernel_once",
  "target": "owner/notebook-slug",
  "resource": {
    "accelerator": "gpu",
    "machine_shape": "NvidiaTeslaT4",
    "expected_runtime_minutes": 100,
    "hard_timeout_minutes": 180
  },
  "side_effects": [
    "create one private notebook version"
  ],
  "automatic_compute_retries": 0
}
```

policy v2では`resource`は**要求する実行環境の記述**であり、bridge-local capacity admissionではありません。

新規policy-v2 requestでは、bridge独自のsession/quota gate field（例: `max_active_runs`, `min_remaining_quota_hours`）を使用しません。詳細とmigration ruleは [`docs/EXECUTION_POLICY_V2.md`](docs/EXECUTION_POLICY_V2.md) を参照してください。

### 2. 二段階validation

Kaggle到達前の実装ミスでEnvironment承認を何度も消費しないため、static correctnessとprotected runtime authorizationを明確に分離します。

#### Stage A: SecretなしPR CI

可能な限りここで完了させます。

- request JSON parse/schema
- Python compile
- exact Notebook materialization
- Notebook/source/blob SHA-256
- title/slug/ref一致
- dependency/serializer version lock
- frozen scientific config/formula contract
- callable/signature/schema compatibilityのsynthetic test
- bridge-local capacity gateが混入していないこと

新規policy-v2 workflow/requestには `scripts/kaggle_launch_policy_v2.py` のcredential-free lintを適用します。

#### Stage B: protected execution

Stage A通過後のprotected jobは小さく保ちます。原則として次だけを再確認します。

- repository / actor / event / workflow identity
- approved payload/hash
- Kaggle credential availability
- exact target version/state（operation上必要な場合）
- exact required Kaggle input version/status（科学条件上必要な場合）
- Notebook metadataとして要求したprivacy/accelerator/Internet設定
- 1回のapproved Kaggle write/run-start

PR CIで既に証明した大量のstatic assertionをprotected jobへ複製しません。unrelated active sessionやquotaを追加で調べてlaunchを止めません。

### 3. 人間承認

Environment approvalは、そのcommit・request・1回の実行を承認します。

- Dataset/Model/Notebook write/run-start、submission、delete、public化等は明示承認を必要とする
- submissionとFinal Submission選択は分離する
- destructive/public operationは通常writeと分離する
- automatic compute retryは行わない

### 4. Kaggle write/run-start

1 request executionからresource-starting/write callは最大1回です。

write後はread-onlyにreconcileし、clientの戻り値だけで副作用有無を推測しません。

- expected version/resourceが確認できた → write observed
- Kaggleが拒否し、new version/resourceが存在しないことを確認できた → `platform_rejected_no_side_effect`
- write結果を証明できない → `ambiguous_write`
- new version/runが存在する → client errorがあっても同じwriteを再送しない

### 5. Capacity/platform rejection

KaggleがGPU/TPU/CPU枠、quota、temporary platform availability等で拒否し、read-only reconciliationで副作用なしを確認できた場合、**repair PRを作りません**。

同じimmutable requestを、後でfresh Environment approvalを得て再実行できます。これはautomatic retryではなく、同一operationのmanual re-executionです。

bridge独自capacity heuristicの失敗を理由にsuccessor requestを量産する運用は廃止します。

### 6. Notebook working/output contract

`/kaggle/working`はexport surfaceとして扱います。

- clone/source checkout/temporary data/cacheは`/tmp`
- `/kaggle/working`にはdeclared final outputsだけを残す
- credential、`.env`、Git metadata、private source treeを置かない
- final outputはname/size/schema/hashを必要に応じて固定する

### 7. API / polling

- unbounded loopと無制限paginationは禁止
- HTTP 429を高頻度retryしない
- write requestを結果確認なしに再送しない
- status pollingはboundedにし、keep-alive化しない
- immutable metadata/fileを同一request内で不必要に反復取得しない

remote capacityを予測するための`quota_view()`やactive-kernel enumerationは標準launch pathから外します。診断目的で取得する場合もobservation-onlyであり、取得失敗をlaunch blockerにしません。

### 8. Submission

- 明示的に承認されたCompetition/submissionだけを送る
- 1承認につき最大1 submission write call
- build/test成功を理由に自動submitしない
- message、Notebook version、output file/checksumを固定する
- timeout等で結果が曖昧ならsubmission historyを先にreconcileする
- Final Submission選択は別の明示承認とする

Competition側の日次上限や適格性は公式ルールに従います。bridge独自のsubmission tuning policyをCompetition ruleと混同しません。

## 失敗分類とrepair

新規運用では次のclassを使用します。

- `static_validation_failure`: PR CIで発見したdeterministic defect
- `prewrite_identity_or_authorization_failure`: approved target/input/authorizationが現在状態と不一致
- `platform_rejected_no_side_effect`: Kaggleが拒否し、副作用なしを確認済み
- `ambiguous_write`: write結果未確定
- `resource_consumed_runtime_failure`: Kaggle runは開始したがNotebook内部で失敗
- `readout_failure`: compute結果は存在するがoutput/status取得に失敗

`platform_rejected_no_side_effect`はコード修正を必要としません。同じrequestを後で再承認できます。

`ambiguous_write`は再送前にexact reconciliationが必須です。

実装defectまたはNotebook runtime defectでは、failing step、write有無、compute開始有無、root causeを記録したうえで修正します。原因の分からないblind repairは行いません。

詳細な既知failureは [`docs/OPERATIONAL_LESSONS.md`](docs/OPERATIONAL_LESSONS.md) に記録します。

## Security boundary

- GitHub-hosted runnerのみ
- `permissions: {}`を既定とする
- Kaggle tokenは`kaggle-readonry` Environment Secretとしてのみ保持する
- GitHub PAT、SSH秘密鍵、deploy key、cloud long-lived keyを登録しない
- protected Kaggle jobをprivate research repositoryのruntime readへ依存させない
- 外部PR/fork/Issue/comment/`pull_request_target`/`workflow_run`からSecret付きjobを起動しない
- 外部Actionは原則不使用。必要なら完全なcommit SHAへ固定して監査する
- dependencyはversion/hashを固定する
- public logへcredential/private contentを出さない

詳細は [`SECURITY.md`](SECURITY.md) と [`THREAT_MODEL.md`](THREAT_MODEL.md) を参照してください。

## 標準実行フロー

```text
0. OPERATIONAL_LESSONS / EXECUTION_POLICY_V2 を確認
1. Competition・操作・科学条件を確定
2. request/payload/workflowを作成
3. SecretなしPR CIでstatic correctnessを完了
4. PRをreviewしてmainへmerge
5. Environmentでその1回を人間承認
6. protected jobで最小限のlive identity/authorizationを確認
7. Kaggleへ1回だけwrite/run-startを送る
8. side effectをread-only reconcile
9. runner-local dataをcleanup
```

## 運用文書

- [Execution Policy v2](docs/EXECUTION_POLICY_V2.md)
- [Operations](docs/OPERATIONS.md)
- [Operational lessons](docs/OPERATIONAL_LESSONS.md)
- [Bootstrap result](docs/BOOTSTRAP_RESULT.md)
- [Incident response](docs/INCIDENT_RESPONSE.md)
- [Security policy](SECURITY.md)
- [Threat model](THREAT_MODEL.md)

## Upstream documentation

- [NVIDIA/nvidia-kaggle](https://github.com/NVIDIA/nvidia-kaggle)
- [Kaggle公式CLI](https://github.com/Kaggle/kaggle-cli)
- [Kaggle Public API documentation](https://www.kaggle.com/docs/api)

upstream documentationは「どう操作するか」を、このrepositoryは「どのapproved operationを安全に送るか」を定義します。

## 非目標

このrepositoryは、一般用途CI、任意コード実行、Kaggle dataの保管、private repositoryへの広いアクセス、ローカルPCの遠隔操作、複数account管理、resource limitの回避を目的としません。また、Kaggleのremote capacity schedulerをbridge側で再実装することも目的としません。
