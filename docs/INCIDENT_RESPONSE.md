# Incident Response

## Incident indicators

次のいずれかを確認したらincidentとして扱います。

- 予定していないworkflow runまたはactor
- Secret付きjobが外部PR、Issue、コメント等から起動した
- 未承認のKaggle submission、Notebook/Dataset/Model変更
- token、Authorization header、cookieのlog/artifact露出
- 不明な外向き通信、package、Action、binary download
- workflowの`permissions`、trigger、runner label、Environmentが予期せず変更された
- GitHub、Kaggleから未知の認証・操作通知を受信した

## Immediate containment

以下を上から順に実施します。

1. Repositoryの`Settings > Actions > General`でActionsを無効化する。
2. 実行中・queued状態のworkflowをcancelする。
3. Kaggle Settingsで対象API tokenを失効させる。
4. GitHub Environmentから`KAGGLE_API_TOKEN`を削除する。
5. Environmentのdeployment approvalを停止する。
6. 不審な変更を元に戻す前に、commit SHA、run ID、actor、時刻を記録する。

Kaggle token以外の資格情報を誤って登録していた場合は、その発行元でも直ちに失効させます。

## Scope assessment

次を確認します。

- 侵害されたworkflowとcommit SHA
- trigger event、actor ID、repository ID、ref
- Secretを参照したjobとstep
- 使用されたAction、package、download URL、そのversion/hash
- 外向き接続先
- log、cache、artifact、release、commitに機密情報が残っていないか
- Kaggle側のNotebook、Dataset、Model、Submission、API token履歴
- `GITHUB_TOKEN`以外のGitHub credentialが存在したか
- self-hosted runnerが一度でも使用されたか

このrepositoryにKaggle tokenしかなく、GitHub-hosted runnerと`permissions: {}`だけを使用していた場合、他のprivate repositoryやローカルPCへの直接経路は原則ありません。ただし、設定と実際のrunを確認して判断します。

## Evidence handling

- logやartifactを公開場所へ複製しない。
- token値をincident noteへ記録しない。
- run URL、run ID、commit SHA、UTC/JST時刻、HTTP destination、package hashを記録する。
- 悪意ある可能性のあるartifactやscriptをローカルPCで実行しない。

## Eradication

1. 悪意または不明なworkflow、Action、dependencyを削除する。
2. unsafe triggerと過剰permissionを除去する。
3. dependencyを既知の安全なhashへ固定するか、標準ライブラリ実装へ戻す。
4. cacheを使用していた場合は全cacheを削除する。
5. 公開log/artifactにSecretが残った場合、削除可否にかかわらずtokenを恒久的に無効とみなす。

## Recovery

1. Secretなし診断workflowだけを復旧する。
2. repository ID、actor ID、runner environment、network destinationを再確認する。
3. 新しいKaggle tokenを発行する。
4. Environment Secretへ登録し、reviewer approvalを復旧する。
5. read-only操作を1回だけ実行して監視する。
6. write操作は別途review後まで無効化する。

## Post-incident actions

- root causeとattack pathを`THREAT_MODEL.md`へ反映する。
- 再発防止をworkflow外のrepository settings/rulesetにも追加する。
- upstream Action、library、OSSに不具合がある場合、機密情報を除いた再現手順と影響をmaintainerへ報告する。
- Kaggleの提出枠や公開資産に影響した場合は、必要に応じてKaggle Supportへ連絡する。
