# azure-setup-manual のスクリーンショット置き場

`docs/azure-setup-manual.md` から参照される画像をここに置きます。
下表のファイル名で保存すると、マニュアル内の該当位置に表示されます。
（PNG 推奨。機微情報のマスキングはお好みで。）

| ファイル名 | 対応する画面 |
|---|---|
| `01-portal-home.png` | Azure ポータル ホーム（サインイン済み） |
| `02-subscriptions-list.png` | サブスクリプション一覧（「+ 追加」がある画面） |
| `03-create-subscription-basics.png` | サブスクリプションの作成 - 基本情報（名前・課金プロファイル入力後） |
| `04-create-subscription-review.png` | サブスクリプションの作成 - レビューと作成（検証合格） |
| `05-subscription-created.png` | 「サブスクリプションの作成完了」通知 |
| `06-create-resource.png` | 「リソースの作成」画面 |
| `07-marketplace-search.png` | Marketplace で "azure openai service" を検索した一覧 |
| `08-create-openai-basics.png` | Azure OpenAI の作成 - 基本情報（サブスク/RG/リージョン/名前/価格レベル入力後） |
| `09-network-tab.png` | 作成ウィザードのネットワーク タブ（既定: すべてのネットワーク） |
| `10-create-openai-review.png` | Azure OpenAI の作成 - レビューおよび送信（検証合格・作成ボタン有効） |
| `11-deployment-complete.png` | 「デプロイが完了しました」画面（リソースに移動ボタン） |
| `12-resource-overview.png` | リソース `vc20260607` の概要（状態アクティブ／エンドポイント・キー導線） |
| `13-foundry-overview.png` | Foundry ポータル 概要（エンドポイントとキー） |
| `14-deployments-empty.png` | Foundry「モデル デプロイ」一覧（空・+ モデルのデプロイ） |
| `15-select-model.png` | 「モデルを選択してください」ダイアログ |
| `16-embedding-standard.png` | 埋め込みのデプロイ設定（種類を Standard に変更→AI リソースが vc20260607） |
| `17-gpt4o-standard.png` | gpt-4o のデプロイ設定（Standard / Japan East / 50K TPM / vc20260607） |
| `18-gpt4o-deployed.png` | gpt-4o デプロイ完了（ターゲット URI・キー） |

## 補足

- マニュアル本体: `../../azure-setup-manual.md`
- 画像が未配置でもマニュアルのテキストだけで手順は追えます（画像はリンク切れ表示になるだけ）。
- 追加で残したい画面（例: クォータ不足の警告、デプロイ種類のドロップダウン等）があれば
  任意のファイル名で置き、マニュアルに `![説明](images/azure-setup/ファイル名.png)` を足してください。
