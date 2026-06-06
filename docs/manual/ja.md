# voice-classifier — ユーザマニュアル（日本語）

自由記述のテキストを含むCSV（問い合わせ、修理受付等）をクラスタリングし、
LLM で各クラスタにラベルと要約を付与してレポート出力する CLI ツールです。

---

## 1. できること

1. 各行の正規化テキストを Azure OpenAI Embeddings でベクトル化
2. KMeans / HDBSCAN / Leiden を自動スイープし、コサインシルエットで最適解を選択
3. 各クラスタの重心付近の代表行を抽出
4. LLM にデータセット全体の意味を推定させ、それをグラウンディングに各クラスタの
   短いラベルと要約を生成
5. ラベル重複を LLM で解消
6. Markdown / HTML / CSV で成果物を出力

---

## 2. セットアップ

```bash
# Python 3.10 以降が必要
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # AZURE_OPENAI_* の値を記入
```

任意のクラスタリングバックエンド（未インストールでも自動スキップ）:

- `hdbscan` — 密度ベースの高速クラスタリング
- `hnswlib` + `python-igraph` + `leidenalg` — グラフベースの Leiden

---

## 3. 最初の実行

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "対応内容"
```

実行内容:

1. CSV 読込・NFKC 正規化・重複集約
2. ユニーク行ごとに埋め込み取得（キャッシュ優先）
3. クラスタリング候補スイープ
4. 各クラスタから重心近傍 5 件抽出
5. `--no-name-clusters` 未指定なら、データセット意味推定 → 並列ラベル付け
   → 重複解消まで自動実行
6. `data/output/YYYYMMDD_HHMMSS/` に成果物を書き出し

---

## 4. 対象列の指定

### 単一列

```bash
python src/pipeline.py --input tickets.csv --text-col "対応内容"
```

### 複数列（`label: value` 形式で結合）

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "件名,本文" \
    --column-labels "件名=subject,本文=body"
```

### 対話モード

両フラグを省くと候補を点数付きで提示し、番号で選択できます。

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. 出力ファイル

実行ごとにタイムスタンプ付きディレクトリを生成します:

```
data/output/20260416_012345/
├── report.md                           クラスタリング結果（人間用）
├── report.html                          同上の HTML 版（--format html/both 時）
├── parameter_search.html                パラメータ探索の全貌（上部にグラフ）
├── clusters.csv                         クラスタ1件=1行のサマリ
├── <入力ファイル名>_classified.csv       元データ + cluster_id (+ cluster_name)
├── params.json                          機械可読なメタ情報
└── run.log                              実行ログ（INFO 以上）
```

### `clusters.csv` の列

- `cluster_id` — 整数（ノイズは `-1`）
- `cluster_name` — LLM ラベル（`--name-clusters` 時のみ）
- `size` — クラスタ内の行数
- `summary` — LLM 要約（同条件）
- `rep_1` ... `rep_N` — 重心付近の生データ

### `<入力名>_classified.csv`

元データに `cluster_id`（+ `cluster_name`）を付加したもの。列順は元データを保持。

---

## 6. 主要オプション

| オプション | 既定 | 役割 |
|---|---|---|
| `--input PATH` | 必須 | 入力 CSV |
| `--text-col NAME` | — | 単一列モード |
| `--text-cols A,B` | — | 複数列結合モード |
| `--column-labels A=x,B=y` | — | 複数列時のラベル変換 |
| `--output-dir PATH` | `data/output` | 出力ディレクトリ |
| `--cache-dir PATH` | `cache` | キャッシュ保存先 |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure OpenAI 埋め込みデプロイメント |
| `--top-k N` | `5` | `rep_*` に抽出する件数 |
| `--min-clusters N` | `2` | K の下限 |
| `--max-clusters N` | `20` | K の上限 |
| `--target faq|chatbot|insight` | `faq` | 目的別の粒度最適化. `faq`=30〜80クラスタ（FAQページ向き）、`chatbot`=50〜150（意図分岐向き）、`insight`=シルエット最大（探索向き） |
| `--name-clusters` / `--no-name-clusters` | ON | LLM ラベル生成の ON/OFF |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | ラベル生成用の Azure OpenAI チャットデプロイメント |
| `--advise` / `--no-advise` | on | `parameter_search.html` 冒頭に LLM 助言ノートを挿入 |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | 助言ノート用の Azure OpenAI チャットデプロイメント（実行全体を読み解くので強めのモデル） |
| `--format md|html|both` | `md` | `report.*` の形式 |
| `--log-level LEVEL` | `INFO` | stderr のログレベル |

---

## 7. 設定

### 環境変数（`.env`）

- `AZURE_OPENAI_API_KEY`（必須）
- `AZURE_OPENAI_ENDPOINT`（必須、例 `https://<resource>.openai.azure.com`）
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`（必須）
- `AZURE_OPENAI_NAMER_DEPLOYMENT`（必須）
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT`（必須）
- `AZURE_OPENAI_API_VERSION`（任意、既定 `2024-10-21`）
- `AZURE_OPENAI_REQUEST_TIMEOUT`（秒、既定 60）

### キャッシュ

`cache/` には埋め込みベクトルと LLM アノテーションをコンテンツハッシュで保存。
モデルを切り替えると別キャッシュファイルで管理されます。再生成したい場合は
該当する `cache/embeddings_*.pkl` や `cache/cluster_annotations_*.pkl` を削除。

---

## 8. トラブルシューティング

| 症状 | 対処 |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | `.env` に記入、または環境変数に設定 |
| `Column '...' not found` | 表示される利用可能列から正しいものを選択 |
| `Column count mismatch on lines: ...` | CSV の引用符閉じ忘れ・未クオート値にカンマを疑う |
| すべての候補がノイズ率フィルタで除外 | データ構造が希薄。自動でフィルタが緩和される |
| スコアが `poor` (< 0.20) | 入力テキストが短すぎる可能性。列選択を見直すか `--no-name-clusters` で生データを直接確認 |
| `hdbscan` の Windows インストール失敗 | `pip install hdbscan --only-binary=:all:` |
| Leiden がスキップされる | `pip install hnswlib python-igraph leidenalg` |

---

## 9. プライバシー上の注意

- 入力 CSV は PII を含み得ます。`data/input/` / `data/output/` は `.gitignore` 済み。
- 処理中に Azure OpenAI Embeddings と（任意で）Chat Completions へテキストが送信されます。
  センシティブなデータは事前にマスキング推奨。
- `cache/` には埋め込みベクトルと LLM 生成のラベル・要約が保存されます。
  元 CSV と同レベルの取り扱いをしてください。
