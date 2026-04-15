# voice-classifier

[![CI](https://github.com/Nakanokappei/voice-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Nakanokappei/voice-classifier/actions/workflows/ci.yml)

顧客の声（カスタマーサポート対応履歴CSV）を自動で分類し、
クラスターごとの代表テキストと洞察レポートを生成する分析パイプライン。

## 特徴

- **埋め込み取得**: OpenAI `text-embedding-3-small` によるテキスト数値化（キャッシュ付き）
- **自動チューニング**: K-Means / DBSCAN / HDBSCAN の候補を走査し、シルエットスコアで最適解を選択
- **代表テキスト**: 各クラスタの重心に最も近い上位N件を抽出
- **レポート出力**: Markdown 要約 + 全レコード付きCSV + 選択パラメータ JSON

## 必要環境

- Python 3.10+
- OpenAI APIキー
- Windows / macOS / Linux

## セットアップ

```bash
# 1. 仮想環境（任意）
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. 依存インストール
pip install -r requirements.txt

# 3. APIキーを設定
cp .env.example .env
# .env を開いて OPENAI_API_KEY を記入
```

> **Note (Windows)**: `hdbscan` はビルドに C コンパイラが必要な場合があります。
> 失敗したら `pip install hdbscan --only-binary=:all:` を試してください。

## 使い方

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "対応内容"
```

### 主要オプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--input` | 必須 | 入力CSVパス |
| `--text-col` | 省略可 | 分類対象テキストの列名. 省略時は候補を対話的に提示 |
| `--text-cols` | 省略可 | カンマ区切り複数列. 各列を `label: value` 形式で結合して埋め込み |
| `--column-labels` | 省略可 | `列名=ラベル` の組をカンマ区切り. 複数列モードでのプレフィックス変更 |
| `--output-dir` | `data/output` | 出力ディレクトリルート |
| `--cache-dir` | `cache` | 埋め込みキャッシュ保存先 |
| `--model` | `text-embedding-3-small` | 埋め込みモデル |
| `--top-k` | `5` | クラスタ代表テキストの抽出件数 |
| `--min-clusters` | `2` | 探索するクラスタ数下限 |
| `--max-clusters` | `20` | 探索するクラスタ数上限 |
| `--name-clusters` | OFF | LLM でクラスタに短いラベルを自動生成（Chat API 追加呼び出し） |
| `--name-model` | `gpt-4o-mini` | クラスタ名生成に使う Chat モデル |
| `--format` | `md` | レポート形式. `md` / `html` / `both` |

### 使用例

```bash
# 単一列
python src/pipeline.py --input tickets.csv --text-col "対応内容"

# 複数列結合（CKPS 互換）
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"

# LLM 自動ネーミング + HTML レポート
python src/pipeline.py --input tickets.csv --text-col "対応内容" \
    --name-clusters --format both
```

## 出力

`data/output/YYYYMMDD_HHMMSS/` 配下に以下を生成:

- `report.md` / `report.html`               — クラスタリング結果（採用設定・クラスタ別代表テキスト）
- `parameter_search.md` / `.html`           — パラメータ探索の全貌（採用/除外理由・全試行ランキング）
- `clusters.csv`                             — 入力CSV + `cluster_id`（`--name-clusters` 時は `cluster_name` も）
- `params.json`                              — 採用アルゴリズム・パラメータ・スコア・探索メタ情報

`--format md`（既定）では Markdown のみ、`--format html` では HTML のみ、`--format both` で両方.

## ドキュメント

- [`docs/architecture.md`](docs/architecture.md) — モジュール設計・データフロー
- [`docs/algorithm.md`](docs/algorithm.md) — チューニング戦略・スコア評価
- [`docs/data-format.md`](docs/data-format.md) — 入力CSV仕様

## テスト

```bash
pytest
```
