# voice-classifier
**顧客の声 自動分類・洞察レポートシステム**

カスタマーサポート対応履歴CSVをOpenAI Embeddings で数値化し、
最適なクラスタリング手法・パラメータを自動選択、クラスター代表テキストと
サマリーレポートを出力する分析パイプライン。

---

## Architecture

```
voice-classifier/
├── CLAUDE.md               ← このファイル
├── README.md
├── requirements.txt
├── .env.example            ← OPENAI_API_KEY のテンプレート（.env は gitignore）
│
├── data/
│   ├── input/              ← 顧客から受領したCSVを置く（gitignore）
│   └── output/             ← レポート・クラスタリング結果の出力先（gitignore）
│
├── src/
│   ├── pipeline.py         ← エントリポイント。全ステップを順に呼び出す
│   ├── loader.py           ← CSV読み込み・前処理・テキスト列の抽出
│   ├── embedder.py         ← OpenAI Embeddings API呼び出し・キャッシュ管理
│   ├── tuner.py            ← パラメータサンプリングと最適手法の自動選択
│   ├── clusterer.py        ← クラスタリング実行・代表テキスト取得
│   └── reporter.py         ← レポート（Markdown + CSV）の生成・出力
│
├── cache/                  ← 埋め込みベクトルのローカルキャッシュ（gitignore）
│
├── tests/
│   └── test_*.py
│
└── docs/
    ├── architecture.md     ← 設計詳細・モジュール間データフロー
    ├── algorithm.md        ← チューニング戦略・スコア評価基準
    └── data-format.md      ← 入力CSVのフォーマット仕様
```

---

## Tech Stack

| 用途 | ライブラリ |
|---|---|
| 埋め込み取得 | `openai` (text-embedding-3-small) |
| クラスタリング | `scikit-learn` (KMeans, DBSCAN), `hdbscan` |
| スコア評価 | `scikit-learn` (silhouette_score, davies_bouldin_score) |
| データ処理 | `pandas`, `numpy` |
| 進捗表示 | `tqdm` |
| 環境変数 | `python-dotenv` |

Python バージョン: **3.10+**

---

## Key Constraints

- **実行環境は就業先の業務PC（Windows）**。インストール可能なパッケージはPython標準＋pip経由のみ。
- **APIキーは `.env` で管理**。コード・出力ファイルにキーを埋め込まない。
- **入力CSVはPIIを含む可能性あり**。`data/input/` と `data/output/` は `.gitignore` に含める。
- **埋め込みはキャッシュ必須**。同一テキストに対して再度APIを叩かないよう `cache/` に保存する（`.pkl` or `.npy`）。
- **外部ネットワークはOpenAI APIのみ**。他のクラウドサービスへの通信は行わない。

---

## Pipeline Flow

```
CSV読み込み → テキスト前処理 → Embedding取得（キャッシュ優先）
    → パラメータ候補をサンプリング → シルエットスコアで最適解選択
    → 最適パラメータでクラスタリング実行 → 代表テキスト取得
    → Markdownレポート + クラスタ別CSV 出力
```

詳細は `@docs/architecture.md` を参照。

---

## Coding Standards

- 関数・変数名は **英語スネークケース**
- 型ヒント必須（`def func(x: list[str]) -> pd.DataFrame:`）
- 各モジュールは **単一責務**。`pipeline.py` 以外でI/Oを直接扱わない
- エラーは握りつぶさず `raise` または `logging.error` で明示する
- コメントは **日本語可**（社内利用のため可読性優先）

---

## Output Format

`data/output/YYYYMMDD_HHMMSS/` 以下に出力：

| ファイル | 内容 |
|---|---|
| `report.md` | クラスター数・シルエットスコア・手法・代表テキスト一覧 |
| `clusters.csv` | 全レコードにクラスターIDを付与したCSV |
| `params.json` | 選択されたアルゴリズム名・パラメータ・スコアの記録 |

---

## Quick Start

```bash
cp .env.example .env          # OPENAI_API_KEY を記入
pip install -r requirements.txt
python src/pipeline.py --input data/input/sample.csv --text-col "対応内容"
```

---

## Docs Index

詳細情報は都度 `@` で参照すること（毎セッション自動読み込みしない）：

- `@docs/architecture.md` — モジュール設計・データフロー詳細
- `@docs/algorithm.md` — チューニング戦略・使用アルゴリズムの選択基準
- `@docs/data-format.md` — 入力CSV仕様・前処理ルール
