# Architecture

## モジュール構成とデータフロー

```
┌──────────┐   DataFrame     ┌──────────┐   ndarray(N,D)    ┌─────────┐
│ loader   │ ─────────────► │ embedder │ ─────────────────► │ tuner   │
└──────────┘                 └──────────┘                    └────┬────┘
                                                                   │ BestConfig
                                                                   ▼
                              ┌──────────┐     labels / reps   ┌───────────┐
                              │ reporter │ ◄─────────────────── │ clusterer │
                              └────┬─────┘                      └───────────┘
                                   │
                                   ▼
                      data/output/YYYYMMDD_HHMMSS/
                      ├─ report.md
                      ├─ clusters.csv
                      └─ params.json
```

`pipeline.py` が唯一のエントリポイントで、各モジュールを順に呼び出す。

## モジュール責務

### `loader.py`
- CSVを読み込み、指定テキスト列を抽出
- 文字正規化（NFKC）、空白圧縮、空行除去
- 返り値: `pandas.DataFrame`（元の全列 + `_normalized_text` 列）

### `embedder.py`
- `text-embedding-3-small` をデフォルトに OpenAI API を呼び出し
- **キャッシュ必須**: テキストのSHA-256ハッシュをキーに `cache/embeddings_<model>.pkl` に保存
- バッチ取得（最大 `BATCH_SIZE` 件）
- 返り値: `np.ndarray` shape=(N, D)

### `tuner.py`
- 候補手法と候補パラメータを列挙
    - K-Means: `k ∈ [min_k, max_k]`
    - DBSCAN: `eps` をサンプリング、`min_samples = max(3, ⌈ln N⌉)`
    - HDBSCAN: `min_cluster_size` をサンプリング
- 各候補でクラスタリングし、シルエットスコア（cosine）を評価
- スコア最大の設定を `BestConfig(algorithm, params, score, labels)` として返却
- 詳細は [`algorithm.md`](algorithm.md) を参照

### `clusterer.py`
- `BestConfig` を受け取り、代表テキストを抽出
- 各クラスタについて、重心に最も近い上位N件（コサイン距離）を返す
- ノイズラベル（-1）は除外してまとめて扱う

### `reporter.py`
- `report.md` ... クラスタ数、スコア、件数、代表テキストを整形
- `clusters.csv` ... 元CSVに `cluster_id` 列を付与して保存
- `params.json` ... アルゴリズム名・選択パラメータ・スコアをJSON化

### `pipeline.py`
- argparse でCLI引数を受け取る
- タイムスタンプ付き出力ディレクトリを作成
- `loader → embedder → tuner → clusterer → reporter` を直列実行
- ログは `logging` モジュールで stderr + `output_dir/run.log` に出す

## 型とインタフェース

```python
# tuner.py
@dataclass
class BestConfig:
    algorithm: Literal["kmeans", "dbscan", "hdbscan"]
    params: dict[str, Any]
    silhouette: float
    labels: np.ndarray  # shape (N,)
    n_clusters: int
    n_noise: int

# clusterer.py
@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    representative_indices: list[int]  # len = top_k
    representative_texts: list[str]
```

## I/O 規約

- **I/Oを直接扱うのは `pipeline.py` のみ**（CSV読込は `loader.py` が担うが、入出力パスは `pipeline.py` から渡す）
- 他モジュールは純粋関数として設計し、テスト容易性を確保

## 失敗モード

| 事象 | 扱い |
|---|---|
| APIキー未設定 | 起動直後に `RuntimeError` |
| OpenAI API 失敗（レートリミット等） | 指数バックオフ + 最大N回リトライ、超えたら `raise` |
| 有効サンプル数 < `min_clusters` | チューニング前に `ValueError` で中断 |
| 全クラスタがノイズ | `clusterer` がノイズのみのレポートを出し、スコアは `nan` 記録 |
