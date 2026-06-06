# voice-classifier — 使用手冊（繁體中文）

一個命令列工具，可將含有客戶聲音（客服工單、維修紀錄等）的 CSV 進行分群，
並使用 LLM 為每個群集生成標籤與摘要，輸出人類可讀與機器可讀的報告。

---

## 1. 功能概覽

給定一份每列含有客戶自由文本的 CSV：

1. 以 Azure OpenAI embedding 模型對每一列（去除重複後）進行向量化。
2. 對 KMeans / HDBSCAN / Leiden 候選設定進行掃描，以 cosine silhouette 選出最佳。
3. 擷取離每個群集中心最近的代表列作為原始範例。
4. 請 LLM 依照事先推斷的「資料集脈絡」為每個群集生成簡短標籤與描述性摘要。
5. 透過 LLM 讓相同標籤的較小群集再取得差異化標籤。
6. 輸出 Markdown、HTML 與 CSV 報告。

---

## 2. 安裝

```bash
# 需要 Python 3.10 以上
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # 編輯 .env 填入 AZURE_OPENAI_* 各項值
```

選用後端（未安裝時會自動略過對應的掃描）：

- `hdbscan` — 高速密度式分群。
- `hnswlib` + `python-igraph` + `leidenalg` — 圖論式 Leiden 分群。

---

## 3. 第一次執行

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

執行流程：

1. 讀取 CSV，執行 NFKC 正規化、空白收斂、去重。
2. 為每個唯一列取得（或快取）embedding。
3. 搜尋最佳分群設定。
4. 擷取離每個中心最近的 5 列。
5. 若未指定 `--no-name-clusters`，自動推斷資料集脈絡並行生成標籤，最後解消重複。
6. 結果寫入 `data/output/YYYYMMDD_HHMMSS/`。

---

## 4. 欄位選擇

### 單一欄位

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### 多欄位（以 `label: value` 格式串接）

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### 互動式選擇

若不指定兩個參數，CLI 會列出評分後的候選欄位讓您以編號選擇：

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. 輸出結構

每次執行都會建立帶時間戳的目錄：

```
data/output/20260416_012345/
├── report.md                           人類閱讀的分群結果
├── report.html                          上述的 HTML 版（--format html/both 時）
├── parameter_search.html                完整搜尋報告（頂端含圖表）
├── clusters.csv                         每列一個群集：id、name、size、summary、rep_1..N
├── <input>_classified.csv               原始資料 + cluster_id（+ cluster_name）
├── params.json                          機器可讀的中繼資料
└── run.log                              INFO 以上的執行日誌
```

### `clusters.csv` 欄位

- `cluster_id` — 整數，`-1` 為雜訊。
- `cluster_name` — LLM 標籤（僅 `--name-clusters` 啟用時）。
- `size` — 群集內列數。
- `summary` — LLM 摘要（同上）。
- `rep_1` ... `rep_N` — 離中心最近的原始列。

---

## 6. 主要選項

| 選項 | 預設 | 用途 |
|---|---|---|
| `--input PATH` | 必填 | 輸入 CSV 路徑 |
| `--text-col NAME` | — | 單一欄位模式 |
| `--text-cols A,B` | — | 多欄位串接模式 |
| `--column-labels A=x,B=y` | — | 多欄位標籤 |
| `--output-dir PATH` | `data/output` | 輸出根目錄 |
| `--cache-dir PATH` | `cache` | 快取目錄 |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure OpenAI embedding 部署 |
| `--top-k N` | `5` | 每群集代表列的數量 |
| `--min-clusters N` | `2` | K 下界 |
| `--max-clusters N` | `20` | K 上界 |
| `--target faq|chatbot|insight` | `faq` | 依用途的粒度. `faq`=30-80 分群（FAQ 頁面）, `chatbot`=50-150 (意圖), `insight`=最大化 silhouette |
| `--name-clusters` / `--no-name-clusters` | 開 | LLM 標籤開關 |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | 用於標籤的 Azure OpenAI chat 部署 |
| `--advise` / `--no-advise` | on | 在 `parameter_search.html` 頂端插入 LLM 建議說明 |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Azure OpenAI chat 部署，建議說明使用的聊天模型（對整次執行進行分析） |
| `--format md|html|both` | `md` | `report.*` 格式 |
| `--log-level LEVEL` | `INFO` | stderr 日誌等級 |

---

## 7. 設定

### 環境變數（`.env`）

- `AZURE_OPENAI_API_KEY`（必填）
- `AZURE_OPENAI_ENDPOINT`（必填，例如 `https://<resource>.openai.azure.com`）
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`（必填）
- `AZURE_OPENAI_NAMER_DEPLOYMENT`（必填）
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT`（必填）
- `AZURE_OPENAI_API_VERSION`（選填，預設 `2024-10-21`）
- `AZURE_OPENAI_REQUEST_TIMEOUT`（秒，預設 60）

### 快取

`cache/` 以內容雜湊保存 embedding 向量與 LLM 生成的標籤/摘要，更換模型時會
使用不同的快取檔案。若需重新生成，可刪除對應的
`cache/embeddings_*.pkl` 或 `cache/cluster_annotations_*.pkl`。

---

## 8. 疑難排解

| 症狀 | 解決方案 |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | 在 `.env` 填入或以環境變數設定。 |
| `Column '...' not found` | CLI 會列出可用欄位；請從中選擇。 |
| `Column count mismatch on lines: ...` | 可能是 CSV 有未關閉的引號或欄位內未引號的逗號。 |
| 所有候選都被雜訊比例過濾掉 | 系統會自動放寬並發出警告。 |
| 分數為 `poor`（< 0.20） | 嘗試提供更豐富的文字，或手動檢查 raw 代表列。 |
| Windows 下安裝 `hdbscan` 失敗 | `pip install hdbscan --only-binary=:all:` |
| Leiden 被跳過 | `pip install hnswlib python-igraph leidenalg` |

---

## 9. 隱私須知

- 輸入 CSV 可能含有個資。`data/input/` 與 `data/output/` 已列入 `.gitignore`。
- 本工具會將文字送至 Azure OpenAI Embeddings 及（選用）Chat Completions。
  敏感資料請先在本機進行遮罩。
- `cache/` 儲存 embedding 與 LLM 生成的標籤/摘要，請以與原始 CSV
  相同的標準保護。
