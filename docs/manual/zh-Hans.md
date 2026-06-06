# voice-classifier — 使用手册（简体中文）

一个命令行工具，可将含有客户声音（客服工单、维修记录等）的 CSV 进行聚类，
并使用 LLM 为每个聚类生成标签与摘要，输出人类可读与机器可读的报告。

---

## 1. 功能概览

给定一份每行含有客户自由文本的 CSV：

1. 使用 Azure OpenAI embedding 模型对每一行（去重后）进行向量化。
2. 扫描 KMeans / HDBSCAN / Leiden 候选配置，通过 cosine silhouette 选出最佳。
3. 提取距每个聚类中心最近的代表行作为原始样本。
4. 让 LLM 按照事先推断的"数据集上下文"为每个聚类生成简短标签与描述性摘要。
5. 通过 LLM 让相同标签的较小聚类重新获得差异化标签。
6. 输出 Markdown、HTML 与 CSV 报告。

---

## 2. 安装

```bash
# 需要 Python 3.10 及以上
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # 编辑 .env 填入 AZURE_OPENAI_* 各项值
```

可选后端（未安装时会自动跳过相应扫描）：

- `hdbscan` — 高速密度聚类。
- `hnswlib` + `python-igraph` + `leidenalg` — 基于图的 Leiden 聚类。

---

## 3. 第一次运行

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

执行流程：

1. 读取 CSV，执行 NFKC 正规化、空白压缩、去重。
2. 为每个唯一行获取（或使用缓存的）embedding。
3. 搜索最佳聚类配置。
4. 提取距每个中心最近的 5 行。
5. 若未指定 `--no-name-clusters`，自动推断数据集上下文并行生成标签，最后消除重复。
6. 结果写入 `data/output/YYYYMMDD_HHMMSS/`。

---

## 4. 列选择

### 单列

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### 多列（以 `label: value` 格式拼接）

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### 交互式选择

若不指定上述两个参数，CLI 会列出带评分的候选列供编号选择：

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. 输出结构

每次运行都会创建带时间戳的目录：

```
data/output/20260416_012345/
├── report.md                           人类阅读的聚类结果
├── report.html                          同上的 HTML 版（--format html/both 时）
├── parameter_search.html                完整搜索报告（顶部含图表）
├── clusters.csv                         每行一个聚类：id、name、size、summary、rep_1..N
├── <input>_classified.csv               原始数据 + cluster_id（+ cluster_name）
├── params.json                          机器可读的元数据
└── run.log                              INFO 及以上的执行日志
```

### `clusters.csv` 列说明

- `cluster_id` — 整数，`-1` 为噪声。
- `cluster_name` — LLM 标签（仅在 `--name-clusters` 启用时）。
- `size` — 聚类内行数。
- `summary` — LLM 摘要（同上）。
- `rep_1` ... `rep_N` — 距中心最近的原始行。

---

## 6. 主要选项

| 选项 | 默认值 | 用途 |
|---|---|---|
| `--input PATH` | 必填 | 输入 CSV 路径 |
| `--text-col NAME` | — | 单列模式 |
| `--text-cols A,B` | — | 多列拼接模式 |
| `--column-labels A=x,B=y` | — | 多列前缀标签 |
| `--output-dir PATH` | `data/output` | 输出根目录 |
| `--cache-dir PATH` | `cache` | 缓存目录 |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure OpenAI embedding 部署 |
| `--top-k N` | `5` | 每聚类代表行数量 |
| `--min-clusters N` | `2` | K 下界 |
| `--max-clusters N` | `20` | K 上界 |
| `--target faq|chatbot|insight` | `faq` | 按用途的粒度. `faq`=30-80 聚类（FAQ 页面）, `chatbot`=50-150 (意图), `insight`=最大化 silhouette |
| `--name-clusters` / `--no-name-clusters` | 开 | LLM 标签生成开关 |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | 用于标签生成的 Azure OpenAI chat 部署 |
| `--advise` / `--no-advise` | on | 在 `parameter_search.html` 顶部插入 LLM 助言说明 |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Azure OpenAI chat 部署，助言说明使用的聊天模型（对整次运行进行分析） |
| `--format md|html|both` | `md` | `report.*` 格式 |
| `--log-level LEVEL` | `INFO` | stderr 日志级别 |

---

## 7. 配置

### 环境变量（`.env`）

- `AZURE_OPENAI_API_KEY`（必填）
- `AZURE_OPENAI_ENDPOINT`（必填，例如 `https://<resource>.openai.azure.com`）
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`（必填）
- `AZURE_OPENAI_NAMER_DEPLOYMENT`（必填）
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT`（必填）
- `AZURE_OPENAI_API_VERSION`（可选，默认 `2024-10-21`）
- `AZURE_OPENAI_REQUEST_TIMEOUT`（秒，默认 60）

### 缓存

`cache/` 以内容哈希保存 embedding 向量与 LLM 生成的标签/摘要，更换模型时会
使用不同的缓存文件。若需强制重新生成，可删除对应的
`cache/embeddings_*.pkl` 或 `cache/cluster_annotations_*.pkl`。

---

## 8. 故障排查

| 症状 | 解决方案 |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | 在 `.env` 填入或设置环境变量。 |
| `Column '...' not found` | CLI 会列出可用列；请从中选择。 |
| `Column count mismatch on lines: ...` | CSV 引号未闭合或字段中含有未引号的逗号。 |
| 所有候选都被噪声比例过滤 | 过滤器会自动放宽并给出警告。 |
| 分数为 `poor`（< 0.20） | 尝试更丰富的文本，或手动检查原始代表行。 |
| Windows 下 `hdbscan` 安装失败 | `pip install hdbscan --only-binary=:all:` |
| Leiden 被跳过 | `pip install hnswlib python-igraph leidenalg` |

---

## 9. 隐私须知

- 输入 CSV 可能包含个人信息。`data/input/` 与 `data/output/` 已列入 `.gitignore`。
- 工具会将文字发送至 Azure OpenAI Embeddings 及（可选）Chat Completions。
  敏感数据请先在本地进行脱敏处理。
- `cache/` 保存 embedding 与 LLM 生成的标签/摘要，请按照原始 CSV
  的标准进行保护。
