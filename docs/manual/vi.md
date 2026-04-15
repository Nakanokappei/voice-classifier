# voice-classifier — Hướng dẫn sử dụng (Tiếng Việt)

Công cụ dòng lệnh giúp phân cụm các file CSV chứa tiếng nói khách hàng
(ticket hỗ trợ, biên bản sửa chữa, v.v.), gắn nhãn mỗi cụm bằng LLM và
xuất báo cáo ở định dạng dễ đọc cho người lẫn máy.

---

## 1. Công cụ làm gì

Với file CSV mà mỗi dòng chứa văn bản tự do của khách hàng:

1. Nhúng (embedding) từng dòng duy nhất bằng mô hình của OpenAI.
2. Quét các cấu hình ứng viên (KMeans / HDBSCAN / Leiden) và chọn cấu hình
   tốt nhất theo silhouette cosine.
3. Lấy những dòng gần centroid của mỗi cụm nhất làm ứng viên thô.
4. Yêu cầu LLM tạo nhãn ngắn và bản tóm tắt cho từng cụm, dựa trên ngữ cảnh
   tập dữ liệu đã được suy ra trước.
5. Khử trùng nhãn bằng cách tái sinh nhãn cho cụm nhỏ hơn.
6. Ghi báo cáo ở dạng Markdown, HTML và CSV.

---

## 2. Cài đặt

```bash
# Cần Python 3.10 hoặc mới hơn.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # sau đó mở .env và điền OPENAI_API_KEY
```

Các backend tuỳ chọn (sẽ bị bỏ qua an toàn nếu thiếu):

- `hdbscan` — phân cụm theo mật độ nhanh.
- `hnswlib` + `python-igraph` + `leidenalg` — phân cụm Leiden dựa trên đồ thị.

---

## 3. Lần chạy đầu

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

Các bước:

1. Đọc CSV, chuẩn hoá văn bản (NFKC, nén khoảng trắng, khử trùng lặp).
2. Lấy / cache embeddings cho từng dòng duy nhất.
3. Tìm cấu hình phân cụm tốt nhất.
4. Trích 5 dòng gần mỗi centroid nhất.
5. Trừ khi có `--no-name-clusters`, suy ra ngữ cảnh, sinh nhãn song song và
   xử lý nhãn trùng.
6. Ghi kết quả vào `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Chọn cột

### Cột đơn

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Nhiều cột (nối thành `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Chọn tương tác

Nếu bỏ cả hai tham số, CLI sẽ hiển thị danh sách ứng viên đã chấm điểm và
yêu cầu chọn:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Cấu trúc đầu ra

Mỗi lần chạy tạo một thư mục có dấu thời gian:

```
data/output/20260416_012345/
├── report.md                           Kết quả phân cụm cho người đọc
├── report.html                          Tương tự dưới dạng HTML (với --format html/both)
├── parameter_search.html                Báo cáo đầy đủ kèm đồ thị trên cùng
├── clusters.csv                         Mỗi hàng là một cụm: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               Dữ liệu gốc + cluster_id (+ cluster_name)
├── params.json                          Siêu dữ liệu dạng máy đọc
└── run.log                              Log cấp INFO của lần chạy
```

### Các cột của `clusters.csv`

- `cluster_id` — số nguyên, `-1` cho nhiễu.
- `cluster_name` — nhãn ngắn từ LLM (chỉ khi bật `--name-clusters`).
- `size` — số dòng.
- `summary` — tóm tắt từ LLM (cùng điều kiện).
- `rep_1` ... `rep_N` — dòng gốc gần centroid.

---

## 6. Các tuỳ chọn chính

| Tuỳ chọn | Mặc định | Công dụng |
|---|---|---|
| `--input PATH` | bắt buộc | Đường dẫn CSV đầu vào |
| `--text-col NAME` | — | Cột đơn cho embedding |
| `--text-cols A,B` | — | Nhiều cột nối lại |
| `--column-labels A=x,B=y` | — | Nhãn cho chế độ nhiều cột |
| `--output-dir PATH` | `data/output` | Thư mục gốc đầu ra |
| `--cache-dir PATH` | `cache` | Thư mục cache |
| `--model NAME` | `text-embedding-3-small` | Mô hình embedding |
| `--top-k N` | `5` | Số đại diện mỗi cụm |
| `--min-clusters N` | `2` | Biên dưới của K |
| `--max-clusters N` | `20` | Biên trên của K |
| `--name-clusters` / `--no-name-clusters` | bật | Bật/tắt gắn nhãn LLM |
| `--name-model NAME` | `gpt-5.4-nano` | Mô hình chat để gắn nhãn |
| `--format md|html|both` | `md` | Định dạng `report.*` |
| `--log-level LEVEL` | `INFO` | Mức độ chi tiết stderr |

---

## 7. Cấu hình

### Biến môi trường (`.env`)

- `OPENAI_API_KEY` (bắt buộc)
- `OPENAI_EMBEDDING_MODEL` (override tuỳ chọn)
- `OPENAI_REQUEST_TIMEOUT` (giây, mặc định 60)

### Cache

`cache/` lưu vector embedding và chú thích LLM theo hash nội dung. Đổi mô
hình = file cache khác. Muốn tái tạo, hãy xoá `cache/embeddings_*.pkl`
hoặc `cache/cluster_annotations_*.pkl` tương ứng.

---

## 8. Xử lý sự cố

| Triệu chứng | Giải pháp |
|---|---|
| `OPENAI_API_KEY is not set` | Điền khoá vào `.env` hoặc biến môi trường. |
| `Column '...' not found` | CLI liệt kê các cột có sẵn; chọn một trong đó. |
| `Column count mismatch on lines: ...` | Dấu ngoặc kép chưa đóng hoặc có dấu phẩy trong giá trị không trích dẫn. |
| Tất cả ứng viên bị lọc do tỷ lệ nhiễu | Bộ lọc tự động nới lỏng kèm cảnh báo. |
| Điểm `poor` (< 0.20) | Thử văn bản phong phú hơn hoặc kiểm tra thủ công các đại diện. |
| `hdbscan` không cài được trên Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden bị bỏ qua | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Lưu ý quyền riêng tư

- CSV đầu vào có thể chứa thông tin cá nhân. `data/input/` và `data/output/`
  đều nằm trong `.gitignore`.
- Pipeline gửi văn bản tới OpenAI Embeddings và (tuỳ chọn) Chat Completions.
  Hãy che dữ liệu nhạy cảm trên máy cục bộ trước khi xử lý.
- Thư mục `cache/` lưu embedding và nhãn/tóm tắt do LLM sinh ra. Hãy bảo vệ
  nó ở mức tương đương với CSV gốc.
