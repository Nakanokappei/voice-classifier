# voice-classifier — 사용자 매뉴얼 (한국어)

고객의 목소리를 담은 CSV(지원 티켓, 수리 접수 기록 등)를 클러스터로
분류하고, 각 클러스터에 LLM으로 레이블을 붙인 뒤 사람과 기계 모두가
읽을 수 있는 리포트를 생성하는 CLI 도구입니다.

---

## 1. 무엇을 하는가

각 행에 고객의 자유 텍스트가 들어 있는 CSV를 받아서:

1. OpenAI 임베딩 모델로 고유 행마다 임베딩을 계산합니다.
2. 후보 설정(KMeans / HDBSCAN / Leiden)을 스윕하여 cosine 실루엣 기준으로
   최적 설정을 선택합니다.
3. 각 클러스터의 중심에 가장 가까운 행을 원본 대표로 추출합니다.
4. 사전에 추론한 데이터셋 컨텍스트를 토대로 LLM에 짧은 레이블과 요약을
   요청합니다.
5. 레이블 충돌이 있으면 작은 클러스터의 레이블을 재생성해 차별화합니다.
6. Markdown, HTML, CSV 리포트를 기록합니다.

---

## 2. 설치

```bash
# Python 3.10 이상이 필요합니다.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # .env 를 열어 OPENAI_API_KEY 를 입력합니다
```

선택적 백엔드(없으면 자동으로 건너뜁니다):

- `hdbscan` — 밀도 기반 고속 클러스터링.
- `hnswlib` + `python-igraph` + `leidenalg` — 그래프 기반 Leiden 클러스터링.

---

## 3. 첫 실행

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

도구는:

1. CSV 를 읽고 텍스트를 정규화합니다 (NFKC, 공백 압축, 중복 제거).
2. 고유 행의 임베딩을 가져오거나 캐시합니다.
3. 최적 클러스터링 설정을 탐색합니다.
4. 각 중심점에 가장 가까운 5개 행을 추출합니다.
5. `--no-name-clusters` 가 없으면 데이터셋 컨텍스트를 추론해 레이블을
   병렬 생성하고 중복을 해결합니다.
6. 결과를 `data/output/YYYYMMDD_HHMMSS/` 에 기록합니다.

---

## 4. 컬럼 선택

### 단일 컬럼

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### 복수 컬럼 (`label: value` 형식으로 결합)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### 대화형 선택

두 플래그 모두 생략하면 CLI 가 점수로 정렬된 후보를 보여 주고 선택을
요청합니다:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. 출력 구조

각 실행은 타임스탬프 디렉터리를 만듭니다:

```
data/output/20260416_012345/
├── report.md                           사람이 읽는 클러스터링 결과
├── report.html                          동일 내용의 HTML (--format html/both)
├── parameter_search.html                파라미터 탐색 전체 리포트 + 상단 차트
├── clusters.csv                         클러스터 1건당 1행: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               원본 행 + cluster_id (+ cluster_name)
├── params.json                          기계 판독용 메타데이터
└── run.log                              실행 INFO 로그
```

### `clusters.csv` 의 컬럼

- `cluster_id` — 정수, `-1` 은 노이즈.
- `cluster_name` — LLM 이 만든 짧은 레이블 (`--name-clusters` 사용 시).
- `size` — 클러스터 내 행 수.
- `summary` — LLM 요약 (조건 동일).
- `rep_1` ... `rep_N` — 중심점에 가장 가까운 원본 행.

---

## 6. 주요 옵션

| 옵션 | 기본값 | 역할 |
|---|---|---|
| `--input PATH` | 필수 | 입력 CSV 경로 |
| `--text-col NAME` | — | 단일 컬럼 모드 |
| `--text-cols A,B` | — | 복수 컬럼 결합 모드 |
| `--column-labels A=x,B=y` | — | 복수 컬럼 모드에서의 프리픽스 레이블 |
| `--output-dir PATH` | `data/output` | 출력 루트 디렉터리 |
| `--cache-dir PATH` | `cache` | 캐시 디렉터리 |
| `--model NAME` | `text-embedding-3-small` | 임베딩 모델 |
| `--top-k N` | `5` | 클러스터별 대표 행 수 |
| `--min-clusters N` | `2` | K 하한 |
| `--max-clusters N` | `20` | K 상한 |
| `--target faq|chatbot|insight` | `faq` | 용도별 세분화. `faq`=30-80 클러스터 (FAQ), `chatbot`=50-150 (인텐트), `insight`=silhouette 최대화 |
| `--name-clusters` / `--no-name-clusters` | 사용 | LLM 레이블링 on/off |
| `--name-model NAME` | `gpt-5.4-nano` | 레이블링용 챗 모델 |
| `--advise` / `--no-advise` | on | `parameter_search.html` 상단에 LLM 조언 노트 추가 |
| `--advisor-model NAME` | `gpt-5.4` | 조언 노트용 챗 모델 (실행 전체를 분석하므로 더 강력한 모델) |
| `--format md|html|both` | `md` | `report.*` 형식 |
| `--log-level LEVEL` | `INFO` | stderr 로그 수준 |

---

## 7. 설정

### 환경 변수 (`.env`)

- `OPENAI_API_KEY` (필수)
- `OPENAI_EMBEDDING_MODEL` (선택적 오버라이드)
- `OPENAI_REQUEST_TIMEOUT` (초, 기본 60)

### 캐시

`cache/` 는 콘텐츠 해시로 임베딩 벡터와 LLM 주석을 저장합니다. 모델을
바꾸면 별도 파일로 관리됩니다. 재생성을 강제하려면 해당
`cache/embeddings_*.pkl` 또는 `cache/cluster_annotations_*.pkl` 를 삭제
하세요.

---

## 8. 문제 해결

| 증상 | 대처 |
|---|---|
| `OPENAI_API_KEY is not set` | `.env` 또는 환경 변수로 키를 설정합니다. |
| `Column '...' not found` | CLI 가 사용 가능한 컬럼을 출력하므로 그 중에서 고릅니다. |
| `Column count mismatch on lines: ...` | 닫히지 않은 인용 부호 또는 인용 없이 쉼표가 포함된 값일 수 있습니다. |
| 모든 후보가 노이즈 비율 필터에서 탈락 | 자동으로 필터가 완화되며 경고가 출력됩니다. |
| 점수 `poor` (< 0.20) | 더 풍부한 텍스트를 시도하거나 원본 대표를 수동으로 확인합니다. |
| Windows 에서 `hdbscan` 설치 실패 | `pip install hdbscan --only-binary=:all:` |
| Leiden 이 건너뛰어짐 | `pip install hnswlib python-igraph leidenalg` |

---

## 9. 개인정보 주의사항

- 입력 CSV 에는 개인정보가 포함될 수 있습니다. `data/input/` 와
  `data/output/` 는 `.gitignore` 에 포함되어 있습니다.
- 파이프라인은 OpenAI Embeddings 와 (선택적으로) Chat Completions 에
  텍스트를 전송합니다. 민감한 데이터는 처리 전에 로컬에서 가려 주세요.
- `cache/` 에는 임베딩과 LLM 이 생성한 레이블/요약이 저장됩니다.
  원본 CSV 와 같은 수준으로 관리하세요.
