# voice-classifier — Руководство пользователя (Русский)

Инструмент командной строки, который кластеризует CSV с голосом клиента
(тикеты поддержки, записи ремонтов и т. п.), присваивает каждой группе
метку через LLM и формирует отчёты, читаемые как людьми, так и машинами.

---

## 1. Что делает инструмент

Для CSV, строки которого содержат свободный текст клиента:

1. Вычислить embedding для каждой уникальной строки с помощью модели Azure OpenAI.
2. Перебрать конфигурации кластеризации (KMeans / HDBSCAN / Leiden) и
   выбрать лучшую по силуэтной мере (cosine).
3. Извлечь ближайшие к центроиду каждой группы строки как первичных
   представителей.
4. Запросить у LLM короткую метку и резюме для каждой группы, опираясь на
   заранее выведенный контекст датасета.
5. Разрешить дубликаты меток, перегенерируя метки для меньших групп.
6. Сформировать отчёты в Markdown, HTML и CSV.

---

## 2. Установка

```bash
# Требуется Python 3.10 или новее.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # затем отредактируйте .env и укажите значения AZURE_OPENAI_*
```

Необязательные бекенды (при отсутствии пропускаются автоматически):

- `hdbscan` — быстрая кластеризация по плотности.
- `hnswlib` + `python-igraph` + `leidenalg` — графовая кластеризация Leiden.

---

## 3. Первый запуск

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

Инструмент:

1. Читает CSV, нормализует текст (NFKC, сжатие пробелов, дедупликация).
2. Получает / кэширует embeddings для каждой уникальной строки.
3. Ищет наилучшую конфигурацию кластеризации.
4. Извлекает 5 ближайших к центроиду строк.
5. Если не указан `--no-name-clusters`, выводит контекст, генерирует метки
   параллельно и разрешает дубликаты.
6. Записывает результаты в `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Выбор столбцов

### Одна колонка

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Несколько колонок (объединяются как `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Интерактивный выбор

Если обе опции опущены, CLI выводит кандидатов с оценкой и просит выбрать:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Структура вывода

Каждый запуск создаёт каталог с меткой времени:

```
data/output/20260416_012345/
├── report.md                           Результат кластеризации для человека
├── report.html                          То же в HTML (при --format html/both)
├── parameter_search.html                Полный отчёт поиска с графиком сверху
├── clusters.csv                         Одна строка на кластер: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               Исходные строки + cluster_id (+ cluster_name)
├── params.json                          Метаданные для машин
└── run.log                              Журнал INFO
```

### Столбцы `clusters.csv`

- `cluster_id` — целое число, `-1` для шума.
- `cluster_name` — короткая метка LLM (только при `--name-clusters`).
- `size` — число строк.
- `summary` — резюме LLM (то же условие).
- `rep_1` ... `rep_N` — строки, ближайшие к центроиду.

---

## 6. Основные опции

| Опция | По умолчанию | Назначение |
|---|---|---|
| `--input PATH` | обязательно | Путь к исходному CSV |
| `--text-col NAME` | — | Одна колонка для embeddings |
| `--text-cols A,B` | — | Несколько колонок, объединяются |
| `--column-labels A=x,B=y` | — | Метки для режима нескольких колонок |
| `--output-dir PATH` | `data/output` | Корневой каталог вывода |
| `--cache-dir PATH` | `cache` | Каталог кэша |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding-деплоймент Azure OpenAI |
| `--top-k N` | `5` | Сколько представителей на кластер |
| `--min-clusters N` | `2` | Нижний предел K |
| `--max-clusters N` | `20` | Верхний предел K |
| `--target faq|chatbot|insight` | `faq` | Гранулярность по цели. `faq`=30-80 кластеров (FAQ), `chatbot`=50-150 (интенты), `insight`=максимум silhouette |
| `--name-clusters` / `--no-name-clusters` | вкл. | Разметка LLM вкл/выкл |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | Чат-деплоймент Azure OpenAI для разметки |
| `--advise` / `--no-advise` | on | Советующее примечание LLM в начале `parameter_search.html` |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Чат-деплоймент Azure OpenAI для советующего примечания (анализирует весь прогон) |
| `--format md|html|both` | `md` | Формат `report.*` |
| `--log-level LEVEL` | `INFO` | Подробность stderr |

---

## 7. Конфигурация

### Переменные окружения (`.env`)

- `AZURE_OPENAI_API_KEY` (обязательно)
- `AZURE_OPENAI_ENDPOINT` (обязательно, напр. `https://<resource>.openai.azure.com`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (обязательно)
- `AZURE_OPENAI_NAMER_DEPLOYMENT` (обязательно)
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT` (обязательно)
- `AZURE_OPENAI_API_VERSION` (необязательно, по умолчанию `2024-10-21`)
- `AZURE_OPENAI_REQUEST_TIMEOUT` (секунды, по умолчанию 60)

### Кэш

`cache/` хранит векторы embedding и аннотации LLM по хэшу содержимого.
Смена модели → другой файл кэша. Для принудительной регенерации удалите
соответствующий `cache/embeddings_*.pkl` или
`cache/cluster_annotations_*.pkl`.

---

## 8. Устранение неполадок

| Симптом | Решение |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | Укажите ключ в `.env` или в переменной окружения. |
| `Column '...' not found` | CLI выводит доступные колонки — выберите нужную. |
| `Column count mismatch on lines: ...` | Незакрытые кавычки или запятые в значениях без кавычек. |
| Все кандидаты отброшены фильтром шума | Фильтр автоматически смягчается (с предупреждением). |
| Оценка `poor` (< 0.20) | Попробуйте более насыщенный текст или проверьте сырые представители. |
| `hdbscan` не ставится в Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden пропущен | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Конфиденциальность

- Исходные CSV могут содержать ПД. `data/input/` и `data/output/` входят в
  `.gitignore`.
- Пайплайн отправляет текст в Azure OpenAI Embeddings и, опционально, в
  Chat Completions. Маскируйте чувствительные данные локально заранее.
- В `cache/` хранятся embeddings и метки/резюме, сгенерированные LLM.
  Обращайтесь с этим каталогом так же бережно, как с исходным CSV.
