# voice-classifier — Gebruikershandleiding (Nederlands)

Commandline-tool die CSV-bestanden met klantstem (supporttickets,
reparatieregistraties, enz.) in clusters groepeert, elk cluster labelt via
een LLM en rapporten oplevert die zowel mensen- als machineleesbaar zijn.

---

## 1. Wat het doet

Gegeven een CSV met vrije tekst per klantregel:

1. Bereken embeddings voor elke unieke rij met een OpenAI-model.
2. Doorloop kandidaat-configuraties (KMeans / HDBSCAN / Leiden) en kies de
   beste op basis van cosine-silhouette.
3. Selecteer de rijen dichtst bij het centroïde van elk cluster als ruwe
   representanten.
4. Vraag het LLM per cluster een kort label en een samenvatting, gebaseerd
   op een vooraf afgeleide datasetcontext.
5. Los dubbele labels op door kleinere clusters opnieuw te labelen.
6. Schrijf rapporten in Markdown, HTML en CSV.

---

## 2. Installatie

```bash
# Python 3.10 of nieuwer vereist.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # bewerk .env en vul OPENAI_API_KEY in
```

Optionele backends (worden automatisch overgeslagen als ze ontbreken):

- `hdbscan` — snelle dichtheidsgebaseerde clustering.
- `hnswlib` + `python-igraph` + `leidenalg` — grafgebaseerde Leiden-clustering.

---

## 3. Eerste run

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

De tool:

1. Leest de CSV, normaliseert tekst (NFKC, spaties comprimeren, dedup).
2. Haalt / cachet embeddings voor elke unieke rij.
3. Zoekt de beste clustering-configuratie.
4. Extraheert de 5 rijen die het dichtst bij elk centroïde liggen.
5. Tenzij `--no-name-clusters` meegegeven is, leidt context af, genereert
   labels parallel en lost duplicaten op.
6. Schrijft de resultaten naar `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Kolomkeuze

### Enkele kolom

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Meerdere kolommen (samengevoegd als `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Interactieve keuze

Zonder beide vlaggen toont de CLI gescoorde kandidaten en vraagt om een
keuze:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Uitvoerstructuur

Elke run maakt een map met tijdstempel:

```
data/output/20260416_012345/
├── report.md                           Cluster-resultaat voor menselijk gebruik
├── report.html                          Idem in HTML (bij --format html/both)
├── parameter_search.html                Volledig zoekrapport met grafiek bovenaan
├── clusters.csv                         Eén rij per cluster: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               Originele rijen + cluster_id (+ cluster_name)
├── params.json                          Machineleesbare metadata
└── run.log                              Uitvoerlog op INFO-niveau
```

### Kolommen van `clusters.csv`

- `cluster_id` — integer, `-1` voor ruis.
- `cluster_name` — kort LLM-label (alleen met `--name-clusters`).
- `size` — aantal rijen.
- `summary` — LLM-samenvatting (zelfde voorwaarde).
- `rep_1` ... `rep_N` — originele rijen nabij het centroïde.

---

## 6. Belangrijke opties

| Optie | Standaard | Doel |
|---|---|---|
| `--input PATH` | verplicht | Invoer-CSV |
| `--text-col NAME` | — | Enkele kolom voor embeddings |
| `--text-cols A,B` | — | Meerdere kolommen samenvoegen |
| `--column-labels A=x,B=y` | — | Labels voor multi-kolom-modus |
| `--output-dir PATH` | `data/output` | Hoofdmap voor uitvoer |
| `--cache-dir PATH` | `cache` | Cachemap |
| `--model NAME` | `text-embedding-3-small` | Embedding-model |
| `--top-k N` | `5` | Representanten per cluster |
| `--min-clusters N` | `2` | Ondergrens voor K |
| `--max-clusters N` | `20` | Bovengrens voor K |
| `--name-clusters` / `--no-name-clusters` | aan | LLM-labelling aan/uit |
| `--name-model NAME` | `gpt-5.4-nano` | Chat-model voor labelling |
| `--format md|html|both` | `md` | Formaat van `report.*` |
| `--log-level LEVEL` | `INFO` | Verbositeit op stderr |

---

## 7. Configuratie

### Omgevingsvariabelen (`.env`)

- `OPENAI_API_KEY` (verplicht)
- `OPENAI_EMBEDDING_MODEL` (optionele override)
- `OPENAI_REQUEST_TIMEOUT` (seconden, standaard 60)

### Cache

`cache/` bewaart embedding-vectoren en LLM-annotaties per content-hash.
Wissel van model = ander cachebestand. Wil je een herberekening afdwingen,
verwijder het bijbehorende `cache/embeddings_*.pkl` of
`cache/cluster_annotations_*.pkl`.

---

## 8. Probleemoplossing

| Symptoom | Oplossing |
|---|---|
| `OPENAI_API_KEY is not set` | Vul de sleutel in via `.env` of de omgeving. |
| `Column '...' not found` | De CLI toont beschikbare kolommen; kies er een. |
| `Column count mismatch on lines: ...` | Niet-gesloten aanhalingstekens of komma's in ongecitaliseerde waarden. |
| Alle kandidaten afgewezen door ruisfilter | Het filter wordt automatisch versoepeld (waarschuwing). |
| Score `poor` (< 0.20) | Probeer rijkere tekst of bekijk de ruwe representanten handmatig. |
| `hdbscan` installeert niet op Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden wordt overgeslagen | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Privacy-opmerkingen

- Invoer-CSV's kunnen persoonsgegevens bevatten. `data/input/` en
  `data/output/` staan in `.gitignore`.
- De pipeline stuurt tekst naar OpenAI Embeddings en (optioneel) Chat
  Completions. Maskeer gevoelige data lokaal voor verwerking.
- De map `cache/` bevat embeddings en LLM-gegenereerde labels/samenvattingen.
  Behandel deze met dezelfde zorg als de oorspronkelijke CSV.
