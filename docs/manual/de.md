# voice-classifier — Benutzerhandbuch (Deutsch)

Kommandozeilenwerkzeug, das CSVs mit Kundenstimmen (Support-Tickets,
Reparaturprotokolle usw.) in Cluster einteilt, jeden Cluster per LLM
etikettiert und Berichte in menschen- und maschinenlesbaren Formaten liefert.

---

## 1. Funktionsumfang

Für eine CSV mit freien Textzeilen pro Kunde:

1. Embeddings für jede eindeutige Zeile mit einem Azure-OpenAI-Modell berechnen.
2. Kandidatenkonfigurationen (KMeans / HDBSCAN / Leiden) durchlaufen und die
   beste per Cosine-Silhouette wählen.
3. Zeilen, die dem Zentroid jedes Clusters am nächsten liegen, als
   Repräsentanten extrahieren.
4. Das LLM um Kurzlabel und Zusammenfassung pro Cluster bitten, gestützt
   auf einen vorab inferierten Dataset-Kontext.
5. Label-Duplikate durch Differenzierung der kleineren Cluster auflösen.
6. Markdown-, HTML- und CSV-Berichte schreiben.

---

## 2. Installation

```bash
# Python 3.10 oder neuer erforderlich.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # dann .env bearbeiten und Ihre AZURE_OPENAI_*-Werte eintragen
```

Optionale Backends (bei Fehlen automatisch übersprungen):

- `hdbscan` — schnelles dichtebasiertes Clustering.
- `hnswlib` + `python-igraph` + `leidenalg` — graphenbasiertes Leiden-Clustering.

---

## 3. Erste Ausführung

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

Das Tool:

1. Liest die CSV, normalisiert den Text (NFKC, Whitespace-Kollaps, Dedupe).
2. Holt / cached Embeddings für jede eindeutige Zeile.
3. Sucht die beste Clustering-Konfiguration.
4. Extrahiert die 5 Zeilen, die jedem Zentroid am nächsten sind.
5. Sofern `--no-name-clusters` nicht gesetzt ist: inferiert Dataset-Kontext,
   generiert Labels parallel, löst Duplikate.
6. Schreibt Ergebnisse nach `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Spaltenauswahl

### Einzelne Spalte

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Mehrere Spalten (als `label: value` zusammengeführt)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Interaktive Auswahl

Ohne beide Flags zeigt die CLI bewertete Kandidaten und fragt nach der Wahl:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Ausgabestruktur

Jeder Lauf legt ein Verzeichnis mit Zeitstempel an:

```
data/output/20260416_012345/
├── report.md                           Clustering-Ergebnis für Menschen
├── report.html                          Dasselbe als HTML (bei --format html/both)
├── parameter_search.html                Vollständiger Suchbericht mit Chart oben
├── clusters.csv                         Eine Zeile pro Cluster: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               Originalzeilen + cluster_id (+ cluster_name)
├── params.json                          Maschinenlesbare Metadaten
└── run.log                              INFO-Log der Ausführung
```

### Spalten von `clusters.csv`

- `cluster_id` — Ganzzahl, `-1` für Noise.
- `cluster_name` — Kurz-Label vom LLM (nur mit `--name-clusters`).
- `size` — Anzahl Zeilen.
- `summary` — LLM-Zusammenfassung (gleiche Bedingung).
- `rep_1` ... `rep_N` — Originalzeilen nahe am Zentroid.

---

## 6. Wichtige Optionen

| Option | Standard | Zweck |
|---|---|---|
| `--input PATH` | erforderlich | Eingabe-CSV |
| `--text-col NAME` | — | Einzelspaltenmodus |
| `--text-cols A,B` | — | Mehrere Spalten zusammenführen |
| `--column-labels A=x,B=y` | — | Präfixe für den Mehrspaltenmodus |
| `--output-dir PATH` | `data/output` | Ausgabe-Rootverzeichnis |
| `--cache-dir PATH` | `cache` | Cache-Verzeichnis |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure-OpenAI-Embedding-Deployment |
| `--top-k N` | `5` | Repräsentanten pro Cluster |
| `--min-clusters N` | `2` | Untere K-Grenze |
| `--max-clusters N` | `20` | Obere K-Grenze |
| `--target faq|chatbot|insight` | `faq` | Zielgranularität. `faq`=30-80 Cluster (FAQ), `chatbot`=50-150 (Intents), `insight`=maximale Silhouette |
| `--name-clusters` / `--no-name-clusters` | an | LLM-Labeling ein/aus |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | Azure-OpenAI-Chat-Deployment fürs Labeling |
| `--advise` / `--no-advise` | on | LLM-Hinweis oben in `parameter_search.html` ein/aus |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Azure-OpenAI-Chat-Deployment für den Hinweis (analysiert den gesamten Lauf) |
| `--format md|html|both` | `md` | Format von `report.*` |
| `--log-level LEVEL` | `INFO` | Verbosität auf stderr |

---

## 7. Konfiguration

### Umgebungsvariablen (`.env`)

- `AZURE_OPENAI_API_KEY` (erforderlich)
- `AZURE_OPENAI_ENDPOINT` (erforderlich, z. B. `https://<resource>.openai.azure.com`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (erforderlich)
- `AZURE_OPENAI_NAMER_DEPLOYMENT` (erforderlich)
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT` (erforderlich)
- `AZURE_OPENAI_API_VERSION` (optional, Standard `2024-10-21`)
- `AZURE_OPENAI_REQUEST_TIMEOUT` (Sekunden, Standard 60)

### Cache

`cache/` speichert Embedding-Vektoren und LLM-Annotationen nach
Content-Hash. Modellwechsel → andere Cachedatei. Zum Erzwingen einer
Neuberechnung die passende `cache/embeddings_*.pkl`- oder
`cache/cluster_annotations_*.pkl`-Datei löschen.

---

## 8. Problembehebung

| Symptom | Lösung |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | Schlüssel in `.env` oder Umgebungsvariable setzen. |
| `Column '...' not found` | CLI listet verfügbare Spalten; eine davon wählen. |
| `Column count mismatch on lines: ...` | Nicht geschlossene Anführungszeichen oder ungequotete Kommas. |
| Alle Kandidaten durch Noise-Filter verworfen | Filter wird automatisch gelockert (Warnung). |
| Bewertung `poor` (< 0.20) | Reichhaltigeren Text liefern oder Rohrepräsentanten manuell sichten. |
| `hdbscan`-Installation scheitert auf Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden wird übersprungen | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Datenschutzhinweise

- Eingabe-CSVs können personenbezogene Daten enthalten. `data/input/` und
  `data/output/` sind in `.gitignore`.
- Die Pipeline schickt Text an Azure OpenAI Embeddings und optional an
  Chat Completions. Sensible Daten vorher lokal maskieren.
- `cache/` speichert Embeddings sowie LLM-generierte Labels/Zusammenfassungen.
  Mit dem gleichen Schutzniveau wie die Quell-CSV behandeln.
