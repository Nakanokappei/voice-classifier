# voice-classifier — Manuale utente (Italiano)

Strumento a riga di comando che classifica CSV con voci dei clienti
(ticket di assistenza, schede di riparazione, ecc.) in cluster, etichetta
ogni cluster tramite un LLM e produce report leggibili sia dalle persone
sia dai sistemi.

---

## 1. Cosa fa

Dato un CSV le cui righe contengono testo libero:

1. Calcola un embedding per ogni riga unica con un modello OpenAI.
2. Esplora configurazioni candidate (KMeans / HDBSCAN / Leiden) e sceglie
   la migliore in base alla silhouette coseno.
3. Estrae le righe più vicine al centroide di ciascun cluster.
4. Chiede al LLM un'etichetta breve e un riassunto per ogni cluster,
   basandosi su un contesto del dataset inferito in precedenza.
5. Risolve le etichette duplicate differenziando i cluster più piccoli.
6. Scrive report in Markdown, HTML e CSV.

---

## 2. Installazione

```bash
# È richiesto Python 3.10 o superiore.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # poi modifica .env inserendo OPENAI_API_KEY
```

Backend facoltativi (automaticamente ignorati se mancanti):

- `hdbscan` — clustering per densità veloce.
- `hnswlib` + `python-igraph` + `leidenalg` — clustering Leiden su grafo.

---

## 3. Prima esecuzione

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

Lo strumento:

1. Legge il CSV, normalizza il testo (NFKC, compattazione spazi, deduplicazione).
2. Ottiene / memorizza in cache gli embedding di ogni riga unica.
3. Cerca configurazioni di clustering.
4. Estrae le 5 righe più vicine a ogni centroide.
5. Salvo `--no-name-clusters`, inferisce il contesto, genera etichette in
   parallelo e risolve i duplicati.
6. Scrive i risultati in `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Selezione delle colonne

### Colonna singola

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Più colonne (concatenate come `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Selettore interattivo

Omettendo entrambi i flag, la CLI mostra i candidati con punteggio e ti
chiede di scegliere:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Struttura di output

Ogni esecuzione crea una directory con timestamp:

```
data/output/20260416_012345/
├── report.md                           Risultato leggibile dalle persone
├── report.html                          Idem in HTML (con --format html/both)
├── parameter_search.html                Report completo con grafico in alto
├── clusters.csv                         Una riga per cluster: id, nome,
│                                       dimensione, summary, rep_1..N
├── <input>_classified.csv               Righe originali + cluster_id (+ cluster_name)
├── params.json                          Metadati leggibili dalla macchina
└── run.log                              Log INFO dell'esecuzione
```

### Colonne di `clusters.csv`

- `cluster_id` — intero, `-1` per il rumore.
- `cluster_name` — etichetta breve LLM (solo con `--name-clusters`).
- `size` — numero di righe.
- `summary` — riassunto LLM (stessa condizione).
- `rep_1` ... `rep_N` — righe più vicine al centroide.

---

## 6. Opzioni principali

| Opzione | Default | Scopo |
|---|---|---|
| `--input PATH` | obbligatoria | CSV di input |
| `--text-col NAME` | — | Colonna singola per gli embedding |
| `--text-cols A,B` | — | Più colonne concatenate |
| `--column-labels A=x,B=y` | — | Etichette per il modo multi-colonna |
| `--output-dir PATH` | `data/output` | Directory radice di output |
| `--cache-dir PATH` | `cache` | Directory della cache |
| `--model NAME` | `text-embedding-3-small` | Modello di embedding |
| `--top-k N` | `5` | Rappresentanti per cluster |
| `--min-clusters N` | `2` | Limite inferiore per K |
| `--max-clusters N` | `20` | Limite superiore per K |
| `--target faq|chatbot|insight` | `faq` | Granularità per caso d'uso. `faq`=30-80 cluster (FAQ), `chatbot`=50-150 (intenti), `insight`=silhouette massima |
| `--name-clusters` / `--no-name-clusters` | on | Etichettatura LLM on/off |
| `--name-model NAME` | `gpt-5.4-nano` | Modello chat per l'etichettatura |
| `--advise` / `--no-advise` | on | Nota consultiva LLM all'inizio di `parameter_search.html` |
| `--advisor-model NAME` | `gpt-5.4` | Modello di chat per la nota consultiva (ragiona sull'intera esecuzione) |
| `--format md|html|both` | `md` | Formato di `report.*` |
| `--log-level LEVEL` | `INFO` | Verbosità stderr |

---

## 7. Configurazione

### Variabili d'ambiente (`.env`)

- `OPENAI_API_KEY` (obbligatoria)
- `OPENAI_EMBEDDING_MODEL` (override facoltativo)
- `OPENAI_REQUEST_TIMEOUT` (secondi, default 60)

### Cache

`cache/` contiene embedding e annotazioni LLM indicizzati per hash del
contenuto. Cambiando modello si usa un file distinto. Per forzare una
rigenerazione elimina il relativo `cache/embeddings_*.pkl` o
`cache/cluster_annotations_*.pkl`.

---

## 8. Risoluzione dei problemi

| Sintomo | Soluzione |
|---|---|
| `OPENAI_API_KEY is not set` | Inserisci la chiave in `.env` o nell'ambiente. |
| `Column '...' not found` | La CLI elenca le colonne disponibili; scegline una. |
| `Column count mismatch on lines: ...` | CSV con virgolette non chiuse o virgole non protette. |
| Tutti i candidati rimossi dal filtro di rumore | Il filtro viene rilassato automaticamente con avviso. |
| Punteggio `poor` (< 0.20) | Usa testo più ricco o ispeziona manualmente i rappresentanti. |
| `hdbscan` non si installa su Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden ignorato | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Note sulla privacy

- I CSV di input possono contenere dati personali. `data/input/` e
  `data/output/` sono inseriti in `.gitignore`.
- Il pipeline invia testo a OpenAI Embeddings e, facoltativamente, a
  Chat Completions. Maschera localmente i dati sensibili prima dell'uso.
- La cartella `cache/` conserva embedding e etichette / riassunti generati
  dal LLM. Trattala con la stessa cura del CSV originale.
