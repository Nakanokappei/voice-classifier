# voice-classifier — Manual de usuario (Español)

Herramienta de línea de comandos que clasifica CSV con voz del cliente
(tickets de soporte, registros de reparación, etc.) en grupos, etiqueta cada
grupo mediante un LLM y genera informes legibles y procesables.

---

## 1. Qué hace

Dado un CSV cuyas filas contienen texto libre del cliente:

1. Genera embeddings para cada fila única con un modelo de Azure OpenAI.
2. Barre configuraciones candidatas (KMeans / HDBSCAN / Leiden) y selecciona
   la mejor según la silueta coseno.
3. Extrae las filas más cercanas al centroide de cada grupo como
   representantes brutos.
4. Pide al LLM una etiqueta corta y un resumen para cada grupo, apoyándose
   en un contexto de dataset inferido previamente.
5. Resuelve etiquetas duplicadas pidiendo al LLM que diferencie los grupos
   más pequeños del grupo principal.
6. Escribe informes en Markdown, HTML y CSV.

---

## 2. Instalación

```bash
# Se requiere Python 3.10 o superior.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # edite .env y rellene sus valores AZURE_OPENAI_*
```

Backends opcionales (se omiten con seguridad si faltan):

- `hdbscan` — clustering por densidad rápido.
- `hnswlib` + `python-igraph` + `leidenalg` — clustering Leiden basado en grafo.

---

## 3. Primera ejecución

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

La herramienta:

1. Lee el CSV, normaliza texto (NFKC, colapso de espacios, deduplicación).
2. Obtiene / cachea los embeddings de cada fila única.
3. Busca configuraciones de clustering.
4. Extrae las 5 filas más cercanas a cada centroide.
5. Si no se pasa `--no-name-clusters`, infiere contexto, genera etiquetas
   LLM en paralelo y resuelve duplicados.
6. Escribe los resultados en `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Selección de columna

### Columna única

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Varias columnas (concatenadas como `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Selector interactivo

Si omite ambos flags, la CLI muestra candidatos puntuados y le pide elegir:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Estructura de salida

Cada ejecución crea un directorio con timestamp:

```
data/output/20260416_012345/
├── report.md                           Resultado del clustering para humanos
├── report.html                          Igual en HTML (con --format html/both)
├── parameter_search.html                Informe completo de búsqueda con gráfico
├── clusters.csv                         Una fila por cluster: id, nombre,
│                                       tamaño, resumen, rep_1..N
├── <entrada>_classified.csv             Filas originales + cluster_id (+ cluster_name)
├── params.json                          Metadatos legibles por máquina
└── run.log                              Log INFO de la ejecución
```

### Columnas de `clusters.csv`

- `cluster_id` — entero, `-1` para ruido.
- `cluster_name` — etiqueta corta del LLM (sólo con `--name-clusters`).
- `size` — número de filas.
- `summary` — resumen LLM (misma condición).
- `rep_1` ... `rep_N` — filas más cercanas al centroide.

---

## 6. Opciones principales

| Opción | Valor por defecto | Función |
|---|---|---|
| `--input PATH` | requerida | CSV de entrada |
| `--text-col NAME` | — | Columna única para embeddings |
| `--text-cols A,B` | — | Varias columnas concatenadas |
| `--column-labels A=x,B=y` | — | Etiquetas para columnas múltiples |
| `--output-dir PATH` | `data/output` | Directorio raíz de salida |
| `--cache-dir PATH` | `cache` | Directorio de caché |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment de embeddings de Azure OpenAI |
| `--top-k N` | `5` | Representantes por cluster |
| `--min-clusters N` | `2` | Cota inferior de K |
| `--max-clusters N` | `20` | Cota superior de K |
| `--target faq|chatbot|insight` | `faq` | Granularidad por caso de uso. `faq`=30-80 clusters (páginas FAQ), `chatbot`=50-150 (intenciones), `insight`=silhouette máxima |
| `--name-clusters` / `--no-name-clusters` | on | Etiquetado LLM on/off |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | Deployment de chat de Azure OpenAI para etiquetado |
| `--advise` / `--no-advise` | on | Nota de asesoramiento LLM al inicio de `parameter_search.html` |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Deployment de chat de Azure OpenAI para la nota de asesoramiento (razona sobre toda la ejecución) |
| `--format md|html|both` | `md` | Formato de `report.*` |
| `--log-level LEVEL` | `INFO` | Verbosidad en stderr |

---

## 7. Configuración

### Variables de entorno (`.env`)

- `AZURE_OPENAI_API_KEY` (obligatoria)
- `AZURE_OPENAI_ENDPOINT` (obligatoria, p. ej. `https://<resource>.openai.azure.com`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (obligatoria)
- `AZURE_OPENAI_NAMER_DEPLOYMENT` (obligatoria)
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT` (obligatoria)
- `AZURE_OPENAI_API_VERSION` (opcional, `2024-10-21` por defecto)
- `AZURE_OPENAI_REQUEST_TIMEOUT` (segundos, 60 por defecto)

### Caché

`cache/` guarda embeddings y anotaciones del LLM por hash de contenido. Si
cambia de modelo, se usa un caché distinto. Para forzar una ejecución limpia,
elimine el archivo correspondiente en `cache/embeddings_*.pkl` o
`cache/cluster_annotations_*.pkl`.

---

## 8. Solución de problemas

| Síntoma | Solución |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | Defina la clave en `.env` o en el entorno. |
| `Column '...' not found` | La CLI lista las columnas disponibles; use una. |
| `Column count mismatch on lines: ...` | CSV con comillas sin cerrar o comas sueltas. |
| Todos los candidatos rechazados por ruido | El filtro se relaja automáticamente con una advertencia. |
| Puntuación `poor` (< 0.20) | Pruebe con texto más rico o revise los representantes brutos manualmente. |
| `hdbscan` no instala en Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden omitido | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Notas de privacidad

- Los CSV de entrada pueden contener PII. `data/input/` y `data/output/` están
  en `.gitignore`.
- El pipeline envía texto a Azure OpenAI Embeddings y, opcionalmente, a Chat
  Completions. Enmascare datos sensibles localmente antes de procesar.
- La carpeta `cache/` almacena embeddings y etiquetas / resúmenes generados
  por el LLM. Trátela con el mismo cuidado que el CSV de origen.
