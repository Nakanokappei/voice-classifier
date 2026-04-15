# voice-classifier — Manual do Utilizador (Português)

Ferramenta de linha de comandos que agrupa CSVs com voz do cliente (tickets
de apoio, registos de reparação, etc.) em clusters, etiqueta cada cluster
através de um LLM e produz relatórios legíveis por humanos e por máquinas.

---

## 1. O que faz

Dado um CSV cujas linhas contêm texto livre do cliente:

1. Calcular embeddings para cada linha única com um modelo OpenAI.
2. Percorrer configurações candidatas (KMeans / HDBSCAN / Leiden) e
   seleccionar a melhor segundo silhueta coseno.
3. Extrair as linhas mais próximas do centróide de cada cluster como
   representantes em bruto.
4. Pedir ao LLM uma etiqueta curta e um resumo por cluster, fundamentado
   num contexto de dataset previamente inferido.
5. Resolver etiquetas duplicadas diferenciando os clusters mais pequenos.
6. Escrever relatórios em Markdown, HTML e CSV.

---

## 2. Instalação

```bash
# Requer Python 3.10 ou superior.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # edite .env e preencha OPENAI_API_KEY
```

Backends opcionais (ignorados com segurança se estiverem em falta):

- `hdbscan` — clustering por densidade rápido.
- `hnswlib` + `python-igraph` + `leidenalg` — clustering Leiden baseado em grafo.

---

## 3. Primeira execução

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

A ferramenta:

1. Lê o CSV, normaliza texto (NFKC, compactação de espaços, deduplicação).
2. Obtém / coloca em cache os embeddings de cada linha única.
3. Procura configurações de clustering.
4. Extrai as 5 linhas mais próximas de cada centróide.
5. Sem `--no-name-clusters`, infere contexto, gera etiquetas em paralelo
   e resolve duplicados.
6. Escreve os resultados em `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Selecção de colunas

### Coluna única

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Várias colunas (concatenadas como `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Selector interactivo

Se omitir ambas as flags, a CLI mostra candidatos pontuados e pede-lhe para
escolher:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Estrutura de saída

Cada execução cria um directório com marca temporal:

```
data/output/20260416_012345/
├── report.md                           Resultado de clustering para humanos
├── report.html                          Mesmo em HTML (com --format html/both)
├── parameter_search.html                Relatório completo com gráfico no topo
├── clusters.csv                         Uma linha por cluster: id, nome,
│                                       tamanho, resumo, rep_1..N
├── <input>_classified.csv               Linhas originais + cluster_id (+ cluster_name)
├── params.json                          Metadados legíveis por máquina
└── run.log                              Log INFO da execução
```

### Colunas de `clusters.csv`

- `cluster_id` — inteiro, `-1` para ruído.
- `cluster_name` — etiqueta curta do LLM (apenas com `--name-clusters`).
- `size` — número de linhas.
- `summary` — resumo do LLM (mesma condição).
- `rep_1` ... `rep_N` — linhas mais próximas do centróide.

---

## 6. Opções principais

| Opção | Predefinição | Função |
|---|---|---|
| `--input PATH` | obrigatória | CSV de entrada |
| `--text-col NAME` | — | Coluna única para embeddings |
| `--text-cols A,B` | — | Várias colunas concatenadas |
| `--column-labels A=x,B=y` | — | Etiquetas para modo multi-coluna |
| `--output-dir PATH` | `data/output` | Directório raiz de saída |
| `--cache-dir PATH` | `cache` | Directório de cache |
| `--model NAME` | `text-embedding-3-small` | Modelo de embedding |
| `--top-k N` | `5` | Representantes por cluster |
| `--min-clusters N` | `2` | Limite inferior de K |
| `--max-clusters N` | `20` | Limite superior de K |
| `--target faq|chatbot|insight` | `faq` | Granularidade por caso de uso. `faq`=30-80 clusters (FAQ), `chatbot`=50-150 (intenções), `insight`=silhueta máxima |
| `--name-clusters` / `--no-name-clusters` | ligado | Etiquetagem LLM on/off |
| `--name-model NAME` | `gpt-5.4-nano` | Modelo chat para etiquetagem |
| `--advise` / `--no-advise` | on | Nota consultiva LLM no topo de `parameter_search.html` |
| `--advisor-model NAME` | `gpt-5.4` | Modelo de chat para a nota consultiva (analisa toda a execução) |
| `--format md|html|both` | `md` | Formato de `report.*` |
| `--log-level LEVEL` | `INFO` | Verbosidade em stderr |

---

## 7. Configuração

### Variáveis de ambiente (`.env`)

- `OPENAI_API_KEY` (obrigatória)
- `OPENAI_EMBEDDING_MODEL` (override opcional)
- `OPENAI_REQUEST_TIMEOUT` (segundos, predefinição 60)

### Cache

`cache/` guarda embeddings e anotações LLM por hash de conteúdo. Mudar de
modelo implica um ficheiro de cache distinto. Para forçar regeneração,
apague o `cache/embeddings_*.pkl` ou `cache/cluster_annotations_*.pkl`
correspondente.

---

## 8. Resolução de problemas

| Sintoma | Solução |
|---|---|
| `OPENAI_API_KEY is not set` | Preencha a chave em `.env` ou no ambiente. |
| `Column '...' not found` | A CLI lista as colunas disponíveis; escolha uma. |
| `Column count mismatch on lines: ...` | Aspas não fechadas ou vírgulas em valores sem aspas. |
| Todos os candidatos rejeitados pelo rácio de ruído | O filtro é relaxado automaticamente com aviso. |
| Pontuação `poor` (< 0.20) | Experimente texto mais rico ou inspeccione os representantes manualmente. |
| `hdbscan` falha a instalar em Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden ignorado | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Notas de privacidade

- Os CSVs de entrada podem conter dados pessoais. `data/input/` e
  `data/output/` estão em `.gitignore`.
- O pipeline envia texto para OpenAI Embeddings e, opcionalmente, para
  Chat Completions. Mascare dados sensíveis localmente antes do processamento.
- A pasta `cache/` guarda embeddings e etiquetas/resumos gerados por LLM.
  Trate-a com o mesmo cuidado do CSV original.
