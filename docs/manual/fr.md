# voice-classifier — Manuel utilisateur (Français)

Outil en ligne de commande qui regroupe des CSV de voix du client (tickets
de support, fiches de réparation, etc.) en clusters, étiquette chaque
cluster via un LLM et produit des rapports lisibles et automatisables.

---

## 1. Ce que fait l'outil

Pour un CSV dont les lignes contiennent du texte libre :

1. Vectoriser chaque ligne unique avec un modèle d'embedding d'OpenAI.
2. Balayer des configurations candidates (KMeans / HDBSCAN / Leiden) et
   sélectionner la meilleure selon la silhouette cosinus.
3. Extraire les lignes les plus proches du centroïde de chaque cluster.
4. Demander au LLM une étiquette courte et un résumé par cluster, en
   s'appuyant sur un contexte de jeu de données préalablement inféré.
5. Dédupliquer les étiquettes identiques en différenciant les plus petits
   clusters.
6. Écrire des rapports en Markdown, HTML et CSV.

---

## 2. Installation

```bash
# Python 3.10 ou plus récent requis.
python -m venv .venv
source .venv/bin/activate           # Windows : .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # puis éditez .env pour renseigner OPENAI_API_KEY
```

Backends optionnels (ignorés automatiquement s'ils manquent) :

- `hdbscan` — clustering par densité rapide.
- `hnswlib` + `python-igraph` + `leidenalg` — clustering Leiden sur graphe.

---

## 3. Première exécution

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

L'outil :

1. Lit le CSV, normalise le texte (NFKC, compaction des espaces, déduplication).
2. Récupère / met en cache les embeddings de chaque ligne unique.
3. Recherche la meilleure configuration de clustering.
4. Extrait les 5 lignes les plus proches de chaque centroïde.
5. Sauf si `--no-name-clusters` est passé, infère le contexte, génère les
   étiquettes en parallèle et résout les doublons.
6. Écrit les résultats dans `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Sélection des colonnes

### Colonne unique

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Plusieurs colonnes (fusionnées en `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Sélecteur interactif

Omettez les deux options et la CLI affiche les colonnes candidates
classées et vous demande de choisir :

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Arborescence de sortie

Chaque exécution crée un répertoire horodaté :

```
data/output/20260416_012345/
├── report.md                           Résultat du clustering pour humains
├── report.html                          Idem en HTML (avec --format html/both)
├── parameter_search.html                Rapport complet avec graphique en tête
├── clusters.csv                         Une ligne par cluster : id, nom,
│                                       taille, résumé, rep_1..N
├── <entrée>_classified.csv              Lignes originales + cluster_id (+ cluster_name)
├── params.json                          Métadonnées lisibles par machine
└── run.log                              Log INFO de l'exécution
```

### Colonnes de `clusters.csv`

- `cluster_id` — entier, `-1` pour le bruit.
- `cluster_name` — étiquette courte du LLM (uniquement avec `--name-clusters`).
- `size` — nombre de lignes.
- `summary` — résumé LLM (même condition).
- `rep_1` ... `rep_N` — lignes les plus proches du centroïde.

---

## 6. Options principales

| Option | Défaut | Rôle |
|---|---|---|
| `--input PATH` | obligatoire | CSV d'entrée |
| `--text-col NAME` | — | Colonne unique pour les embeddings |
| `--text-cols A,B` | — | Plusieurs colonnes concaténées |
| `--column-labels A=x,B=y` | — | Étiquettes pour le mode multi-colonnes |
| `--output-dir PATH` | `data/output` | Répertoire racine de sortie |
| `--cache-dir PATH` | `cache` | Répertoire de cache |
| `--model NAME` | `text-embedding-3-small` | Modèle d'embedding |
| `--top-k N` | `5` | Représentants par cluster |
| `--min-clusters N` | `2` | Borne inférieure de K |
| `--max-clusters N` | `20` | Borne supérieure de K |
| `--name-clusters` / `--no-name-clusters` | activé | Étiquetage LLM on/off |
| `--name-model NAME` | `gpt-5.4-nano` | Modèle chat pour l'étiquetage |
| `--format md|html|both` | `md` | Format de `report.*` |
| `--log-level LEVEL` | `INFO` | Verbosité stderr |

---

## 7. Configuration

### Variables d'environnement (`.env`)

- `OPENAI_API_KEY` (obligatoire)
- `OPENAI_EMBEDDING_MODEL` (override optionnel)
- `OPENAI_REQUEST_TIMEOUT` (secondes, 60 par défaut)

### Cache

`cache/` conserve les embeddings et les annotations LLM, indexés par hash
de contenu. Le changement de modèle écrit dans un fichier distinct. Pour
forcer une régénération, supprimez le `cache/embeddings_*.pkl` ou
`cache/cluster_annotations_*.pkl` concerné.

---

## 8. Dépannage

| Symptôme | Solution |
|---|---|
| `OPENAI_API_KEY is not set` | Renseigner la clé dans `.env` ou l'environnement. |
| `Column '...' not found` | La CLI liste les colonnes disponibles ; choisissez-en une. |
| `Column count mismatch on lines: ...` | CSV avec guillemets non fermés ou virgules non protégées. |
| Tous les candidats filtrés par ratio de bruit | Le filtre est automatiquement assoupli avec un avertissement. |
| Score `poor` (< 0.20) | Fournissez un texte plus riche ou inspectez manuellement les représentants. |
| `hdbscan` ne s'installe pas sous Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden est ignoré | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Confidentialité

- Les CSV d'entrée peuvent contenir des données personnelles. `data/input/` et
  `data/output/` sont dans `.gitignore`.
- Le pipeline envoie du texte à OpenAI Embeddings et (optionnellement)
  Chat Completions. Masquez les données sensibles en local avant traitement.
- Le dossier `cache/` stocke les embeddings et les étiquettes / résumés
  générés par LLM. Protégez-le au même niveau que le CSV source.
