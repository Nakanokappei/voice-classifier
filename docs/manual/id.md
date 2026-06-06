# voice-classifier — Panduan Pengguna (Bahasa Indonesia)

Alat baris perintah yang mengelompokkan CSV berisi suara pelanggan (tiket
dukungan, catatan reparasi, dll.) menjadi klaster, memberi label setiap
klaster dengan LLM, dan menghasilkan laporan yang terbaca manusia maupun
terbaca mesin.

---

## 1. Fungsinya

Diberi CSV dengan baris berisi teks bebas pelanggan:

1. Hitung embedding untuk setiap baris unik dengan model Azure OpenAI.
2. Lakukan sapuan konfigurasi kandidat (KMeans / HDBSCAN / Leiden) dan pilih
   yang terbaik berdasarkan silhouette cosine.
3. Ekstrak baris terdekat ke pusat tiap klaster sebagai representasi mentah.
4. Minta LLM memberi label pendek dan ringkasan per klaster, berdasarkan
   konteks dataset yang telah diinferensi sebelumnya.
5. Selesaikan label duplikat dengan mendiferensiasi klaster yang lebih kecil.
6. Tulis laporan Markdown, HTML, dan CSV.

---

## 2. Instalasi

```bash
# Diperlukan Python 3.10 atau lebih baru.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # kemudian edit .env isi nilai AZURE_OPENAI_* Anda
```

Backend opsional (dilewati otomatis jika tidak ada):

- `hdbscan` — klasterisasi berbasis densitas yang cepat.
- `hnswlib` + `python-igraph` + `leidenalg` — klasterisasi Leiden berbasis graf.

---

## 3. Jalankan pertama kali

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

Alur kerjanya:

1. Baca CSV, normalisasi teks (NFKC, kompaksi spasi, deduplikasi).
2. Ambil / cache embedding untuk tiap baris unik.
3. Cari konfigurasi klasterisasi terbaik.
4. Ambil 5 baris terdekat ke tiap pusat.
5. Kecuali `--no-name-clusters`, infer konteks dataset, generasikan label
   paralel, lalu selesaikan duplikat.
6. Tulis hasil ke `data/output/YYYYMMDD_HHMMSS/`.

---

## 4. Pemilihan kolom

### Satu kolom

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### Beberapa kolom (digabung sebagai `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### Pemilih interaktif

Bila kedua flag tidak diberikan, CLI menampilkan kandidat berperingkat dan
meminta Anda memilih:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. Struktur keluaran

Setiap eksekusi membuat direktori dengan timestamp:

```
data/output/20260416_012345/
├── report.md                           Hasil klasterisasi untuk manusia
├── report.html                          Sama dalam HTML (dengan --format html/both)
├── parameter_search.html                Laporan pencarian lengkap + grafik
├── clusters.csv                         Satu baris per klaster: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               Baris asli + cluster_id (+ cluster_name)
├── params.json                          Metadata terbaca mesin
└── run.log                              Log eksekusi tingkat INFO
```

### Kolom `clusters.csv`

- `cluster_id` — integer, `-1` untuk noise.
- `cluster_name` — label pendek dari LLM (hanya saat `--name-clusters`).
- `size` — jumlah baris.
- `summary` — ringkasan LLM (syarat sama).
- `rep_1` ... `rep_N` — baris asli terdekat ke pusat.

---

## 6. Opsi utama

| Opsi | Default | Kegunaan |
|---|---|---|
| `--input PATH` | wajib | Path CSV masukan |
| `--text-col NAME` | — | Kolom tunggal untuk embedding |
| `--text-cols A,B` | — | Beberapa kolom digabungkan |
| `--column-labels A=x,B=y` | — | Label untuk mode multi-kolom |
| `--output-dir PATH` | `data/output` | Direktori output utama |
| `--cache-dir PATH` | `cache` | Direktori cache |
| `--model NAME` | `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment embedding Azure OpenAI |
| `--top-k N` | `5` | Baris representatif per klaster |
| `--min-clusters N` | `2` | Batas bawah K |
| `--max-clusters N` | `20` | Batas atas K |
| `--target faq|chatbot|insight` | `faq` | Granularitas berdasarkan kasus pemakaian. `faq`=30-80 klaster (FAQ), `chatbot`=50-150 (intent), `insight`=silhouette maksimum |
| `--name-clusters` / `--no-name-clusters` | on | Aktif/mati pelabelan LLM |
| `--name-model NAME` | `$AZURE_OPENAI_NAMER_DEPLOYMENT` | Deployment chat Azure OpenAI untuk pelabelan |
| `--advise` / `--no-advise` | on | Catatan saran LLM di bagian atas `parameter_search.html` |
| `--advisor-model NAME` | `$AZURE_OPENAI_ADVISOR_DEPLOYMENT` | Deployment chat Azure OpenAI untuk catatan saran (menganalisis keseluruhan run) |
| `--format md|html|both` | `md` | Format `report.*` |
| `--log-level LEVEL` | `INFO` | Verbositas stderr |

---

## 7. Konfigurasi

### Variabel lingkungan (`.env`)

- `AZURE_OPENAI_API_KEY` (wajib)
- `AZURE_OPENAI_ENDPOINT` (wajib, mis. `https://<resource>.openai.azure.com`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (wajib)
- `AZURE_OPENAI_NAMER_DEPLOYMENT` (wajib)
- `AZURE_OPENAI_ADVISOR_DEPLOYMENT` (wajib)
- `AZURE_OPENAI_API_VERSION` (opsional, default `2024-10-21`)
- `AZURE_OPENAI_REQUEST_TIMEOUT` (detik, default 60)

### Cache

`cache/` menyimpan vektor embedding dan anotasi LLM per hash konten. Ganti
model = file cache berbeda. Untuk memaksa regenerasi, hapus
`cache/embeddings_*.pkl` atau `cache/cluster_annotations_*.pkl` yang sesuai.

---

## 8. Pemecahan masalah

| Gejala | Solusi |
|---|---|
| `AZURE_OPENAI_API_KEY is not set` | Isi kunci di `.env` atau environment. |
| `Column '...' not found` | CLI mencetak kolom tersedia; pilih salah satu. |
| `Column count mismatch on lines: ...` | Tanda kutip belum ditutup atau koma di nilai tanpa kutip. |
| Semua kandidat ditolak oleh filter noise | Filter otomatis dilonggarkan dengan peringatan. |
| Skor `poor` (< 0.20) | Coba teks lebih kaya atau periksa representatif mentah. |
| `hdbscan` gagal dipasang di Windows | `pip install hdbscan --only-binary=:all:` |
| Leiden dilewati | `pip install hnswlib python-igraph leidenalg` |

---

## 9. Catatan privasi

- CSV masukan dapat berisi PII. `data/input/` dan `data/output/` sudah
  dalam `.gitignore`.
- Pipeline mengirim teks ke Azure OpenAI Embeddings dan (opsional) Chat Completions.
  Sembunyikan data sensitif secara lokal sebelum memproses.
- Folder `cache/` menyimpan embedding dan label/ringkasan hasil LLM.
  Perlakukan dengan standar perlindungan yang sama seperti CSV asli.
