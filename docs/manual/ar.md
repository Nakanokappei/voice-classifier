<div dir="rtl" lang="ar">

# voice-classifier — دليل المستخدم (العربية)

أداة سطر أوامر تُصنّف ملفات CSV التي تحتوي على أصوات العملاء (تذاكر الدعم،
سجلات الإصلاح، وغيرها) إلى مجموعات، وتضع لكل مجموعة تسمية وملخصًا عبر نموذج
لغوي كبير (LLM)، ثم تُنتج تقارير مقروءة من قِبَل البشر والآلة.

---

## ١. ما الذي تفعله الأداة

لملف CSV تحتوي صفوفه على نص حرّ من العملاء:

1. حساب تمثيلات (embeddings) لكل صف فريد باستخدام نموذج OpenAI.
2. تجربة تكوينات للتجميع (KMeans / HDBSCAN / Leiden) واختيار الأفضل وفق
   مقياس silhouette القائم على cosine.
3. استخراج أقرب الصفوف إلى مركز كل مجموعة بوصفها ممثلات أولية.
4. طلب تسمية قصيرة وملخص لكل مجموعة من النموذج اللغوي، مع ترسيخ السياق
   العام للمجموعة المستنتج مسبقًا.
5. معالجة التسميات المتكررة بإعادة توليد التسمية للمجموعات الأصغر.
6. كتابة تقارير Markdown وHTML وCSV.

---

## ٢. التثبيت

```bash
# يتطلب Python 3.10 أو أحدث
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # عدّل .env لإدخال OPENAI_API_KEY
```

خلفيات اختيارية (يجري تجاوزها تلقائيًا إن كانت مفقودة):

- `hdbscan` — تجميع سريع قائم على الكثافة.
- `hnswlib` + `python-igraph` + `leidenalg` — تجميع Leiden عبر رسم بياني.

---

## ٣. أول تشغيل

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

ما الذي يحدث:

1. قراءة CSV، تطبيع النص (NFKC، تقليص المسافات، إزالة التكرار).
2. جلب / تخزين embeddings لكل صف فريد.
3. البحث عن أفضل تكوين للتجميع.
4. استخراج أقرب ٥ صفوف إلى مركز كل مجموعة.
5. ما لم يُمرَّر `--no-name-clusters`، يُستنتج سياق البيانات، ثم تُولَّد
   التسميات بالتوازي، ثم تُحلُّ التسميات المتكررة.
6. تُكتب النتائج في `data/output/YYYYMMDD_HHMMSS/`.

---

## ٤. اختيار الأعمدة

### عمود واحد

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### أعمدة متعددة (تُدمج بصيغة `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### اختيار تفاعلي

عند إغفال كلا الخيارين، تعرض الأداة قائمة مرشحة مقيَّمة وتطلب منك الاختيار:

```bash
python src/pipeline.py --input tickets.csv
```

---

## ٥. بنية المخرجات

يُنشأ لكل تشغيل مجلد موسوم بالتاريخ والوقت:

```
data/output/20260416_012345/
├── report.md                           نتيجة التجميع للقراء البشر
├── report.html                          نفس المحتوى بصيغة HTML (عند --format html/both)
├── parameter_search.html                تقرير البحث مع رسم بياني في الأعلى
├── clusters.csv                         صف واحد لكل مجموعة: id، name، size،
│                                       summary، rep_1..N
├── <input>_classified.csv               الصفوف الأصلية + cluster_id (+ cluster_name)
├── params.json                          بيانات وصفية بصيغة يقرؤها الحاسوب
└── run.log                              سجل التنفيذ (INFO وأعلى)
```

### أعمدة `clusters.csv`

- `cluster_id` — عدد صحيح، `-1` للضجيج.
- `cluster_name` — تسمية قصيرة من النموذج (فقط عند `--name-clusters`).
- `size` — عدد الصفوف في المجموعة.
- `summary` — ملخص من النموذج (بنفس الشرط).
- `rep_1` ... `rep_N` — الصفوف الأصلية الأقرب إلى المركز.

---

## ٦. أهم الخيارات

| الخيار | الافتراضي | الغرض |
|---|---|---|
| `--input PATH` | إلزامي | مسار ملف CSV المدخل |
| `--text-col NAME` | — | عمود واحد للتضمين |
| `--text-cols A,B` | — | عدة أعمدة مدموجة |
| `--column-labels A=x,B=y` | — | تسميات لوضع الأعمدة المتعددة |
| `--output-dir PATH` | `data/output` | مجلد الإخراج الجذر |
| `--cache-dir PATH` | `cache` | مجلد الذاكرة المؤقتة |
| `--model NAME` | `text-embedding-3-small` | نموذج التضمين |
| `--top-k N` | `5` | عدد الصفوف الممثِّلة لكل مجموعة |
| `--min-clusters N` | `2` | الحد الأدنى لقيمة K |
| `--max-clusters N` | `20` | الحد الأعلى لقيمة K |
| `--target faq|chatbot|insight` | `faq` | تحديد الدقة حسب الاستخدام. `faq`=30-80 مجموعة (صفحات الأسئلة الشائعة)، `chatbot`=50-150 (نوايا)، `insight`=أعلى silhouette |
| `--name-clusters` / `--no-name-clusters` | مفعَّل | تشغيل/إيقاف التسمية عبر LLM |
| `--name-model NAME` | `gpt-5.4-nano` | نموذج الدردشة للتسمية |
| `--advise` / `--no-advise` | on | ملاحظة استشارية من LLM في أعلى `parameter_search.html` |
| `--advisor-model NAME` | `gpt-5.4` | نموذج الدردشة للملاحظة الاستشارية (يحلل التشغيل بالكامل) |
| `--format md|html|both` | `md` | صيغة `report.*` |
| `--log-level LEVEL` | `INFO` | مستوى التفاصيل على stderr |

---

## ٧. الإعدادات

### متغيرات البيئة (`.env`)

- `OPENAI_API_KEY` (إلزامي)
- `OPENAI_EMBEDDING_MODEL` (تجاوز اختياري)
- `OPENAI_REQUEST_TIMEOUT` (بالثواني، الافتراضي 60)

### الذاكرة المؤقتة

يخزن `cache/` متجهات التضمين وتعليقات LLM وفق تجزئة المحتوى. عند تغيير
النموذج يُستخدم ملف مختلف. لإجبار إعادة التوليد، احذف الملف المناسب في
`cache/embeddings_*.pkl` أو `cache/cluster_annotations_*.pkl`.

---

## ٨. حل المشكلات

| العَرَض | الحل |
|---|---|
| `OPENAI_API_KEY is not set` | ضع المفتاح في `.env` أو في متغير البيئة. |
| `Column '...' not found` | تعرض الأداة الأعمدة المتوفرة؛ اختر منها. |
| `Column count mismatch on lines: ...` | علامات اقتباس غير مغلقة أو فواصل داخل قيم غير مقتبسة. |
| استُبعد كل المرشحين بسبب نسبة الضجيج | يتخفف الفلتر تلقائيًا مع تحذير. |
| درجة `poor` (< 0.20) | جرّب نصًّا أغنى أو افحص الممثلات يدويًا. |
| فشل تثبيت `hdbscan` على Windows | `pip install hdbscan --only-binary=:all:` |
| تم تجاوز Leiden | `pip install hnswlib python-igraph leidenalg` |

---

## ٩. ملاحظات الخصوصية

- قد تحتوي ملفات CSV المُدخَلة على بيانات شخصية. المجلدان `data/input/` و
  `data/output/` مُدرَجان في `.gitignore`.
- يُرسل الأنبوب النص إلى OpenAI Embeddings، واختياريًا إلى Chat Completions.
  أخفِ البيانات الحساسة محليًا قبل المعالجة.
- يحفظ `cache/` متجهات التضمين والتسميات/الملخصات المولَّدة. عامِلها بنفس
  مستوى الحماية التي تعامل بها ملف CSV الأصلي.

</div>
