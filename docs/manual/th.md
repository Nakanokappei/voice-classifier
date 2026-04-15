# voice-classifier — คู่มือผู้ใช้ (ภาษาไทย)

เครื่องมือ command-line ที่ใช้จัดกลุ่ม (cluster) ไฟล์ CSV ที่บันทึก
เสียงของลูกค้า (ตั๋วสนับสนุน บันทึกการซ่อม ฯลฯ) ติดป้ายกำกับแต่ละกลุ่ม
ด้วย LLM และสร้างรายงานในรูปแบบที่ทั้งคนและเครื่องอ่านได้

---

## 1. ทำอะไรได้บ้าง

เมื่อรับไฟล์ CSV ที่แต่ละแถวมีข้อความอิสระของลูกค้า:

1. คำนวณ embedding สำหรับแต่ละแถวที่ไม่ซ้ำ ด้วยโมเดลของ OpenAI
2. ไล่สำรวจการตั้งค่าผู้สมัคร (KMeans / HDBSCAN / Leiden) แล้วเลือกตัวที่ดีที่สุด
   ตาม cosine silhouette
3. ดึงแถวที่ใกล้ centroid ของแต่ละกลุ่มมาเป็นตัวแทนดิบ
4. ขอให้ LLM สร้างป้ายกำกับสั้นและสรุปสำหรับแต่ละกลุ่ม โดยอิงจาก
   บริบทของชุดข้อมูลที่ถูกอนุมานไว้ก่อน
5. แก้ไขป้ายกำกับที่ซ้ำกันด้วยการสร้างป้ายกลุ่มเล็กให้แตกต่าง
6. เขียนรายงานเป็น Markdown, HTML และ CSV

---

## 2. การติดตั้ง

```bash
# ต้องใช้ Python 3.10 หรือใหม่กว่า
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # แก้ไข .env แล้วใส่ OPENAI_API_KEY
```

Backend เสริม (ถูกข้ามโดยปลอดภัยเมื่อไม่ได้ติดตั้ง):

- `hdbscan` — การจัดกลุ่มเชิงความหนาแน่นแบบเร็ว
- `hnswlib` + `python-igraph` + `leidenalg` — การจัดกลุ่ม Leiden บนกราฟ

---

## 3. การรันครั้งแรก

```bash
python src/pipeline.py \
    --input data/input/sample.csv \
    --text-col "response_body"
```

ขั้นตอน:

1. อ่าน CSV และปรับข้อความ (NFKC, รวบช่องว่าง, ตัดรายการซ้ำ)
2. ดึง / แคช embedding สำหรับแต่ละแถวที่ไม่ซ้ำ
3. ค้นหาการตั้งค่าการจัดกลุ่มที่ดีที่สุด
4. ดึง 5 แถวที่ใกล้ centroid มากที่สุดของแต่ละกลุ่ม
5. ถ้าไม่ระบุ `--no-name-clusters` จะอนุมานบริบทชุดข้อมูล สร้างป้ายกำกับ
   แบบขนาน แล้วแก้ไขชื่อซ้ำ
6. เขียนผลลัพธ์ไปยัง `data/output/YYYYMMDD_HHMMSS/`

---

## 4. การเลือกคอลัมน์

### คอลัมน์เดียว

```bash
python src/pipeline.py --input tickets.csv --text-col "response_body"
```

### หลายคอลัมน์ (เชื่อมเป็นรูปแบบ `label: value`)

```bash
python src/pipeline.py --input tickets.csv \
    --text-cols "Ticket Subject,Ticket Description" \
    --column-labels "Ticket Subject=subject,Ticket Description=body"
```

### การเลือกแบบโต้ตอบ

ถ้าไม่ระบุทั้งสองค่า CLI จะแสดงคอลัมน์ผู้สมัครพร้อมคะแนน และขอให้คุณเลือก:

```bash
python src/pipeline.py --input tickets.csv
```

---

## 5. โครงสร้างของผลลัพธ์

แต่ละการรันจะสร้างโฟลเดอร์ที่มี timestamp:

```
data/output/20260416_012345/
├── report.md                           ผลการจัดกลุ่มสำหรับคน
├── report.html                          เหมือนกันในรูปแบบ HTML (--format html/both)
├── parameter_search.html                รายงานการค้นหาพร้อมกราฟด้านบน
├── clusters.csv                         หนึ่งแถวต่อหนึ่งกลุ่ม: id, name, size,
│                                       summary, rep_1..N
├── <input>_classified.csv               แถวต้นฉบับ + cluster_id (+ cluster_name)
├── params.json                          เมตะดาต้าสำหรับเครื่องอ่าน
└── run.log                              log ระดับ INFO ของการรัน
```

### คอลัมน์ของ `clusters.csv`

- `cluster_id` — จำนวนเต็ม, `-1` คือ noise
- `cluster_name` — ป้ายกำกับสั้นจาก LLM (เมื่อเปิด `--name-clusters`)
- `size` — จำนวนแถว
- `summary` — สรุปจาก LLM (เงื่อนไขเดียวกัน)
- `rep_1` ... `rep_N` — แถวต้นฉบับที่ใกล้ centroid

---

## 6. ตัวเลือกหลัก

| ตัวเลือก | ค่าเริ่มต้น | หน้าที่ |
|---|---|---|
| `--input PATH` | จำเป็น | พาธไฟล์ CSV |
| `--text-col NAME` | — | โหมดคอลัมน์เดียว |
| `--text-cols A,B` | — | โหมดหลายคอลัมน์ |
| `--column-labels A=x,B=y` | — | ป้ายกำกับในโหมดหลายคอลัมน์ |
| `--output-dir PATH` | `data/output` | โฟลเดอร์รากของผลลัพธ์ |
| `--cache-dir PATH` | `cache` | โฟลเดอร์แคช |
| `--model NAME` | `text-embedding-3-small` | โมเดล embedding |
| `--top-k N` | `5` | จำนวนตัวแทนต่อกลุ่ม |
| `--min-clusters N` | `2` | ขอบล่างของ K |
| `--max-clusters N` | `20` | ขอบบนของ K |
| `--name-clusters` / `--no-name-clusters` | เปิด | เปิด/ปิดการติดป้ายกำกับด้วย LLM |
| `--name-model NAME` | `gpt-5.4-nano` | โมเดล chat สำหรับติดป้าย |
| `--format md|html|both` | `md` | รูปแบบของ `report.*` |
| `--log-level LEVEL` | `INFO` | ระดับรายละเอียดใน stderr |

---

## 7. การตั้งค่า

### ตัวแปรสภาพแวดล้อม (`.env`)

- `OPENAI_API_KEY` (จำเป็น)
- `OPENAI_EMBEDDING_MODEL` (override ตามใจ)
- `OPENAI_REQUEST_TIMEOUT` (วินาที ค่าเริ่มต้น 60)

### แคช

`cache/` จัดเก็บเวกเตอร์ embedding และคำอธิบาย LLM ตาม hash ของเนื้อหา
การเปลี่ยนโมเดลจะเขียนลงไฟล์คนละไฟล์ หากต้องการบังคับสร้างใหม่
ให้ลบไฟล์ `cache/embeddings_*.pkl` หรือ `cache/cluster_annotations_*.pkl`
ที่สอดคล้องกัน

---

## 8. การแก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `OPENAI_API_KEY is not set` | กรอกคีย์ใน `.env` หรือในตัวแปรสภาพแวดล้อม |
| `Column '...' not found` | CLI จะแสดงคอลัมน์ที่ใช้ได้; เลือกจากรายการนั้น |
| `Column count mismatch on lines: ...` | อัญประกาศไม่ปิด หรือมีคอมมาในค่าที่ไม่ได้ใส่อัญประกาศ |
| ตัวเลือกทั้งหมดถูกกรองด้วยอัตราส่วน noise | ตัวกรองจะผ่อนปรนเองพร้อมคำเตือน |
| คะแนน `poor` (< 0.20) | ลองใช้ข้อความที่เนื้อหาหลากหลายขึ้น หรือตรวจตัวแทนเองด้วยมือ |
| ติดตั้ง `hdbscan` ใน Windows ล้มเหลว | `pip install hdbscan --only-binary=:all:` |
| Leiden ถูกข้าม | `pip install hnswlib python-igraph leidenalg` |

---

## 9. หมายเหตุด้านความเป็นส่วนตัว

- CSV ข้อมูลเข้าอาจมีข้อมูลส่วนบุคคล `data/input/` และ `data/output/`
  อยู่ใน `.gitignore` แล้ว
- ไพพ์ไลน์ส่งข้อความไปยัง OpenAI Embeddings และ (ทางเลือก) Chat Completions
  โปรดปิดบังข้อมูลอ่อนไหวในเครื่องก่อนประมวลผล
- โฟลเดอร์ `cache/` เก็บ embedding และป้ายกำกับ/สรุปที่สร้างโดย LLM
  กรุณาดูแลระดับเดียวกับ CSV ต้นฉบับ
