# Internship Recommender Prototype

This repository implements a **student ↔ internship recommender system** with a modular ML pipeline and a FastAPI service.  
It supports **data ingestion, cleaning, augmentation, featurization, and recommendation generation** in three modes: **Baseline, Gap-Analysis, Weighted**.

---

## 📂 Project Structure

```
ml_engine/
├── api/               # FastAPI app (app.py)
├── data/
│   ├── raw/           # Original datasets (students_.csv, internships_.csv)
│   ├── cleaned/       # Cleaned datasets
│   └── processed/     # Augmented + recommendations outputs
└── steps/             # Modular ML pipeline steps
    ├── ingest.py          # Load raw data
    ├── data_cleaning.py   # Standardize + clean raw data
    ├── augment.py         # Augmentation (rebalancing, noise, skill injection)
    └── featurize.py       # TF-IDF vectorization & recommendation

pipeline.py                # Master pipeline runner
verify_seed.py             # Script to check reproducibility of augmentation
show_ids.py                # Helper to list student IDs to test in API
Makefile                   # Shortcuts to run/stop API
requirements.txt           # Dependencies
README.md                  # Project documentation
```

---

## ⚙️ Setup

1. **Clone repo & create virtual environment**
   ```bash
   git clone <repo-url>
   cd 01_myjob
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure raw data is available**
   Place your original CSVs inside `ml_engine/data/raw/`:
   - `students_.csv`
   - `internships_.csv`

---

## 🚀 Usage

### Run the full pipeline

#### 1. Baseline (skills-only matching)
```bash
python pipeline.py --mode baseline --top_k 5
```
- Cleans data → saves to `ml_engine/data/cleaned/`
- Runs TF-IDF skill matching
- Outputs baseline recommendations → `ml_engine/data/processed/matches.csv`

#### 2. Gap Analysis (skills + missing gap info)
```bash
python pipeline.py --mode gap --top_k 10
```
- Adds missing skills + missing % info
- Saves results in:
  - `matches_with_gaps.csv` (flattened)
  - `matches_with_gaps_nested.csv` (nested)

#### 3. Weighted Hybrid Matching
```bash
python pipeline.py --mode weighted --top_k 7
```
- Combines baseline + gap + extra weights
- Outputs → `matches_weighted.csv`

#### 4. With augmentation + seed for reproducibility
```bash
python pipeline.py --mode weighted --augment --seed 42
```
- Runs augmentation pipeline:
  - `students_rebalanced.csv`
  - `students_augmented_skills.csv`
  - `internships_rebalanced.csv`
- Guarantees same augmented data on re-runs

---

## 📊 Outputs

- **Cleaned Datasets:**
  - `ml_engine/data/cleaned/students_cleaned.csv`
  - `ml_engine/data/cleaned/internships_cleaned.csv`

- **Augmented Datasets (if `--augment`):**
  - `ml_engine/data/processed/students_rebalanced.csv`
  - `ml_engine/data/processed/students_augmented_skills.csv`
  - `ml_engine/data/processed/internships_rebalanced.csv`

- **Recommendations:**
  - `ml_engine/data/processed/matches.csv` (baseline)
  - `ml_engine/data/processed/matches_with_gaps.csv` (gap flat)
  - `ml_engine/data/processed/matches_with_gaps_nested.csv` (gap nested)
  - `ml_engine/data/processed/matches_weighted.csv` (weighted)

---

## 🌐 API (FastAPI)

Run API with:
```bash
uvicorn ml_engine.api.app:app --reload --port 8001
```

Base URL: [http://127.0.0.1:8001](http://127.0.0.1:8001)

- **Swagger Docs (interactive dashboard):**
  [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

- **ReDoc:**
  [http://127.0.0.1:8001/redoc](http://127.0.0.1:8001/redoc)

### API Endpoints

1. **Home**
```http
GET /
```
Response:
```json
{"message": "🎓 Internship Recommendation API with Baseline, Gap-Analysis, and Weighted results is running!"}
```

2. **Get Recommendations**
```http
POST /recommend
```
Request Body:
```json
{
  "student_id": "S00001",
  "top_n": 5
}
```
Response (sample):
```json
{
  "student_id": "S00001",
  "recommendations": {
    "baseline": [...],
    "gap_analysis": [...],
    "weighted": [...]
  }
}
```

3. **Check available student data**
```http
GET /check/{student_id}
```
Example:
```http
GET /check/S00005
```

---

## 🛠️ Utilities

### Verify deterministic augmentation
```bash
python verify_seed.py
```

### Show available student IDs for testing
```bash
python show_ids.py
```
Output:
```
✅ Found 5000 student IDs in ml_engine/data/processed/matches.csv
🔹 Sample Student IDs you can test:
['S00001', 'S00002', 'S00003', ...]
```

---

## ⚡ Makefile Shortcuts

- **Run API (kills old server first):**
```bash
make run-api
```

- **Stop API manually:**
```bash
make stop-api
```

---

## 📦 Requirements

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.2
scikit-learn==1.5.1
fastapi==0.115.0
uvicorn==0.30.6
jupyter==1.1.1
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 📝 Example Pipeline Output

```bash
python pipeline.py --mode baseline --top_k 3
```

```
  student_id                                        top_matches
0     S00001  [(J01063, 1.0), (J02586, 0.8936), (J02465, 0.8207)]
1     S00002  [(J01459, 0.8577), (J04299, 0.8557), (J03390, 0.7699)]
2     S00003  [(J04757, 0.8344), (J03288, 0.8652), (J00638, 0.8677)]
```

---

## 🔮 Next Steps

- Extend featurization to include **location, qualifications, and sectors**
- Build a **frontend dashboard** for users to enter student_id and view results
- Deploy API on **Render / Railway / EC2** for public access

