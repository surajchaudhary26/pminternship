# Internship Recommender Prototype

This repository implements a **student ↔ internship recommender system**.  
The pipeline is modular, professional, and designed for **data ingestion, cleaning, augmentation, feature extraction, and recommendation generation**.

---

## 📂 Project Structure

```
ml_engine/
├── data/
│   ├── raw/          # Original untouched datasets (students_.csv, internships_.csv)
│   ├── cleaned/      # Cleaned datasets (students_cleaned.csv, internships_cleaned.csv)
│   └── processed/    # Augmented / noisy datasets for robustness testing
└── steps/            # Modular ML pipeline steps
    ├── ingest.py          # Load raw data
    ├── data_cleaning.py   # Standardize + clean raw data
    ├── augment.py         # Augmentation (rebalancing, noise, skill injection)
    └── featurize.py       # TF-IDF vectorization & recommendation

pipeline.py                # Master pipeline runner
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

#### 1. Default (clean only)
```bash
python pipeline.py
```
- Loads raw datasets  
- Cleans & saves to `ml_engine/data/cleaned/`  
- Generates recommendations (TF-IDF similarity)  

#### 2. With augmentation
```bash
python pipeline.py --augment
```
- Cleans raw datasets  
- Augments students + internships (boost skills, add noise) → `ml_engine/data/processed/`  
- Still runs featurization on **clean data**  
- Augmented datasets are available separately for robustness testing  

#### 3. Control number of top matches
```bash
python pipeline.py --top_k 3
```
- Returns top-3 internship matches per student  

---

## 📊 Outputs

- **Cleaned Datasets (always generated):**
  - `ml_engine/data/cleaned/students_cleaned.csv`
  - `ml_engine/data/cleaned/internships_cleaned.csv`

- **Augmented Datasets (if `--augment`):**
  - `ml_engine/data/processed/students_rebalanced.csv`
  - `ml_engine/data/processed/students_augmented_skills.csv`
  - `ml_engine/data/processed/internships_rebalanced.csv`

- **Recommendations:**  
  Printed in terminal (format: `student_id → [(job_id, similarity_score), ...]`).

---

## 📦 Requirements

Minimal dependencies:

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
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

## ⚠️ Notes

- Always keep `raw/` datasets untouched (backup).  
- Use `cleaned/` datasets for training, evaluation, and main pipeline.  
- Use `processed/` datasets only for augmentation / robustness experiments.  
- Any Pandas `FutureWarning` about dtypes is safe to ignore (not breaking).  

---

## 📝 Example Output

```bash
python pipeline.py --top_k 3
```

Sample:

```
  student_id                                        top_matches
0     S00001  [(J01063, 1.0), (J02586, 0.8936), (J02465, 0.8207)]
1     S00002  [(J01459, 0.8577), (J04299, 0.8557), (J03390, 0.7699)]
2     S00003  [(J04757, 0.8344), (J03288, 0.8652), (J00638, 0.8677)]
```

---

## 🔮 Next Steps

- Extend featurization to include **location preferences, qualifications, and sectors**.  
- Add a web frontend to serve recommendations dynamically.  
- Integrate with a database for real-time updates.  
