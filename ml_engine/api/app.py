from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import ast

from ml_engine.steps.featurize import (
    match_students_to_internships,
    match_students_to_internships_with_gaps,
    flatten_matches_df,
)
from ml_engine.steps.ingest import load_data
from ml_engine.steps.data_cleaning import clean_students, clean_internships

app = FastAPI(title="Internship Recommendation API", version="2.0")

# --- Data Loading (on startup) ---
students_raw, internships_raw = load_data()
students_clean = clean_students(students_raw)
internships_clean = clean_internships(internships_raw)

# Load already processed recommendation CSVs (saved by pipeline.py)
processed_dir = Path("ml_engine/data/processed")
matches_base = pd.read_csv(processed_dir / "matches.csv")
matches_gap = pd.read_csv(processed_dir / "matches_with_gaps.csv")
matches_weighted = pd.read_csv(processed_dir / "matches_weighted.csv")


# --- Request Models ---
class StudentRequest(BaseModel):
    student_id: str
    top_n: Optional[int] = 5


# --- Utility for match labels ---
def get_match_label(score: float, mode: str = "baseline") -> str:
    if mode == "baseline":
        return "👍 Good Match" if score >= 0.7 else "🙂 Average"
    if mode == "weighted":
        if score >= 0.7:
            return "🔥 Strong Match"
        elif score >= 0.3:
            return "⚠️ Weak Match"
        else:
            return "❌ Poor Match"
    return ""


# --- Routes ---
@app.get("/")
def home():
    return {"message": "🎓 Internship Recommendation API with Baseline, Gap-Analysis, and Weighted results is running!"}


@app.post("/recommend")
def recommend(data: StudentRequest):
    sid = data.student_id
    top_n = data.top_n

    response = {"student_id": sid, "recommendations": {}}

    # Baseline
    row = matches_base[matches_base["student_id"] == sid]
    if not row.empty:
        recs = ast.literal_eval(row.iloc[0]["top_matches"])
        baseline_recs = [
            {
                "job_id": jid,
                "score": round(score, 4),
                "label": get_match_label(score, "baseline"),
            }
            for jid, score in recs[:top_n]
        ]
        response["recommendations"]["baseline"] = baseline_recs

    # Gap-Analysis
    gap_recs = matches_gap[matches_gap["student_id"] == sid].head(top_n)
    if not gap_recs.empty:
        gap_list = []
        for _, r in gap_recs.iterrows():
            gap_list.append(
                {
                    "job_id": r["job_id"],
                    "score": round(r["score"], 4),
                    "missing_skills": r["missing_skills"].split(";") if isinstance(r["missing_skills"], str) else [],
                    "missing_pct": r["missing_pct"],
                }
            )
        response["recommendations"]["gap_analysis"] = gap_list

    # Weighted
    row_w = matches_weighted[matches_weighted["student_id"] == sid]
    if not row_w.empty:
        recs = ast.literal_eval(row_w.iloc[0]["top_matches"])
        weighted_recs = [
            {
                "job_id": jid,
                "score": round(score, 4),
                "label": get_match_label(score, "weighted"),
            }
            for jid, score in recs[:top_n]
        ]
        response["recommendations"]["weighted"] = weighted_recs

    return response
