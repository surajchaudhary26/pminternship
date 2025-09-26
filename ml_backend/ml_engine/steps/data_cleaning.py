# ml_engine/steps/data_cleaning.py
"""
Data Cleaning Module
--------------------
Cleans student & internship DataFrames:
- standardize column names
- parse list-like strings to Python lists (skills, cities, tags, etc.)
- normalize experience fields to integers (months)
- drop duplicates and trivial empty rows

Usage:
    from data_cleaning import clean_students, clean_internships
    students_clean = clean_students(students_df)
    internships_clean = clean_internships(internships_df)
"""

from typing import List
from pathlib import Path
import pandas as pd
import ast
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def _safe_parse_list(value):
    """Convert list-like string to Python list. Return empty list if not parsable."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    val = value.strip()
    # handle JSON-like or Python list strings
    try:
        if val.startswith("[") and val.endswith("]"):
            parsed = ast.literal_eval(val)
            return [str(x).strip() for x in parsed] if isinstance(parsed, (list, tuple)) else []
    except Exception:
        pass
    # fallback: comma-separated string
    if "," in val:
        return [p.strip() for p in val.split(",") if p.strip()]
    # single token
    if val:
        return [val]
    return []


def _extract_int_from_str(s: str) -> int:
    if pd.isna(s):
        return 0
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return 0


def clean_students(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    df = df.copy()
    df = _standardize_columns(df)
    logging.info("Cleaning students: standardizing columns & dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    # Columns commonly present in your dataset
    list_cols = ["skills", "location_preferences", "sector_interests"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # Experience numeric (months)
    if "experience" in df.columns:
        df["experience"] = df["experience"].apply(_extract_int_from_str).fillna(0).astype(int)
    else:
        df["experience"] = 0

    # Normalize boolean-ish fields
    for c in ["past_internship_participation"]:
        if c in df.columns:
            df[c] = df[c].astype("boolean")

    # Fill missing essential fields with sensible defaults
    if "student_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "student_id"})

    # optional save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logging.info(f"Saved cleaned students to {save_path}")

    return df


def clean_internships(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    df = df.copy()
    df = _standardize_columns(df)
    logging.info("Cleaning internships: standardizing columns & dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    list_cols = ["skills_required", "cities", "states", "diversity_preferences", "tags"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # Experience required numeric (months)
    if "experience_required" in df.columns:
        df["experience_required"] = df["experience_required"].apply(_extract_int_from_str).fillna(0).astype(int)
    else:
        df["experience_required"] = 0

    # Convert posted_date to datetime if present
    if "posted_date" in df.columns:
        try:
            df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
        except Exception:
            df["posted_date"] = pd.NaT

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logging.info(f"Saved cleaned internships to {save_path}")

    return df


if __name__ == "__main__":
    try:
        from ingest import load_data
        students_raw, internships_raw = load_data()

        students_clean = clean_students(
            students_raw,
            save_path="ml_engine/data/cleaned/students_cleaned.csv"
        )
        internships_clean = clean_internships(
            internships_raw,
            save_path="ml_engine/data/cleaned/internships_cleaned.csv"
        )

        print("✔ Students cleaned shape:", students_clean.shape)
        print("✔ Internships cleaned shape:", internships_clean.shape)
        print("✔ Cleaned files saved in ml_engine/data/cleaned/")
    except Exception as e:
        logging.error("Run failed: %s", e)

