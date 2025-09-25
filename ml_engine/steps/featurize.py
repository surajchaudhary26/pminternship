# ml_engine/steps/featurize.py
"""
Feature Engineering Module
--------------------------
- parse_skills: safely parse skills fields into space-joined tokens
- vectorize_skills: TF-IDF vectorizer (sparse)
- compute_similarity: cosine similarity
- match_students_to_internships: helper that returns top-k internship matches for each student

Usage:
    from featurize import match_students_to_internships
    matches_df = match_students_to_internships(students_df, internships_df, top_k=5)
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
import ast
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_list_to_text(value) -> str:
    """Return a space-joined lowercase token string from list or list-like string."""
    if isinstance(value, list):
        tokens = [str(x).strip().lower().replace(" ", "_") for x in value if x is not None]
        return " ".join(tokens)
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        s = value.strip()
        # try literal_eval
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return " ".join([str(x).strip().lower().replace(" ", "_") for x in parsed])
            except Exception:
                pass
        # fallback: comma-separated
        if "," in s:
            parts = [p.strip().lower().replace(" ", "_") for p in s.split(",") if p.strip()]
            return " ".join(parts)
        return s.lower().replace(" ", "_")
    return ""


def parse_skills_column(series: pd.Series) -> pd.Series:
    """Convert a Series of skill lists/strings into cleaned text tokens for vectorization."""
    return series.fillna("").apply(_safe_list_to_text)


def vectorize_skills(texts: List[str], max_features: int = 5000) -> Tuple:
    """Fit TF-IDF on texts and return (vectors, vectorizer)."""
    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    vectors = vectorizer.fit_transform(texts)
    logging.info("TF-IDF vectorized input with shape %s", vectors.shape)
    return vectors, vectorizer


def compute_similarity(vectors_a, vectors_b=None) -> np.ndarray:
    """
    Compute cosine similarity.
    - if vectors_b is None: compute pairwise within vectors_a (n x n)
    - else compute similarity between vectors_a and vectors_b (n_a x n_b)
    """
    if vectors_b is None:
        sim = cosine_similarity(vectors_a)
    else:
        sim = cosine_similarity(vectors_a, vectors_b)
    return sim


def match_students_to_internships(students_df: pd.DataFrame,
                                  internships_df: pd.DataFrame,
                                  student_skills_col: str = "skills",
                                  internship_skills_col: str = "skills_required",
                                  top_k: int = 5,
                                  max_features: int = 5000) -> pd.DataFrame:
    """
    Return a DataFrame with columns: student_id, top_matches [(job_id, score)...]
    Each top_matches is a list of tuples (job_id, score).
    """
    s_df = students_df.copy()
    j_df = internships_df.copy()

    # prepare text
    s_text = parse_skills_column(s_df.get(student_skills_col, pd.Series([""] * len(s_df))))
    j_text = parse_skills_column(j_df.get(internship_skills_col, pd.Series([""] * len(j_df))))

    # combine to fit vectorizer (so vocab consistent)
    combined = pd.concat([s_text, j_text], ignore_index=True)
    vectors, vectorizer = vectorize_skills(combined.tolist(), max_features=max_features)

    s_vec = vectors[:len(s_df)]
    j_vec = vectors[len(s_df):]

    sim = compute_similarity(s_vec, j_vec)  # shape (n_students, n_jobs)

    results = []
    job_ids = list(j_df.get("job_id", j_df.index.astype(str)))
    for i, student_id in enumerate(s_df.get("student_id", s_df.index.astype(str))):
        row_scores = sim[i]
        top_idx = np.argsort(row_scores)[-top_k:][::-1]
        matches = [(job_ids[idx], float(row_scores[idx])) for idx in top_idx]
        results.append({"student_id": student_id, "top_matches": matches})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # quick demo (requires data)
    try:
        from ingest import load_data
        from data_cleaning import clean_students, clean_internships

        students_raw, internships_raw = load_data()
        students = clean_students(students_raw)
        internships = clean_internships(internships_raw)

        matches = match_students_to_internships(students, internships, top_k=3)
        print(matches.head())
    except Exception as e:
        logging.error("Demo failed: %s", e)
