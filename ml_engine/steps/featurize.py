# ml_engine/steps/featurize.py
"""
Feature Engineering Module
--------------------------
- parse_skills: safely parse skills fields into space-joined tokens
- vectorize_skills: TF-IDF vectorizer (sparse)
- compute_similarity: cosine similarity
- match_students_to_internships: helper that returns top-k internship matches for each student
- match_students_to_internships_with_gaps: extended version with skill-gap analysis
- flatten_matches_df: utility to flatten nested matches into a clean CSV

Usage:
    from featurize import match_students_to_internships, match_students_to_internships_with_gaps
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
import ast
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------
# Helpers for parsing
# -------------------------
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


# -------------------------
# Basic matching (as before)
# -------------------------
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
    vectors, _ = vectorize_skills(combined.tolist(), max_features=max_features)

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


# -------------------------
# Gap-analysis version
# -------------------------
def _safe_parse_list(value) -> List[str]:
    """Convert various list-like inputs into a Python list of strings."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None]
    if pd.isna(value):
        return []
    s = str(value).strip()
    # try literal eval
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if x is not None]
        except Exception:
            pass
    # comma-separated fallback
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s] if s else []


def _norm_tokens(tokens: List[str]) -> List[str]:
    """Normalize tokens: lower, strip, replace spaces with underscore."""
    return [t.strip().lower().replace(" ", "_") for t in tokens if t and str(t).strip()]


def _skills_to_set(value) -> set:
    """Return normalized skill set from a field (list-like or string)."""
    return set(_norm_tokens(_safe_parse_list(value)))


def _list_to_text(tokens: List[str]) -> str:
    """Convert token list to space-joined string for TF-IDF."""
    return " ".join(_norm_tokens(tokens))


def match_students_to_internships_with_gaps(
    students_df: pd.DataFrame,
    internships_df: pd.DataFrame,
    student_skills_col: str = "skills",
    internship_skills_col: str = "skills_required",
    student_id_col: str = "student_id",
    internship_id_col: str = "job_id",
    top_k: int = 5,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Extended matching: return top_k internships plus missing skills info.
    Each top_matches is a list of dicts:
    {job_id, score, missing_skills, missing_count, missing_pct}
    """
    s_df = students_df.copy().reset_index(drop=True)
    j_df = internships_df.copy().reset_index(drop=True)

    # Prepare normalized skill text
    s_skill_lists = [_safe_parse_list(x) for x in s_df.get(student_skills_col, pd.Series([""] * len(s_df)))]
    j_skill_lists = [_safe_parse_list(x) for x in j_df.get(internship_skills_col, pd.Series([""] * len(j_df)))]

    s_text = [_list_to_text(lst) for lst in s_skill_lists]
    j_text = [_list_to_text(lst) for lst in j_skill_lists]

    # TF-IDF
    combined = s_text + j_text
    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    vectors = vectorizer.fit_transform(combined)
    logging.info("TF-IDF vectorized input with shape %s", vectors.shape)

    s_vec = vectors[:len(s_df)]
    j_vec = vectors[len(s_df):]

    sim = cosine_similarity(s_vec, j_vec)

    student_skill_sets = [_skills_to_set(x) for x in s_skill_lists]
    job_skill_sets = [_skills_to_set(x) for x in j_skill_lists]

    results = []
    job_ids = list(j_df.get(internship_id_col, j_df.index.astype(str)))
    student_ids = list(s_df.get(student_id_col, s_df.index.astype(str)))

    for i, sid in enumerate(student_ids):
        row_scores = sim[i]
        top_idx = np.argsort(row_scores)[-top_k:][::-1]
        matches = []
        for idx in top_idx:
            score = float(row_scores[idx])
            jid = job_ids[idx]
            job_skills = job_skill_sets[idx]
            student_skills = student_skill_sets[i]
            missing_skills = sorted(list(job_skills - student_skills))
            missing_count = len(missing_skills)
            required_count = len(job_skills)
            missing_pct = round((missing_count / required_count) * 100, 2) if required_count > 0 else 0.0

            matches.append({
                "job_id": jid,
                "score": round(score, 4),
                "missing_skills": missing_skills,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
            })
        results.append({"student_id": sid, "top_matches": matches})

    return pd.DataFrame(results)


def flatten_matches_df(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested matches (student_id + list of matches) into table form.
    """
    rows = []
    for _, r in matches_df.iterrows():
        sid = r["student_id"]
        for rank, m in enumerate(r["top_matches"], start=1):
            rows.append({
                "student_id": sid,
                "rank": rank,
                "job_id": m["job_id"],
                "score": m["score"],
                "missing_skills": ";".join(m["missing_skills"]) if m["missing_skills"] else "",
                "missing_count": m["missing_count"],
                "missing_pct": m["missing_pct"],
            })
    return pd.DataFrame(rows)


# -------------------------
# Quick demo
# -------------------------
if __name__ == "__main__":
    try:
        from ingest import load_data
        from data_cleaning import clean_students, clean_internships

        students_raw, internships_raw = load_data()
        students = clean_students(students_raw)
        internships = clean_internships(internships_raw)

        matches = match_students_to_internships_with_gaps(students, internships, top_k=3)
        print(matches.head())
        flat = flatten_matches_df(matches)
        print(flat.head())
    except Exception as e:
        logging.error("Demo failed: %s", e)
