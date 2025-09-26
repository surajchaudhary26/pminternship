# ml_engine/steps/augment.py
"""
Data Augmentation Module (with seed + pd.NA fix)
------------------------------------------------
Contains augmentation utilities with reproducibility:
- rebalance_students
- rebalance_internships
- random_missing
- augment_skills
"""

from pathlib import Path
import pandas as pd
import random
import logging
import argparse
import numpy as np
from typing import Optional, List, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_RAW = Path("ml_engine/data/raw")
DEFAULT_PROCESSED = Path("ml_engine/data/processed")


# -----------------------------
# Helpers
# -----------------------------
def _ensure_list_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = [[] for _ in range(len(df))]
        return df

    def _to_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    return list(parsed) if isinstance(parsed, (list, tuple)) else [s]
                except Exception:
                    return [p.strip() for p in s.split(",") if p.strip()]
            if "," in s:
                return [p.strip() for p in s.split(",") if p.strip()]
            return [s]
        return []
    df[col] = df[col].apply(_to_list)
    return df


# -----------------------------
# 1) Rebalance students
# -----------------------------
def rebalance_students(df: Optional[pd.DataFrame] = None,
                       input_path: Optional[Union[str, Path]] = None,
                       output_path: Optional[Union[str, Path]] = None,
                       boost_low_freq_prob: float = 0.1,
                       low_skill_pool: Optional[List[str]] = None,
                       seed: Optional[int] = None) -> pd.DataFrame:
    if df is None:
        if input_path is None:
            raise ValueError("Either df or input_path must be provided")
        df = pd.read_csv(input_path)

    if seed is not None:
        random.seed(seed)

    df = df.copy()
    df = _ensure_list_col(df, "skills")

    if low_skill_pool is None:
        low_skill_pool = ["IoT", "Blockchain", "Game Development", "Embedded_C", "ROS"]

    def _maybe_add_low(skills):
        if random.random() < boost_low_freq_prob:
            to_add = random.choice(low_skill_pool)
            if to_add not in skills:
                skills = skills + [to_add]
        return skills

    df["skills"] = df["skills"].apply(_maybe_add_low)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved rebalanced students to {output_path}")

    return df


# -----------------------------
# 2) Rebalance internships
# -----------------------------
def rebalance_internships(df: Optional[pd.DataFrame] = None,
                          input_path: Optional[Union[str, Path]] = None,
                          output_path: Optional[Union[str, Path]] = None,
                          missing_frac: float = 0.05,
                          target_cols: Optional[List[str]] = None,
                          seed: Optional[int] = None) -> pd.DataFrame:
    if df is None:
        if input_path is None:
            raise ValueError("Either df or input_path must be provided")
        df = pd.read_csv(input_path)

    if seed is not None:
        np.random.seed(seed)

    df = df.copy()
    if target_cols is None:
        target_cols = [c for c in ["sector", "qualification_required", "skills_required", "description"] if c in df.columns]

    for col in target_cols:
        idx = df.sample(frac=missing_frac, random_state=seed).index
        df.loc[idx, col] = pd.NA   # ✅ FIXED (no more warning)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved rebalanced internships to {output_path}")

    return df


# -----------------------------
# 3) Random Missingness
# -----------------------------
def random_missing(df: Optional[pd.DataFrame] = None,
                   input_path: Optional[Union[str, Path]] = None,
                   output_path: Optional[Union[str, Path]] = None,
                   frac: float = 0.1,
                   target_cols: Optional[List[str]] = None,
                   seed: Optional[int] = None) -> pd.DataFrame:
    if df is None:
        if input_path is None:
            raise ValueError("Provide df or input_path")
        df = pd.read_csv(input_path)

    if seed is not None:
        np.random.seed(seed)

    df = df.copy()
    if target_cols is None:
        target_cols = list(df.columns)

    for col in target_cols:
        idx = df.sample(frac=frac, random_state=seed).index
        df[col] = df[col].astype("object")
        df.loc[idx, col] = pd.NA

        # df.loc[idx, col] = pd.NA   # ✅ FIXED

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved random-missing dataset to {output_path}")

    return df


# -----------------------------
# 4) Augment skills
# -----------------------------
def augment_skills(df: Optional[pd.DataFrame] = None,
                   input_path: Optional[Union[str, Path]] = None,
                   output_path: Optional[Union[str, Path]] = None,
                   add_prob: float = 0.25,
                   extra_pool: Optional[List[str]] = None,
                   seed: Optional[int] = None) -> pd.DataFrame:
    if df is None:
        if input_path is None:
            raise ValueError("Provide df or input_path")
        df = pd.read_csv(input_path)

    if seed is not None:
        random.seed(seed)

    df = df.copy()
    df = _ensure_list_col(df, "skills")

    if extra_pool is None:
        extra_pool = ["Python", "C++", "Java", "MS Office", "Communication",
                      "SQL", "Kubernetes", "React", "Figma"]

    def _maybe_add(skills):
        if random.random() < add_prob:
            choice = random.choice(extra_pool)
            if choice not in skills:
                skills = skills + [choice]
        return skills

    df["skills"] = df["skills"].apply(_maybe_add)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved augmented skills dataset to {output_path}")

    return df
