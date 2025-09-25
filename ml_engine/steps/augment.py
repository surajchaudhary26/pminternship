# ml_engine/steps/augment.py
"""
Data Augmentation Module (merged)
---------------------------------
Contains several augmentation utilities:
- rebalance_students: boost under-represented skills or resample
- rebalance_internships: introduce controlled missingness or perturbation
- random_missing: create missingness for robustness testing
- augment_skills: inject synthetic skills (from pools) into skills lists

Functions accept DataFrame OR file paths and return DataFrame (and optionally save to disk).
CLI available for quick runs.
"""

from pathlib import Path
import pandas as pd
import random
import logging
import argparse
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
    # if it's stored as string, try to parse common formats: assume cleaned step already handled, but safe-check:
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
                    # fallback CSV-split
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
                       low_skill_pool: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Boost low-frequency skills by injecting one low-frequency skill into a
    random subset of students. Return updated DataFrame (and save if output_path provided).
    """
    if df is None:
        if input_path is None:
            raise ValueError("Either df or input_path must be provided")
        df = pd.read_csv(input_path)

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
# 2) Rebalance internships (missingness / perturbation)
# -----------------------------
def rebalance_internships(df: Optional[pd.DataFrame] = None,
                          input_path: Optional[Union[str, Path]] = None,
                          output_path: Optional[Union[str, Path]] = None,
                          missing_frac: float = 0.05,
                          target_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Introduce missingness/perturbations in selected columns for internships.
    """
    if df is None:
        if input_path is None:
            raise ValueError("Either df or input_path must be provided")
        df = pd.read_csv(input_path)

    df = df.copy()
    if target_cols is None:
        target_cols = [c for c in ["sector", "qualification_required", "skills_required", "description"] if c in df.columns]

    for col in target_cols:
        idx = df.sample(frac=missing_frac, random_state=None).index
        df.loc[idx, col] = None

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved rebalanced internships to {output_path}")

    return df


# -----------------------------
# 3) Random Missingness (generic)
# -----------------------------
def random_missing(df: Optional[pd.DataFrame] = None,
                   input_path: Optional[Union[str, Path]] = None,
                   output_path: Optional[Union[str, Path]] = None,
                   frac: float = 0.1,
                   target_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Introduce random missing values across given columns (students or internships).
    """
    if df is None:
        if input_path is None:
            raise ValueError("Provide df or input_path")
        df = pd.read_csv(input_path)

    df = df.copy()
    if target_cols is None:
        target_cols = list(df.columns)

    for col in target_cols:
        idx = df.sample(frac=frac).index
        df.loc[idx, col] = None

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved random-missing dataset to {output_path}")

    return df


# -----------------------------
# 4) Augment skills (inject synthetic skills)
# -----------------------------
def augment_skills(df: Optional[pd.DataFrame] = None,
                   input_path: Optional[Union[str, Path]] = None,
                   output_path: Optional[Union[str, Path]] = None,
                   add_prob: float = 0.25,
                   extra_pool: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Inject additional skills into students' skills lists with probability add_prob.
    """
    if df is None:
        if input_path is None:
            raise ValueError("Provide df or input_path")
        df = pd.read_csv(input_path)

    df = df.copy()
    df = _ensure_list_col(df, "skills")

    if extra_pool is None:
        extra_pool = ["Python", "C++", "Java", "MS Office", "Communication", "SQL", "Kubernetes", "React", "Figma"]

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


# -----------------------------
# CLI runner
# -----------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Augmentation utilities")
    p.add_argument("--action", choices=["rebalance_students", "rebalance_internships", "random_missing", "augment_skills", "all"], default="all")
    p.add_argument("--input", type=str, default=None, help="Input CSV path (optional)")
    p.add_argument("--output", type=str, default=None, help="Output CSV path (optional)")
    p.add_argument("--frac", type=float, default=0.1, help="Fraction for missingness")
    p.add_argument("--add_prob", type=float, default=0.25, help="Probability to add a skill")
    return p.parse_args()


def main():
    args = _parse_args()
    act = args.action

    if act == "rebalance_students":
        rebalance_students(input_path=args.input or DEFAULT_PROCESSED / "students_balanced.csv",
                           output_path=args.output or DEFAULT_PROCESSED / "students_rebalanced.csv")
    elif act == "rebalance_internships":
        rebalance_internships(input_path=args.input or DEFAULT_RAW / "internships_.csv",
                              output_path=args.output or DEFAULT_PROCESSED / "internships_rebalanced.csv",
                              missing_frac=args.frac)
    elif act == "random_missing":
        random_missing(input_path=args.input or DEFAULT_PROCESSED / "students_cleaned.csv",
                       output_path=args.output or DEFAULT_PROCESSED / "students_random_missing.csv",
                       frac=args.frac)
    elif act == "augment_skills":
        augment_skills(input_path=args.input or DEFAULT_PROCESSED / "students_random_missing.csv",
                       output_path=args.output or DEFAULT_PROCESSED / "students_augmented_skills.csv",
                       add_prob=args.add_prob)
    elif act == "all":
        # default pipeline order (safe defaults)
        s1 = rebalance_students(input_path=DEFAULT_PROCESSED / "students_balanced.csv",
                                output_path=DEFAULT_PROCESSED / "students_rebalanced.csv")
        s2 = random_missing(df=s1, frac=0.1)
        s3 = augment_skills(df=s2, add_prob=0.25, output_path=DEFAULT_PROCESSED / "students_augmented_skills.csv")

        rebalance_internships(input_path=DEFAULT_RAW / "internships_.csv",
                              output_path=DEFAULT_PROCESSED / "internships_rebalanced.csv",
                              missing_frac=0.05)

        logging.info("Completed all augmentations (students + internships)")

if __name__ == "__main__":
    # Use cleaned datasets as input, not raw/processed duplicates
    s1 = rebalance_students(
        input_path=Path("ml_engine/data/cleaned/students_cleaned.csv"),
        output_path=DEFAULT_PROCESSED / "students_rebalanced.csv"
    )
    s2 = random_missing(df=s1, frac=0.1)
    s3 = augment_skills(
        df=s2,
        add_prob=0.25,
        output_path=DEFAULT_PROCESSED / "students_augmented_skills.csv"
    )

    rebalance_internships(
        input_path=Path("ml_engine/data/cleaned/internships_cleaned.csv"),
        output_path=DEFAULT_PROCESSED / "internships_rebalanced.csv",
        missing_frac=0.05
    )

    logging.info("Completed all augmentations (students + internships)")
