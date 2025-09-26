# pipeline.py
"""
Pipeline Runner (Final with config.yaml support)
------------------------------------------------
Runs the full ML prototype pipeline in correct order:
1. Ingest raw datasets
2. Clean and save to cleaned/
3. Augment (optional, default from config)
4. Featurize (baseline / gap / weighted) and save recommendations
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml_engine.steps.ingest import load_data
from ml_engine.steps.data_cleaning import clean_students, clean_internships
from ml_engine.steps.augment import (
    rebalance_students,
    rebalance_internships,
    random_missing,
    augment_skills,
)
from ml_engine.steps.featurize import (
    match_students_to_internships,
    match_students_to_internships_with_gaps,
    match_students_to_internships_weighted,
    flatten_matches_df,
)

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------------------------------------------
# Config Loader
# ------------------------------------------------------------------
def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("❌ config.yaml not found. Please create one at project root.")
        sys.exit(1)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def preview_df(df: pd.DataFrame, name: str, n: int = 5):
    """Log dataframe shape and sample preview"""
    logging.info(f"{name} → shape={df.shape}")
    logging.debug(f"\n{name} sample:\n{df.head(n)}")


def save_df(df: pd.DataFrame, path: Path, name: str):
    """Save DataFrame safely to CSV"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved {name} to {path}")
    except Exception as e:
        logging.error(f"Failed to save {name} → {e}")
        raise

# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def run_pipeline(
    mode: str,
    run_augment: bool,
    top_k: int,
    save_matches: bool,
    dry_run: bool,
    seed: int,
    cleaned_dir: Path,
    processed_dir: Path,
):
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Step 1: Ingest
    try:
        students_raw, internships_raw = load_data()
    except FileNotFoundError as e:
        logging.error(f"❌ Missing input files: {e}")
        sys.exit(1)

    if dry_run:
        students_raw = students_raw.head(50)
        internships_raw = internships_raw.head(50)
        logging.warning("⚠️ Running in DRY-RUN mode (only first 50 rows)")

    preview_df(students_raw, "Students Raw")
    preview_df(internships_raw, "Internships Raw")

    # Step 2: Clean
    students_clean = clean_students(students_raw, save_path=cleaned_dir / "students_cleaned.csv")
    internships_clean = clean_internships(internships_raw, save_path=cleaned_dir / "internships_cleaned.csv")
    preview_df(students_clean, "Students Cleaned")
    preview_df(internships_clean, "Internships Cleaned")

    # Step 3: Augment (optional)
    if run_augment:
        s1 = rebalance_students(
            input_path=cleaned_dir / "students_cleaned.csv",
            output_path=processed_dir / "students_rebalanced.csv",
        )
        s2 = random_missing(df=s1, frac=0.1, seed=seed)
        s3 = augment_skills(
            df=s2,
            add_prob=0.25,
            output_path=processed_dir / "students_augmented_skills.csv",
            seed=seed,
        )
        preview_df(s3, "Students Augmented")

        rebalance_internships(
            input_path=cleaned_dir / "internships_cleaned.csv",
            output_path=processed_dir / "internships_rebalanced.csv",
            missing_frac=0.05,
            seed=seed,
        )
        logging.info("✨ Augmented datasets generated in processed/")

    # Step 4: Matching
    processed_dir.mkdir(parents=True, exist_ok=True)

    if mode == "baseline":
        matches = match_students_to_internships(students_clean, internships_clean, top_k=top_k)
        logging.info("Generated baseline (skills-only) matches")
        preview_df(matches, "Baseline Matches")

        if save_matches:
            save_df(matches, processed_dir / "matches.csv", "Baseline Matches")

    elif mode == "gap":
        matches_with_gaps = match_students_to_internships_with_gaps(
            students_clean,
            internships_clean,
            student_skills_col="skills",
            internship_skills_col="skills_required",
            student_id_col="student_id",
            internship_id_col="job_id",
            top_k=top_k,
        )
        logging.info("Generated gap-analysis matches")
        preview_df(matches_with_gaps, "Gap Matches")

        if save_matches:
            flat = flatten_matches_df(matches_with_gaps)
            save_df(flat, processed_dir / "matches_with_gaps.csv", "Flat Gap Matches")
            save_df(matches_with_gaps, processed_dir / "matches_with_gaps_nested.csv", "Nested Gap Matches")

    elif mode == "weighted":
        matches_weighted = match_students_to_internships_weighted(students_clean, internships_clean, top_k=top_k)
        logging.info("Generated weighted hybrid matches")
        preview_df(matches_weighted, "Weighted Matches")

        if save_matches:
            save_df(matches_weighted, processed_dir / "matches_weighted.csv", "Weighted Matches")

    else:
        raise ValueError(f"❌ Unknown mode '{mode}'. Use one of: baseline, gap, weighted")

# ------------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()

    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    default_mode = config["modes"]["default"]
    default_top_k = config["params"]["top_k"]
    default_seed = config["params"]["seed"]
    default_augment = config["augment"]["enable"]

    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--mode", type=str, default=default_mode,
                        choices=["baseline", "gap", "weighted"],
                        help="Matching mode")
    parser.add_argument("--augment", action="store_true", default=default_augment,
                        help="Run augmentation step")
    parser.add_argument("--top_k", type=int, default=default_top_k,
                        help="Number of top matches per student")
    parser.add_argument("--no_save", action="store_true", help="Do not save matches to CSV")
    parser.add_argument("--dry_run", action="store_true", help="Run only on first 50 rows")
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed")

    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        run_augment=args.augment,
        top_k=args.top_k,
        save_matches=not args.no_save,
        dry_run=args.dry_run,
        seed=args.seed,
        cleaned_dir=cleaned_dir,
        processed_dir=processed_dir,
    )
