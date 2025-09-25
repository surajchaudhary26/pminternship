# pipeline.py
"""
Pipeline Runner
---------------
Runs the full ML prototype pipeline in correct order:
1. Ingest raw datasets
2. Clean and save to cleaned/
3. Augment (optional, default off)
4. Featurize (basic + gap-analysis) and save recommendations
"""

import argparse
import logging
from pathlib import Path

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
    flatten_matches_df,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CLEANED_DIR = Path("ml_engine/data/cleaned")
PROCESSED_DIR = Path("ml_engine/data/processed")


def run_pipeline(run_augment: bool = False, top_k: int = 5, save_matches: bool = True):
    # Step 1: Ingest
    students_raw, internships_raw = load_data()

    # Step 2: Clean
    students_clean = clean_students(students_raw, save_path=CLEANED_DIR / "students_cleaned.csv")
    internships_clean = clean_internships(internships_raw, save_path=CLEANED_DIR / "internships_cleaned.csv")

    # Step 3: Augment (optional)
    if run_augment:
        s1 = rebalance_students(
            input_path=CLEANED_DIR / "students_cleaned.csv",
            output_path=PROCESSED_DIR / "students_rebalanced.csv",
        )
        s2 = random_missing(df=s1, frac=0.1)
        s3 = augment_skills(
            df=s2,
            add_prob=0.25,
            output_path=PROCESSED_DIR / "students_augmented_skills.csv",
        )

        rebalance_internships(
            input_path=CLEANED_DIR / "internships_cleaned.csv",
            output_path=PROCESSED_DIR / "internships_rebalanced.csv",
            missing_frac=0.05,
        )
        logging.info("Augmented datasets generated in processed/")

    # Step 4a: Basic matching (old version)
    matches = match_students_to_internships(students_clean, internships_clean, top_k=top_k)
    logging.info("Generated basic student → internship matches")

    print("\n✅ Sample Basic Recommendations:")
    print(matches.head())

    if save_matches:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "matches.csv"
        matches.to_csv(out_path, index=False)
        logging.info(f"Saved basic recommendations to {out_path}")

    # Step 4b: Gap-analysis matching (new feature)
    matches_with_gaps = match_students_to_internships_with_gaps(
        students_clean,
        internships_clean,
        student_skills_col="skills",
        internship_skills_col="skills_required",
        student_id_col="student_id",
        internship_id_col="job_id",
        top_k=top_k,
    )
    logging.info("Generated student → internship matches with skill gap analysis")

    print("\n🎯 Sample Gap-Analysis Recommendations:")
    print(matches_with_gaps.head())

    if save_matches:
        # Flatten for CSV
        flat = flatten_matches_df(matches_with_gaps)
        gap_path = PROCESSED_DIR / "matches_with_gaps.csv"
        flat.to_csv(gap_path, index=False)
        logging.info(f"Saved flattened gap-analysis recommendations to {gap_path}")

        # Also save nested JSON-like version
        nested_out = PROCESSED_DIR / "matches_with_gaps_nested.csv"
        matches_with_gaps.to_csv(nested_out, index=False)
        logging.info(f"Saved nested gap-analysis recommendations to {nested_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--augment", action="store_true", help="Run augmentation step")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches per student")
    parser.add_argument("--no_save", action="store_true", help="Do not save matches to CSV")
    args = parser.parse_args()

    run_pipeline(run_augment=args.augment, top_k=args.top_k, save_matches=not args.no_save)
