# pipeline.py
"""
Pipeline Runner
---------------
Runs the full ML prototype pipeline in correct order:
1. Ingest raw datasets
2. Clean and save to cleaned/
3. Augment (optional, default off)
4. Featurize and print sample recommendations
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
from ml_engine.steps.featurize import match_students_to_internships

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CLEANED_DIR = Path("ml_engine/data/cleaned")
PROCESSED_DIR = Path("ml_engine/data/processed")


def run_pipeline(run_augment: bool = False, top_k: int = 5):
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

    # Step 4: Featurize
    matches = match_students_to_internships(students_clean, internships_clean, top_k=top_k)
    logging.info("Generated student → internship matches")
    print(matches.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--augment", action="store_true", help="Run augmentation step")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches per student")
    args = parser.parse_args()

    run_pipeline(run_augment=args.augment, top_k=args.top_k)
