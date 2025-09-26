# ml_engine/steps/ingest.py
"""
Data Ingestion Module
---------------------
Loads raw student & internship CSVs into pandas DataFrames.

Usage:
    from ingest import load_data
    students_df, internships_df = load_data(raw_dir="ml_engine/data/raw")
"""

from pathlib import Path
import pandas as pd
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded {path.name} with shape {df.shape}")
    return df


def load_students(file_name: str = "students_.csv", raw_dir: str = "ml_engine/data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / file_name
    return _read_csv(path)


def load_internships(file_name: str = "internships_.csv", raw_dir: str = "ml_engine/data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / file_name
    return _read_csv(path)


def load_data(raw_dir: str = "ml_engine/data/raw",
              students_file: str = "students_.csv",
              internships_file: str = "internships_.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    students = load_students(students_file, raw_dir)
    internships = load_internships(internships_file, raw_dir)
    logging.info(f"Loaded students: {students.shape}, internships: {internships.shape}")
    return students, internships


if __name__ == "__main__":
    # quick test run
    try:
        load_data()
    except Exception as e:
        logging.error(e)
