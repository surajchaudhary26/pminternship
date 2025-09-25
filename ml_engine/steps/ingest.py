# ml_engine/steps/ingest.py

import pandas as pd
import os

RAW_DATA_PATH = os.path.join("ml_engine", "data", "raw")

def load_data():
    """
    Load students and internships datasets from raw folder.
    Returns:
        students_df (pd.DataFrame)
        internships_df (pd.DataFrame)
    """
    students_file = os.path.join(RAW_DATA_PATH, "students_augmented.csv")
    internships_file = os.path.join(RAW_DATA_PATH, "internships_augmented.csv")

    if not os.path.exists(students_file):
        raise FileNotFoundError(f"❌ {students_file} not found")
    if not os.path.exists(internships_file):
        raise FileNotFoundError(f"❌ {internships_file} not found")

    print("📥 Loading raw datasets...")
    students_df = pd.read_csv(students_file)
    internships_df = pd.read_csv(internships_file)

    print(f"✅ Students dataset loaded: {students_df.shape}")
    print(f"✅ Internships dataset loaded: {internships_df.shape}")

    return students_df, internships_df


if __name__ == "__main__":
    students, internships = load_data()
    print("🔍 Students Preview:\n", students.head())
    print("🔍 Internships Preview:\n", internships.head())
