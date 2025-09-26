# verify_seed.py
"""
Verify reproducibility of augmentation with seed
------------------------------------------------
Runs pipeline twice with same seed and once with a different seed.
Compares outputs to check reproducibility.
"""

import pandas as pd
import subprocess
import os

AUG_FILE = "ml_engine/data/processed/students_augmented_skills.csv"


def run_pipeline(seed: int):
    """Run pipeline with given seed"""
    print(f"\n▶️ Running pipeline with seed={seed}")
    subprocess.run(
        ["python", "pipeline.py", "--mode", "weighted", "--augment", "--seed", str(seed)],
        check=True
    )
    df = pd.read_csv(AUG_FILE)
    return df


def compare_runs():
    # First run with seed=42
    df1 = run_pipeline(42)

    # Second run with same seed=42
    df2 = run_pipeline(42)

    # Third run with different seed=123
    df3 = run_pipeline(123)

    print("\n🔎 Comparing results...")
    print("Same seed reproducible? ", df1.equals(df2))  # ✅ Expect True
    print("Different seed reproducible? ", df1.equals(df3))  # ❌ Expect False


if __name__ == "__main__":
    if not os.path.exists("pipeline.py"):
        print("❌ Run this script from project root where pipeline.py is located.")
    else:
        compare_runs()
