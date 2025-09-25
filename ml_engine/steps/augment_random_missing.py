import pandas as pd
import random
import os

RAW_PATH = os.path.join("ml_engine", "data", "processed")
OUTPUT_PATH = os.path.join("ml_engine", "data", "processed")

def randomize_missing():
    students = pd.read_csv(os.path.join(RAW_PATH, "students_noisy.csv"))

    # Columns where missingness makes sense
    target_cols = ["Skills", "Sector_Interests", "Location_Preferences", "Qualification"]

    for col in target_cols:
        # Random fraction between 5% to 30% missing
        frac = random.uniform(0.05, 0.3)
        idx = students.sample(frac=frac, random_state=random.randint(1,100)).index
        students.loc[idx, col] = None
        print(f"👉 Introduced {len(idx)} missing values in {col} ({round(frac*100,1)}%)")

    # Extra realism: for ~5% of students, drop multiple fields together
    multi_idx = students.sample(frac=0.05, random_state=random.randint(1,100)).index
    for col in target_cols:
        students.loc[multi_idx, col] = None
    print(f"👉 Introduced multi-column missingness for {len(multi_idx)} students")

    # Save randomized noisy dataset
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    students.to_csv(os.path.join(OUTPUT_PATH, "students_noisy_random.csv"), index=False)
    print("✅ Randomized missing values saved to students_noisy_random.csv")

if __name__ == "__main__":
    randomize_missing()
