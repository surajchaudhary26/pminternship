import pandas as pd
import random
import numpy as np
import os

RAW_PATH = os.path.join("ml_engine", "data", "raw")
PROCESSED_PATH = os.path.join("ml_engine", "data", "processed")

# Helper: introduce typos in a word
def introduce_typo(text, prob=0.1):
    if not isinstance(text, str) or random.random() > prob:
        return text
    if len(text) < 3:
        return text
    i = random.randint(0, len(text)-2)
    return text[:i] + text[i+1] + text[i] + text[i+2:]

# ----------------------------
# STUDENTS AUGMENTATION
# ----------------------------
def augment_students(df):
    df = df.copy()

    # 1. Randomly drop Skills (10%)
    df.loc[df.sample(frac=0.1).index, "Skills"] = None

    # 2. Add typos in Names (5%)
    df["Name"] = df["Name"].apply(lambda x: introduce_typo(x, prob=0.05))

    # 3. Email variations (lower/upper, missing chars)
    df["Email"] = df["Email"].apply(
        lambda x: x.replace("@", "_at_") if random.random() < 0.05 else x
    )

    # 4. Phone number noise (some 9 or 11 digit)
    df["Phone"] = df["Phone"].apply(
        lambda x: str(x)[:-1] if random.random() < 0.05 else str(x) + "9" if random.random() < 0.05 else str(x)
    )

    # 5. Qualification inconsistencies
    df["Qualification"] = df["Qualification"].apply(
        lambda x: x.replace("B.Tech", "Btech") if isinstance(x, str) and random.random() < 0.1 else x
    )

    # 6. Location spelling mistakes
    df["Location_Preferences"] = df["Location_Preferences"].apply(
        lambda x: str(x).replace("Bangalore", "Banglore") if isinstance(x, str) and random.random() < 0.05 else x
    )

    return df

# ----------------------------
# INTERNSHIPS AUGMENTATION
# ----------------------------
def augment_internships(df):
    df = df.copy()

    # 1. Randomly set stipend to "Not Disclosed" (10%)
    df.loc[df.sample(frac=0.1).index, "Stipend"] = "Not Disclosed"

    # 2. Outliers in stipend (1% extreme values)
    outlier_idx = df.sample(frac=0.01).index
    df.loc[outlier_idx, "Stipend"] = random.choice([0, 100000, 50000])

    # 3. Duration variations
    df["Duration"] = df["Duration"].apply(
        lambda x: str(x).replace("months", "mo.") if isinstance(x, str) and random.random() < 0.1 else x
    )
    df["Duration"] = df["Duration"].apply(
        lambda x: "90 days" if isinstance(x, str) and random.random() < 0.05 else x
    )

    # 4. Job Title typos
    df["Job_Title"] = df["Job_Title"].apply(lambda x: introduce_typo(x, prob=0.05))

    # 5. Skills Required typos
    df["Skills_Required"] = df["Skills_Required"].apply(
        lambda x: str(x).replace("React", "Recat") if isinstance(x, str) and random.random() < 0.05 else x
    )

    # 6. Sector filling if missing
    df["Sector"] = df.apply(
        lambda row: row["Job_Title"].split()[0] if pd.isna(row["Sector"]) else row["Sector"], axis=1
    )

    # 7. Description template replacement
    templates = [
        "Work on {skill} projects at {company}.",
        "Assist in developing {skill} solutions.",
        "Support team in {skill} related tasks.",
        "Contribute to ongoing {sector} initiatives."
    ]
    def gen_description(row):
        if random.random() < 0.3:  # 30% rows modified
            skill = "Python" if pd.isna(row["Skills_Required"]) else str(row["Skills_Required"]).split(",")[0]
            return random.choice(templates).format(
                skill=skill.strip("[]'"),
                company=row["Company_Name"],
                sector=row["Sector"]
            )
        return row["Description"]
    df["Description"] = df.apply(gen_description, axis=1)

    return df

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    students = pd.read_csv(os.path.join(RAW_PATH, "students_.csv"))
    internships = pd.read_csv(os.path.join(RAW_PATH, "internships_.csv"))

    print("📥 Raw Students:", students.shape, "Internships:", internships.shape)

    noisy_students = augment_students(students)
    noisy_internships = augment_internships(internships)

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    noisy_students.to_csv(os.path.join(PROCESSED_PATH, "students_noisy.csv"), index=False)
    noisy_internships.to_csv(os.path.join(PROCESSED_PATH, "internships_noisy.csv"), index=False)

    print("✅ Augmentation complete! Noisy datasets saved in data/processed/")
