import pandas as pd
import random
import os

RAW_PATH = os.path.join("ml_engine", "data", "raw")
OUTPUT_PATH = os.path.join("ml_engine", "data", "processed")

def rebalance_internships():
    # Load raw internships dataset
    internships = pd.read_csv(os.path.join(RAW_PATH, "internships_.csv"))

    # --- 1. Random Missingness ---
    target_cols = ["Sector", "Qualification_Required", "Skills_Required"]
    for col in target_cols:
        frac = random.uniform(0.05, 0.25)  # 5–25% missing randomly
        idx = internships.sample(frac=frac, random_state=random.randint(1,100)).index
        internships.loc[idx, col] = None
        print(f"👉 Introduced {len(idx)} missing in {col}")

    # --- 2. Stipend Distribution ---
    stipends = (
        ["0"] * 60 +  # unpaid
        [str(random.randint(2000, 5000)) for _ in range(25)] +  # low stipend
        [str(random.randint(6000, 15000)) for _ in range(12)] + # medium
        [str(random.randint(20000, 50000)) for _ in range(3)]   # rare high
    )
    internships["Stipend"] = internships["Stipend"].apply(lambda _: random.choice(stipends))

    # --- 3. Duration Distribution ---
    durations = (
        ["2 Months"] * 40 +
        ["3 Months"] * 35 +
        ["1 Month"] * 10 +
        ["6 Months"] * 10 +
        ["12 Months"] * 5
    )
    internships["Duration"] = internships["Duration"].apply(lambda _: random.choice(durations))

    # --- 4. Job Titles Distribution ---
    high_titles = ["Web Development Intern", "Data Science Intern", "AI/ML Intern", "UI/UX Intern"]
    mid_titles = ["Finance Intern", "Marketing Intern", "Cybersecurity Intern"]
    low_titles = ["Blockchain Intern", "Robotics Intern", "Cloud Computing Intern"]

    def assign_title(_):
        r = random.random()
        if r < 0.5:   # 50%
            return random.choice(high_titles)
        elif r < 0.8: # 30%
            return random.choice(mid_titles)
        else:         # 20%
            return random.choice(low_titles)

    internships["Job_Title"] = internships["Job_Title"].apply(assign_title)

    # --- 5. Skills Required Distribution ---
    high_skills = ["Python", "SQL", "Excel", "Communication", "JavaScript"]
    mid_skills = ["Data Science", "Web Development", "Tableau", "Machine Learning"]
    low_skills = ["Blockchain", "Robotics", "Kubernetes", "Cloud Computing", "IoT"]

    def assign_skills(_):
        n = random.randint(3, 6)
        skills = []
        for _ in range(n):
            r = random.random()
            if r < 0.5:
                skills.append(random.choice(high_skills))
            elif r < 0.85:
                skills.append(random.choice(mid_skills))
            else:
                skills.append(random.choice(low_skills))
        return list(set(skills))

    internships["Skills_Required"] = internships["Skills_Required"].apply(assign_skills)

    # --- Save Cleaned & Balanced Dataset ---
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    internships.to_csv(os.path.join(OUTPUT_PATH, "internships_balanced_v2.csv"), index=False)
    print("✅ Balanced v2 internships dataset saved as internships_balanced_v2.csv")

if __name__ == "__main__":
    rebalance_internships()
