import pandas as pd
import random
import os

RAW_PATH = os.path.join("ml_engine", "data", "processed")
OUTPUT_PATH = os.path.join("ml_engine", "data", "processed")

def rebalance_v2():
    students = pd.read_csv(os.path.join(RAW_PATH, "students_balanced.csv"))

    # --- Skills Distribution ---
    high_freq = ["Python", "C++", "Java", "MS Office", "Communication", "Excel"]
    medium_freq = ["SQL", "Data Science", "Web Development"]
    low_freq = ["AI/ML", "Cloud Computing", "Robotics", "IoT", "Kubernetes"]

    def assign_skills(_):
        n = random.randint(3, 5)
        skills = []
        for _ in range(n):
            r = random.random()
            if r < 0.5:   # 50% chance high freq
                skills.append(random.choice(high_freq))
            elif r < 0.8: # 30% chance medium freq
                skills.append(random.choice(medium_freq))
            else:         # 20% chance low freq
                skills.append(random.choice(low_freq))
        return list(set(skills))

    students["Skills"] = students["Skills"].apply(lambda x: assign_skills(x) if pd.isna(x) else assign_skills(x))

    # --- Preferred Job Type Distribution ---
    job_types = (
        ["Internship"] * 50 +   # 50%
        ["Remote"] * 25 +       # 25%
        ["Part-time"] * 15 +    # 15%
        ["Full-time"] * 10      # 10%
    )

    students["Preferred_Job_Type"] = students["Preferred_Job_Type"].apply(lambda _: random.choice(job_types))

    # Save new balanced dataset
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    students.to_csv(os.path.join(OUTPUT_PATH, "students_balanced_v2.csv"), index=False)
    print("✅ Balanced v2 dataset saved as students_balanced_v2.csv")

if __name__ == "__main__":
    rebalance_v2()
