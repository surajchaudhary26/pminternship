import pandas as pd
import random
import os

RAW_PATH = os.path.join("ml_engine", "data", "processed")
OUTPUT_PATH = os.path.join("ml_engine", "data", "processed")

def rebalance_skills():
    students = pd.read_csv(os.path.join(RAW_PATH, "students_noisy_random.csv"))

    # Define skill pools with weights
    high_freq = ["Python", "C++", "Java", "MS Office", "Communication"]
    medium_freq = ["SQL", "Excel", "Data Science", "Web Development"]
    low_freq = ["AI/ML", "Cloud Computing", "Robotics", "IoT", "Kubernetes"]

    def assign_skills(existing):
        if pd.isna(existing):
            n = random.randint(2, 5)
            skills = []
            for _ in range(n):
                r = random.random()
                if r < 0.4:
                    skills.append(random.choice(high_freq))
                elif r < 0.7:
                    skills.append(random.choice(medium_freq))
                else:
                    skills.append(random.choice(low_freq))
            return list(set(skills))
        return existing

    students["Skills"] = students["Skills"].apply(assign_skills)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    students.to_csv(os.path.join(OUTPUT_PATH, "students_balanced.csv"), index=False)
    print("✅ Skills distribution rebalanced & saved at students_balanced.csv")

if __name__ == "__main__":
    rebalance_skills()
