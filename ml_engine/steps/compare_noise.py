import pandas as pd

# Load raw and noisy data
raw_students = pd.read_csv("ml_engine/data/raw/students_.csv")
raw_internships = pd.read_csv("ml_engine/data/raw/internships_.csv")

noisy_students = pd.read_csv("ml_engine/data/processed/students_noisy.csv")
noisy_internships = pd.read_csv("ml_engine/data/processed/internships_noisy.csv")

# Compare first 10 rows side by side
student_diff = pd.DataFrame({
    "Raw_Name": raw_students["Name"].head(10),
    "Noisy_Name": noisy_students["Name"].head(10),
    "Raw_Email": raw_students["Email"].head(10),
    "Noisy_Email": noisy_students["Email"].head(10),
    "Raw_Phone": raw_students["Phone"].head(10),
    "Noisy_Phone": noisy_students["Phone"].head(10),
    "Raw_Skills": raw_students["Skills"].head(10),
    "Noisy_Skills": noisy_students["Skills"].head(10),
})

internship_diff = pd.DataFrame({
    "Raw_Title": raw_internships["Job_Title"].head(10),
    "Noisy_Title": noisy_internships["Job_Title"].head(10),
    "Raw_Skills": raw_internships["Skills_Required"].head(10),
    "Noisy_Skills": noisy_internships["Skills_Required"].head(10),
    "Raw_Stipend": raw_internships["Stipend"].head(10),
    "Noisy_Stipend": noisy_internships["Stipend"].head(10),
    "Raw_Description": raw_internships["Description"].head(10),
    "Noisy_Description": noisy_internships["Description"].head(10),
})

print("\n=== Student Noise Comparison ===")
# print(student_diff)
print("\n=== Internship Noise Comparison ===")
# print(internship_diff)
internship_diff.shape
