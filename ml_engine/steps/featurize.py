import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np

# Helper function: convert list-like strings to space-separated text
def parse_skills(skills_str):
    try:
        if isinstance(skills_str, str):
            skills = ast.literal_eval(skills_str)
            return " ".join(skills)
        elif isinstance(skills_str, list):
            return " ".join(skills_str)
    except:
        return ""
    return ""

def recommend_internships(student_id, students_path, internships_path, top_n=5,
                          w_text=0.6, w_exp=0.2, w_stipend=0.2):
    # Load cleaned datasets (with scaled features!)
    students = pd.read_csv(students_path)
    internships = pd.read_csv(internships_path)

    # Pick student profile
    student = students[students["Student_ID"] == student_id].iloc[0]

    # Prepare skills text
    students["skills_text"] = students["Skills"].apply(parse_skills)
    internships["skills_text"] = internships["Skills_Required"].apply(parse_skills)

    # Internship combined text (title + qualification + skills)
    internships["combined_text"] = (
        internships["Job_Title"].fillna("") + " " +
        internships["Qualification_Required"].fillna("") + " " +
        internships["skills_text"]
    )

    # Student combined text
    student_text = (
        str(student["Qualification"]) + " " +
        str(student["skills_text"]) + " " +
        str(student["Sector_Interests"])
    )

    # --- Text Similarity ---
    tfidf = TfidfVectorizer()
    internship_matrix = tfidf.fit_transform(internships["combined_text"])
    student_vec = tfidf.transform([student_text])
    text_sim = cosine_similarity(student_vec, internship_matrix).flatten()

    # --- Numeric Similarities ---
    # Experience match (closer scaled values = higher score)
    if "Experience_Scaled" in student and "Experience_Required_Scaled" in internships:
        exp_diff = abs(internships["Experience_Required_Scaled"] - student["Experience_Scaled"])
        exp_sim = 1 - exp_diff  # convert distance → similarity
    else:
        exp_sim = np.zeros(len(internships))

    # Stipend match (prefer internships with stipend >= student's threshold if defined)
    if "Stipend_Scaled" in internships.columns:
        stipend_sim = internships["Stipend_Scaled"]
    else:
        stipend_sim = np.zeros(len(internships))

    # --- Final Score ---
    final_score = (w_text * text_sim) + (w_exp * exp_sim) + (w_stipend * stipend_sim)

    # --- Top N Recommendations ---
    top_indices = final_score.argsort()[::-1][:top_n]
    results = internships.iloc[top_indices][
        ["Job_ID", "Job_Title", "Company_Name", "Stipend", "Duration", "Skills_Required"]
    ].copy()
    results["Text_Similarity"] = text_sim[top_indices]
    results["Exp_Similarity"] = exp_sim.iloc[top_indices]
    results["Stipend_Similarity"] = stipend_sim.iloc[top_indices]
    results["Final_Score"] = final_score[top_indices]

    return results
