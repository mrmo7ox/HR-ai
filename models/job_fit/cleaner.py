import pandas as pd
import numpy as np

def calculate_core_features(row):
    req_skills = [s.strip().lower() for s in str(row['required_skills']).split(',') if s.strip()]
    cand_skills = [s.strip().lower() for s in str(row['candidate_skills']).split(',') if s.strip()]
    
    if not req_skills:
        return 0, 0, 0
    
    overlap = len(set(req_skills) & set(cand_skills))
    match_ratio = overlap / len(req_skills)
    coverage_ratio = overlap / max(len(cand_skills), 1)
    return match_ratio, coverage_ratio, len(req_skills)

def main():
    data = pd.read_csv('job_fit_1000.csv')
    
    data = data.dropna(subset=['required_skills', 'fit'])
    data['candidate_skills'] = data['candidate_skills'].fillna('')
    data['years_experience'] = data['years_experience'].fillna(data['years_experience'].median())
    
    degree_map = {'bachelors': 0, 'masters': 1, 'phd': 2, 'no degree': 0}
    data['degree_encoded'] = data['degree'].str.lower().map(degree_map).fillna(0)

    core_features = data.apply(calculate_core_features, axis=1, result_type='expand')
    data[['skill_match_ratio', 'candidate_coverage', 'required_skill_count']] = core_features
    
    data['skills_text'] = data['required_skills'].fillna('') + ' ' + data['candidate_skills'].fillna('')
    
    data['fit'] = data['fit'].astype(int)
    
    data.to_csv('job_fit_cleaned.csv', index=False)
    
    return data

if __name__ == "__main__":
    main()