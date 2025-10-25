import joblib
import numpy as np
from scipy.sparse import hstack
import os
import pandas as pd  # Add this import

class JobFitPredictor:
    def __init__(self, models_dir="../models/Agents"):
        self.models_dir = models_dir
        self.degree_map = {'bachelors': 0, 'masters': 1, 'phd': 2, 'no degree': 0}
        self.model = self._load_model("job_fit_model.pkl")
        self.tfidf = self._load_model("tfidf_vectorizer.pkl")
        self.scaler = self._load_model("feature_scaler.pkl")
        self.feature_names = self._load_model("feature_names.pkl")
        print("🔧 JobFitPredictor initialized")
        print(f"   Model: {'✅' if self.model else '❌'}")
        print(f"   TF-IDF: {'✅' if self.tfidf else '❌'}")
        print(f"   Scaler: {'✅' if self.scaler else '❌'}")
        print(f"   Features: {'✅' if self.feature_names else '❌'}")
    
    def _load_model(self, filename):
        try:
            filepath = os.path.join(self.models_dir, filename)
            return joblib.load(filepath)
        except Exception as e:
            print(f"⚠️  Could not load {filename}: {e}")
            return None
    
    def predict_fit(self, required_skills, candidate_skills, degree, years_experience):
        try:
            if None in [self.model, self.tfidf]:
                return {'error': 'Essential model files missing. Please run trainer.py first.'}
            
            req_skills = [s.strip().lower() for s in required_skills.split(',') if s.strip()]
            cand_skills = [s.strip().lower() for s in candidate_skills.split(',') if s.strip()]
            
            if not req_skills:
                match_ratio = 0
                coverage = 0
                req_count = 0
                overlap = 0
            else:
                overlap = len(set(req_skills) & set(cand_skills))
                match_ratio = overlap / len(req_skills)
                coverage = overlap / max(len(cand_skills), 1)
                req_count = len(req_skills)
            
            degree_encoded = self.degree_map.get(degree.lower(), 0)
            
            feature_dict = {
                'degree_encoded': degree_encoded,
                'years_experience': years_experience,
                'skill_match_ratio': match_ratio,
                'candidate_coverage': coverage,
                'required_skill_count': req_count
            }
            
            if self.feature_names is None:
                self.feature_names = list(feature_dict.keys())
            
            numeric_values = [feature_dict[feature] for feature in self.feature_names]
            
            if self.scaler:
                numeric_df = pd.DataFrame([numeric_values], columns=self.feature_names)
                numeric_scaled = self.scaler.transform(numeric_df)
            else:
                numeric_scaled = np.array([numeric_values])
            
            skills_text = required_skills + ' ' + candidate_skills
            skills_tfidf = self.tfidf.transform([skills_text])

            features = hstack([skills_tfidf, numeric_scaled])
            probability = self.model.predict_proba(features)[0, 1]
            fit_decision = probability >= 0.5
            
            return {
                'fit': bool(fit_decision),
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}