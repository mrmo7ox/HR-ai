from flask import Flask, Blueprint, request, jsonify
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import os
import sys
from job_fit import JobFitPredictor

app = Flask(__name__)
api = Blueprint('api', __name__)

models_dir = "../models/Agents"

try:
    candidate_model = joblib.load(os.path.join(models_dir, "candidate_model.pkl"))
    salary_model = joblib.load(os.path.join(models_dir, "salary_model.pkl"))
    label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
except Exception as e:
    print(f"{e}")
    sys.exit(1)

predictor = JobFitPredictor

@api.route('/job_fit/predict', methods=['POST'])
def job_fit():
    data = request.get_json()
    print("Received data:", data)
    
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded. Please run trainer.py first.'}), 500
        
        required_skills = data.get('required_skills', '')
        candidate_skills = data.get('candidate_skills', '')
        degree = data.get('degree', 'bachelors')
        years_experience = float(data.get('years_experience', 0))
        
        if not required_skills or not candidate_skills:
            return jsonify({'error': 'required_skills and candidate_skills are required'}), 400
        
        result = predictor.predict_fit(required_skills, candidate_skills, degree, years_experience)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
            
        return jsonify({
            'fit': result['fit'],
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/salary/predict', methods=['POST'])
def salary():
    data = request.get_json()
    print("📊 Salary prediction request:", data)

    try:
        years_experience = data.get('years_experience')
        role = data.get('role', '').title()
        degree = data.get('degree', '').title()
        company_size = data.get('company_size', '').title()
        location = data.get('location', '').title()
        level = data.get('level', '').title()

        if years_experience is None:
            return jsonify({'error': 'years_experience is required'}), 400

        try:
            role_encoded = label_encoders['role'].transform([role])[0]
        except:
            role_encoded = 0
        
        try:
            location_encoded = label_encoders['location'].transform([location])[0]
        except:
            location_encoded = 0

        degree_encoded = label_encoders['degree'].get(degree, 0)
        company_size_encoded = label_encoders['company_size'].get(company_size, 0)
        level_encoded = label_encoders['level'].get(level, 0)

        features = pd.DataFrame([{
            'years_experience': years_experience,
            'role_encoded': role_encoded,
            'degree_encoded': degree_encoded,
            'company_size_encoded': company_size_encoded,
            'location_encoded': location_encoded,
            'level_encoded': level_encoded
        }])

        prediction = salary_model.predict(features)[0]
        
        return jsonify({
            'predicted_salary_mad': round(prediction, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Salary prediction failed: {str(e)}'}), 500
@api.route('/resume_screen/predict', methods=['POST'])
def resume_screen():
   return 'resume_screen'

@api.route('/candidate_priority/predict', methods=['POST'])
def candidate_priority():
   data = request.get_json()

   years_exp_band = data.get("years_exp_band", "").strip()
   skills_coverage_band = data.get("skills_coverage_band", "").strip().capitalize()
   referral_flag = int(data.get("referral_flag", 0))
   english_level = data.get("english_level", "").strip().upper()
   location_match = data.get("location_match", "").strip().title()

   years_map = {"0-1": 0, "1-3": 1, "3-6": 2, "6+": 3}
   skills_map = {"Low": 0, "Medium": 1, "High": 2}
   english_map = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
   location_map = {"Remote": 0, "Remoteok": 1, "Local": 2}

   features = pd.DataFrame([{
      "years_exp_band": years_map.get(years_exp_band, 0),
      "skills_coverage_band": skills_map.get(skills_coverage_band, 0),
      "referral_flag": referral_flag,
      "english_level": english_map.get(english_level, 0),
      "location_match": location_map.get(location_match, 0)
   }])

   prediction = candidate_model.predict(features)[0]

   return jsonify({
      "predicted_priority": prediction
   })
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True, port=5000)