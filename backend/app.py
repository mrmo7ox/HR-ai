from flask import Flask, Blueprint, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

api = Blueprint('api', __name__)

with open("../models/Agents/salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/Agents/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
    
with open("../models/Agents/candidate_model.pkl", "rb") as f:
    candidate_model = pickle.load(f)

@api.route('/salary/predict', methods=['POST'])
def salary():
   data = request.get_json()

   years_experience = data.get('years_experience')
   role = data.get('role', '').title()
   degree = data.get('degree', '').title()
   company_size = data.get('company_size', '').title()
   location = data.get('location', '').title()
   level = data.get('level', '').title()

   role_encoded = label_encoders['role'].transform([role])[0]
   location_encoded = label_encoders['location'].transform([location])[0]
   degree_encoded = label_encoders['degree'][degree]
   company_size_encoded = label_encoders['company_size'][company_size]
   level_encoded = label_encoders['level'][level]

   features = pd.DataFrame([{
      'years_experience': years_experience,
      'role_encoded': role_encoded,
      'degree_encoded': degree_encoded,
      'company_size_encoded': company_size_encoded,
      'location_encoded': location_encoded,
      'level_encoded': level_encoded
   }])

   prediction = model.predict(features)[0]
   return jsonify({'predicted_salary_mad': round(prediction, 2)})

@api.route('/job_fit/predict', methods=['POST'])
def job_fit():
   return 'job_fit'

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
   app.run()