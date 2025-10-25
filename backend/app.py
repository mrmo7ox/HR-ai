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
   data = request.get_json()  # JSON input
   df = pd.DataFrame([data])  # wrap single row in DataFrame

   # Predict directly with the model (model handles preprocessing)
   prediction = candidate_model.predict(df)

   return jsonify({"priority": prediction[0]})

app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
   app.run()