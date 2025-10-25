from flask import Flask, Blueprint

app = Flask(__name__)


api = Blueprint('api', __name__)

@api.route('/salary/predict', methods=['POST'])
def salary():
   return 'salary'

@api.route('/job_fit/predict', methods=['POST'])
def job_fit():
   return 'job_fit'

@api.route('/resume_screen/predict', methods=['POST'])
def resume_screen():
   return 'resume_screen'

@api.route('/candidate_priority/predict', methods=['POST'])
def candidate_priority():
   return 'candidate_priority'

app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
   app.run()