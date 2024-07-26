from flask import Flask, render_template, flash, redirect, url_for, request, jsonify
from flask.json import jsonify
import json
import numpy as np
from flask_wtf import CSRFProtect
from dotenv import load_dotenv
import os
import sys
from forms import PredictionForm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.inference.inference import predict_acceptance
import requests

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['HCAPTCHA_SITE_KEY'] = os.getenv('HCAPTCHA_SITE_KEY')
app.config['HCAPTCHA_SECRET_KEY'] = os.getenv('HCAPTCHA_SECRET_KEY')
csrf = CSRFProtect(app)

def custom_json_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

app.json.encoder = custom_json_encoder

def verify_hcaptcha(hcaptcha_response):
    secret_key = app.config['HCAPTCHA_SECRET_KEY']
    verify_url = 'https://hcaptcha.com/siteverify'
    data = {
        'secret': secret_key,
        'response': hcaptcha_response
    }
    response = requests.post(url=verify_url, data=data)
    return response.json()['success']

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    if form.validate_on_submit():
        hcaptcha_response = request.form.get('h-captcha-response')
        if not verify_hcaptcha(hcaptcha_response):
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Please complete the hCaptcha challenge.'}), 400
            flash('Please complete the hCaptcha challenge.', 'error')
            return render_template('index.html', form=form)

        # Construct the post string from form data
        post = f"""
        Demographics: {form.ethnicity.data}, {form.gender.data}, {form.income_bracket.data}, {'First-Gen' if form.first_gen.data == '1' else 'Not First-Gen'}

        Intended Major: {form.major.data}
        Intended School: {form.school.data}

        Academic Info:
        GPA: {form.gpa.data}
        Test Score: {form.test_score.data}
        AP/IB Courses: {form.ap_ib_courses.data}
        AP/IB Scores: {form.ap_ib_scores.data}

        Additional Info:
        Application Round: {form.app_round.data}
        Location: {form.location.data}
        State Status: {form.state_status.data}
        Legacy: {form.legacy.data}
        Languages: {form.languages.data}

        Hooks:
        {form.hooks.data}

        Extracurricular Activities:
        {form.extracurriculars.data}

        Awards and Honors:
        {form.awards.data}
        """
        
        school = form.school.data
        
        try:
            prediction = predict_acceptance(post, school, form.major.data)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return app.response_class(
                    response=json.dumps(prediction, default=custom_json_encoder),
                    status=200,
                    mimetype='application/json'
                )
            return render_template('index.html', form=form, prediction=prediction)
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': str(e)}), 500
            flash(f"An error occurred: {str(e)}", 'error')
    elif request.method == 'POST':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'errors': form.errors}), 400
    
    return render_template('index.html', form=form)

if __name__ == '__main__':
    if os.environ.get("production")=="true":
        app.run(host='0.0.0.0')
    else:
        app.run(debug=True)