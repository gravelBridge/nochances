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
from train.inference.inference import predict_acceptance, process_post_with_gpt
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
        post = f"{form.info.data}\nWant to go to {form.school.data}, applying to the {form.app_round.data} round, majoring in {form.major.data}\n\n"
        
        school = form.school.data
        major = form.major.data
        try:
            # Process the post with GPT
            gpt_output = process_post_with_gpt(post)
            if gpt_output.get("skip", False):
                error_message = "Please ensure you have filled out all 10 Extracurricular Activities and all 5 Awards/Honors."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('index.html', form=form)
            
            # If we pass the check, proceed with the prediction
            prediction = predict_acceptance(gpt_output, school, major)
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
        app.run(debug=True, port=5001)
