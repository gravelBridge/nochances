import json
import os
from flask import Flask, render_template, flash, redirect, url_for, request, jsonify
from flask.json import jsonify
import json
import numpy as np
from flask_wtf import CSRFProtect
from dotenv import load_dotenv
import os
import sys
from forms import PredictionForm
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.inference.inference import predict_acceptance, process_post_with_gpt

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['HCAPTCHA_SITE_KEY'] = os.getenv('HCAPTCHA_SITE_KEY')
app.config['HCAPTCHA_SECRET_KEY'] = os.getenv('HCAPTCHA_SECRET_KEY')
csrf = CSRFProtect(app)

# Constants for API usage calculation
INITIAL_BALANCE = 95  # Initial balance in dollars
TOKENS_PER_REQUEST = 5600 + 360  # Input + Output tokens
COST_PER_MILLION_TOKENS = (3 * 5600 + 15 * 360) / 1000000  # Cost per request in dollars

# File to store donations and request count
STORAGE_FILE = '/home/ubuntu/nochances/webapp/donation_data.json'

def load_data():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    return {'donations': [], 'request_count': 0}

def save_data(data):
    with open(STORAGE_FILE, 'w') as f:
        json.dump(data, f)

def update_request_count():
    data = load_data()
    data['request_count'] += 1
    save_data(data)
    return data['request_count']

def get_estimated_requests_left():
    data = load_data()
    used_amount = data['request_count'] * COST_PER_MILLION_TOKENS * (TOKENS_PER_REQUEST / 1000000)
    remaining_amount = INITIAL_BALANCE - used_amount
    return max(0, int(remaining_amount / (COST_PER_MILLION_TOKENS * (TOKENS_PER_REQUEST / 1000000))))

@app.route('/webhook', methods=['POST'])
@csrf.exempt
def kofi_webhook():
    data = request.form.get('data')
    print("Received donation webhook:", str(data))
    if data:
        try:
            donation_data = json.loads(data)
            if donation_data['type'] in ['Donation', 'Subscription']:
                stored_data = load_data()
                stored_data['donations'].append({
                    'name': donation_data['from_name'],
                    'amount': donation_data['amount'],
                    'message': donation_data['message'],
                    'timestamp': int(time.time())
                })
                save_data(stored_data)
                return '', 200
        except json.JSONDecodeError:
            pass
    return '', 400

@app.route('/get_updates')
def get_updates():
    data = load_data()
    return jsonify({
        'donations': data['donations'],
        'requests_left': get_estimated_requests_left()
    })

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

        update_request_count() # Update the request count when processing a form
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