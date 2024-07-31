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
import logging
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.inference.inference import predict_acceptance, process_post_with_gpt

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['HCAPTCHA_SITE_KEY'] = os.getenv('HCAPTCHA_SITE_KEY')
app.config['HCAPTCHA_SECRET_KEY'] = os.getenv('HCAPTCHA_SECRET_KEY')
csrf = CSRFProtect(app)

KOFI_VERIFICATION_TOKEN = os.getenv('KOFI_VERIFICATION_TOKEN')

# Constants for API usage calculation
INPUT_TOKENS_PER_REQUEST = 5600
OUTPUT_TOKENS_PER_REQUEST = 360
TOTAL_TOKENS_PER_REQUEST = INPUT_TOKENS_PER_REQUEST + OUTPUT_TOKENS_PER_REQUEST
COST_PER_MILLION_INPUT_TOKENS = 3  # $3 per million input tokens
COST_PER_MILLION_OUTPUT_TOKENS = 15  # $15 per million output tokens

# File to store donations and request count
STORAGE_FILE = 'donation_data.json'

def load_data():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    else:
        app.logger.info("Could not find donation file, creating new one")
        initial_data = {'donations': [], 'request_count': 0, 'balance': 0}
        save_data(initial_data)
        return initial_data

def save_data(data):
    with open(STORAGE_FILE, 'w') as f:
        json.dump(data, f)

def update_balance(amount):
    if amount <= 0:
        app.logger.warning(f"Attempted to update balance with non-positive amount: {amount}")
        return None
    
    data = load_data()
    previous_balance = data['balance']
    data['balance'] += amount
    save_data(data)
    
    app.logger.info(f"Balance updated: {previous_balance} -> {data['balance']} (change: {amount})")
    return data['balance']

def update_request_count():
    data = load_data()
    
    input_cost = (INPUT_TOKENS_PER_REQUEST / 1000000) * COST_PER_MILLION_INPUT_TOKENS
    output_cost = (OUTPUT_TOKENS_PER_REQUEST / 1000000) * COST_PER_MILLION_OUTPUT_TOKENS
    request_cost = input_cost + output_cost
    
    if data['balance'] < request_cost:
        app.logger.warning(f"Insufficient balance for request. Current balance: {data['balance']}, Required: {request_cost}")
        return None
    
    data['request_count'] += 1
    previous_balance = data['balance']
    data['balance'] -= request_cost
    save_data(data)
    
    app.logger.info(f"Request processed. Count: {data['request_count']}, Balance: {previous_balance} -> {data['balance']}")
    return data['request_count']

def get_estimated_requests_left():
    data = load_data()
    current_balance = data['balance']
    
    input_cost = (INPUT_TOKENS_PER_REQUEST / 1000000) * COST_PER_MILLION_INPUT_TOKENS
    output_cost = (OUTPUT_TOKENS_PER_REQUEST / 1000000) * COST_PER_MILLION_OUTPUT_TOKENS
    cost_per_request = input_cost + output_cost
    
    if current_balance <= 0:
        return 0
    
    return int(current_balance / cost_per_request)

@app.route('/webhook', methods=['POST'])
@csrf.exempt
def kofi_webhook():
    app.logger.info(f"Received donation webhook: {request}")    
    data = request.form.get('data')
    if data:
        try:
            donation_data = json.loads(data)
            
            # Check the verification token
            if donation_data.get('verification_token') != KOFI_VERIFICATION_TOKEN:
                app.logger.warning("Invalid verification token received")
                return 'Invalid verification token', 401

            if donation_data['type'] in ['Donation', 'Subscription']:
                stored_data = load_data()
                donation_amount = float(donation_data['amount'])
                new_balance = update_balance(donation_amount)
                if new_balance is not None:
                    stored_data['donations'].append({
                        'name': donation_data['from_name'],
                        'amount': donation_amount,
                        'message': donation_data['message'],
                        'timestamp': int(time.time())
                    })
                    save_data(stored_data)
                    app.logger.info(f"Donation processed: {donation_amount} from {donation_data['from_name']}")
                    return '', 200
                else:
                    app.logger.warning(f"Failed to process donation: {donation_amount} from {donation_data['from_name']}")
                    return '', 400
        except json.JSONDecodeError:
            app.logger.error("Failed to decode JSON in webhook data")
        except Exception as e:
            app.logger.error(f"Error processing webhook: {str(e)}")
    return '', 400

@app.route('/get_updates')
def get_updates():
    data = load_data()
    return jsonify({
        'donations': data['donations'],
        'requests_left': get_estimated_requests_left(),
        'balance': data['balance']
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

        if update_request_count() is None:
            error_message = "Insufficient balance to process request. Please try again later or consider donating <3"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': error_message}), 400
            flash(error_message, 'error')
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
        print("Webapp running")
        app.run(host='0.0.0.0')
    else:
        app.run(debug=True, port=5001)