from flask import Flask, request, render_template
import joblib
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Set the Flask secret key from the environment variable
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

# Load Models and Scaler
decision_tree = joblib.load('models/decision_tree_model.pkl')
svm = joblib.load('models/svm_model.pkl')
gradient_boosting = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get Input Data
        input_features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([input_features])

        # Predictions
        dt_prediction = decision_tree.predict(scaled_features)[0]
        svm_prediction = svm.predict(scaled_features)[0]
        gb_prediction = gradient_boosting.predict(scaled_features)[0]

        # Render the result page with predictions
        return render_template(
            'result.html',
            dt_prediction=dt_prediction,
            svr_prediction=svm_prediction,
            gb_prediction=gb_prediction
        )
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
