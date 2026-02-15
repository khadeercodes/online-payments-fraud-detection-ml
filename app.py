from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model pipeline
MODEL_PATH = 'fraud_model.pkl'
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/submit', methods=['POST'])
def submit():
    if model is None:
        return render_template('submit.html', result=None, error='Model not found. Run train_model.py first.')

    try:
        data = {
            'step': [float(request.form.get('step', 0))],
            'type': [request.form.get('type', '')],
            'amount': [float(request.form.get('amount', 0))],
            'oldbalanceOrg': [float(request.form.get('oldbalanceOrg', 0))],
            'newbalanceOrig': [float(request.form.get('newbalanceOrig', 0))],
            'oldbalanceDest': [float(request.form.get('oldbalanceDest', 0))],
            'newbalanceDest': [float(request.form.get('newbalanceDest', 0))]
        }

        X = pd.DataFrame(data)
        pred_raw = model.predict(X)[0]
        try:
            pred = int(pred_raw)
        except Exception:
            # fallback: if it's array-like, take first element
            try:
                pred = int(pred_raw[0])
            except Exception:
                pred = pred_raw

        proba = None
        try:
            p = model.predict_proba(X)[0]
            # convert numpy arrays to plain list for template safety
            try:
                proba = [float(x) for x in p]
            except Exception:
                proba = p.tolist() if hasattr(p, 'tolist') else p
        except Exception:
            proba = None

        if pred == 1:
            label = 'Fraud Transaction'
            color = 'danger'
        else:
            label = 'Legitimate Transaction'
            color = 'success'

        return render_template('submit.html', result=label, color=color, proba=proba)

    except Exception as e:
        return render_template('submit.html', result=None, error=str(e))


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
