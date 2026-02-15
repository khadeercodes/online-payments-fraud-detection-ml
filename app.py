from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "fraud_model.pkl"

# ---------- AUTO TRAIN MODEL ON SERVER ----------
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training model now...")
    import train_model   # runs train_model.py and creates fraud_model.pkl

# Load trained model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")


# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/submit', methods=['POST'])
def submit():
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

        pred = int(model.predict(X)[0])

        if pred == 1:
            label = "⚠ Fraudulent Transaction"
            color = "danger"
        else:
            label = "✔ Legitimate Transaction"
            color = "success"

        return render_template('submit.html', result=label, color=color)

    except Exception as e:
        return render_template('submit.html', result=None, error=str(e))


# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)
