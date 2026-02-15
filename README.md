# Online Payments Fraud Detection using Machine Learning

Simple Flask web app that trains a model to predict fraudulent online payments and serves a form to make predictions.

Features
- Train models (Logistic Regression, Decision Tree, Random Forest) and select best by accuracy
- Save best model pipeline to `fraud_model.pkl`
- Flask app with routes: `/` (home), `/predict` (form), `/submit` (result)

Getting started (local)
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the dataset at `data/PS_20174392719_1491204439457_log.csv` (already expected in the repo)

3. Train the model (this will create `fraud_model.pkl`):

```bash
python train_model.py
```

4. Run the Flask app locally:

```bash
python app.py
```

Then open http://127.0.0.1:5000

Push to GitHub
1. Initialize git and commit files

```bash
git init
git add .
git commit -m "Initial commit - fraud detection app"
git remote add origin <your-git-remote-url>
git push -u origin main
```

Deploy to Render
1. Create a new Web Service on Render and connect your GitHub repo
2. Set the build command: `pip install -r requirements.txt`
3. Set the start command: `gunicorn app:app`
4. Deploy
