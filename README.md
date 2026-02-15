💳 Online Payments Fraud Detection using Machine Learning

Live Demo 👉 https://payment-fraud-detection-ml.onrender.com/

GitHub Repo 👉 https://github.com/khadeercodes/online-payments-fraud-detection-ml

📌 Project Overview

This project is a Machine Learning based web application that detects fraudulent online payment transactions.

The user enters transaction details through a web interface.
The trained ML model analyzes the transaction and predicts whether it is:

✔ Legitimate Transaction
⚠ Fraudulent Transaction

The system is built using Python, Scikit-Learn, Flask, HTML, CSS and deployed online.

🎯 Objectives

Understand real-world financial fraud detection

Perform data preprocessing & analysis

Train multiple ML models and compare performance

Deploy a trained model using Flask

Create a working web application

🧠 Machine Learning Workflow
1️⃣ Data Collection

Dataset: Online Payment Transactions Dataset
File used:

PS_20174392719_1491204439457_log.csv


Features used:

Feature	Description
step	Time step of transaction
type	Transaction type
amount	Transaction amount
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
oldbalanceDest	Receiver balance before transaction
newbalanceDest	Receiver balance after transaction

Target:

isFraud

2️⃣ Data Preprocessing

Removed unnecessary columns

Handled categorical values

Checked missing values

Feature selection

Train-test split

3️⃣ Model Training

Models compared:

Logistic Regression

Decision Tree

Random Forest ✅ (Best Accuracy Selected)

The best model is saved as:

fraud_model.pkl

4️⃣ Application Flow

User opens website

Enters transaction details

Flask sends data to ML model

Model predicts fraud or legitimate

Result displayed on UI

🖥️ Tech Stack
Category	Technology
Language	Python
ML Library	Scikit-Learn
Backend	Flask
Frontend	HTML, CSS
Data Handling	Pandas, NumPy
Deployment	Render
Version Control	Git & GitHub
📂 Project Structure
online-payments-fraud-detection-ml/
│
├── app.py
├── train_model.py
├── test_model.py
├── fraud_model.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
│
├── data/
│   └── dataset.csv
│
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── submit.html
│
├── static/
│   └── style.css
│
└── README.md

▶️ Run Locally
1. Install Dependencies
pip install -r requirements.txt

2. Train Model
python train_model.py

3. Run Application
python app.py


Open:

http://127.0.0.1:5000

☁️ Deployment (Render)

The project is deployed on Render cloud platform.

Build Command:

pip install -r requirements.txt


Start Command:

gunicorn app:app

📊 Sample Input
step: 1
type: TRANSFER
amount: 10000
oldbalanceOrg: 15000
newbalanceOrig: 5000
oldbalanceDest: 0
newbalanceDest: 10000

📈 Expected Output
Legitimate Transaction
or
Fraudulent Transaction

👨‍💻 Author

Shaik Khadeer
B.Tech Computer Science Student

📜 Conclusion

This project demonstrates how Machine Learning can be applied to detect financial fraud in real-time systems.
It combines data science + backend development + deployment, making it a complete end-to-end AI application.
