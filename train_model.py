import os
import urllib.request
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- DOWNLOAD DATASET ----------------
DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"
DATA_URL = "https://drive.google.com/uc?export=download&id=1_m8oUeX-yj4aTD-sGQ337la4x5-jV-rN"

os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading dataset from Google Drive...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Dataset downloaded successfully!")

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Keep only required columns
df = df[['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']]

# Encode transaction type
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# ---------------- SPLIT DATA ----------------
X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- TRAIN MODEL ----------------
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "fraud_model.pkl")
print("fraud_model.pkl saved successfully!")
