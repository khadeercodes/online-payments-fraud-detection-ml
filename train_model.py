import os
import gdown
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- DOWNLOAD DATASET ----------------
DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"
FILE_ID = "1_m8oUeX-yj4aTD-sGQ337la4x5-jV-rN"

os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading dataset using gdown...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", DATA_PATH, quiet=False)
    print("Dataset downloaded successfully!")

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Columns found in dataset:", df.columns.tolist())

# Keep only required columns
df = df[['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']]

# ---------------- PREPROCESSING ----------------
# Encode transaction type
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Features & Target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ----------------
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "fraud_model.pkl")
print("fraud_model.pkl saved successfully!")
