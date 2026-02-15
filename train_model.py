import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


DATA_PATH = 'data/PS_20174392719_1491204439457_log.csv'
MODEL_PATH = 'fraud_model.pkl'


def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    # Keep only requested columns and the target
    cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
    df = df[cols].copy()
    df = df.dropna()
    X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    y = df['isFraud']
    return X, y


def build_preprocessor():
    numeric_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    categorical_features = ['type']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def train_and_select(X_train, X_test, y_train, y_test):
    pre = build_preprocessor()

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))
        pipelines[name] = pipe

    # Print comparison table
    print('\nModel accuracy comparison:')
    print('{:<20}{}'.format('Model', 'Accuracy'))
    print('-' * 30)
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print('{:<20}{:.4f}'.format(name, acc))

    # Choose best
    best_name, best_acc = max(results, key=lambda x: x[1])
    best_pipeline = pipelines[best_name]
    print(f"\nSelected best model: {best_name} with accuracy {best_acc:.4f}")
    return best_pipeline


def main():
    print('Loading data...')
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training models...')
    best_pipeline = train_and_select(X_train, X_test, y_train, y_test)

    print(f'Saving best model to {MODEL_PATH} ...')
    joblib.dump(best_pipeline, MODEL_PATH)
    print('Done.')


if __name__ == '__main__':
    main()
