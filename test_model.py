import joblib, pandas as pd, traceback

def test():
    try:
        m = joblib.load('fraud_model.pkl')
        print('Model loaded:', type(m))
        sample = pd.DataFrame({
            'step':[1], 'type':['PAYMENT'], 'amount':[1000.0],
            'oldbalanceOrg':[500.0], 'newbalanceOrig':[600.0],
            'oldbalanceDest':[200.0], 'newbalanceDest':[300.0]
        })
        print('Sample dtypes:\n', sample.dtypes)
        print('Predict ->', m.predict(sample))
        try:
            print('Predict_proba ->', m.predict_proba(sample))
        except Exception as e:
            print('predict_proba error:', e)
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    test()
