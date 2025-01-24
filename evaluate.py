 # evaluate.py
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import json
def evaluate_model(model_path, test_data_path):

    # Load model and test data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    return 'metrics.json'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--test-data', type=str)
   
args = parser.parse_args()
evaluate_model(args.model, args.test_data)