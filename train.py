# train.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import pickle

def train_model(data_path):
    # Load training data
    train_data = pd.read_csv(data_path)
    X = train_data.drop('target', axis=1)
    y = train_data['target']

    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return 'model.pkl'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    train_model(args.data)