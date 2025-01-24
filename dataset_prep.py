import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def prepare_data():
    
     # Load sample diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save to files
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    return 'train.csv', 'test.csv'

if __name__ == '__main__':
    prepare_data()