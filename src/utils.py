import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(df, target_column):
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Normalize features
    X = X.astype("float32") / 255.0

    return X, y, encoder
