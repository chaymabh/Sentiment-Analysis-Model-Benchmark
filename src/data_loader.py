import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_dataset(zip_path, extract_to='data'):
    """
    Extracts the dataset from a zip file if the target folder doesn't already exist.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Dataset extraction completed.")

def load_dataset(csv_path):
    """
    Loads the dataset from a CSV file.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """
    Splits data into training, validation, and test sets.
    First splits the data into train and temporary (validation + test),
    then splits the temporary set equally into validation and test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Testing set: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
