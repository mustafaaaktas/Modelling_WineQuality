import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data


def data_overview(data):
    print("First few rows of the dataset:\n", data.head(), "\n")
    print("Dataset information:")
    data.info()
    print("\nStatistical summary:\n", data.describe(), "\n")
    print("Column names:", data.columns.tolist())
    print("\nMissing values per column:\n", data.isna().sum())
    print("\nUnique values per column:\n", data.nunique())
