import pandas as pd

def load_datasets(paths):
    """Load datasets from given file paths."""
    dataframes = [pd.read_csv(p) for p in paths]
    return dataframes
