import os
import pandas as pd
from core.config import DATA_DIR

def save_uploaded_file(file, filename: str) -> str:
    """Save uploaded file to the datasets folder."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(file)
    return filepath

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV into pandas DataFrame."""
    return pd.read_csv(filepath)
