import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, target_col: str, max_unique_cat: int = 20):
    df = df.copy()
    
    # Check if target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe")
    
    # Drop ID-like columns
    id_cols = [col for col in df.columns if any(x in col.lower() for x in ["id","uuid","hash","index","timestamp"])]
    df.drop(columns=id_cols, inplace=True, errors='ignore')
    
    # Drop columns with >20% missing values
    threshold = 0.2
    df = df.loc[:, df.isnull().mean() <= threshold]
    
    # Drop categorical columns with too many unique values
    for col in df.select_dtypes(include=['object','category']).columns:
        if df[col].nunique() > max_unique_cat:
            df.drop(columns=[col], inplace=True)
    
    # Fill missing values
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Drop constant columns
    const_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != target_col]
    df.drop(columns=const_cols, inplace=True, errors='ignore')
    
    # Encode categorical columns with few unique values
    for col in df.select_dtypes(include=['object','category']).columns:
        if df[col].nunique() <= max_unique_cat:
            df[col] = df[col].astype('category').cat.codes
    
    # Drop columns with very low correlation with target (<0.01)
    if df[target_col].dtype in ['int64','float64']:
        corr = df.corr()[target_col].abs()
        low_corr_cols = [c for c in corr[corr < 0.01].index if c != target_col]
        df.drop(columns=low_corr_cols, inplace=True, errors='ignore')
    
    # Drop one of two columns with high correlation (>0.9)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
    df.drop(columns=high_corr_cols, inplace=True, errors='ignore')
    
    # Clip numeric outliers (IQR) - exclude target column
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for col in numeric_cols:
        if col == target_col:  # Skip target column
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        df[col] = df[col].clip(low, high)
    
    # Scale numeric columns - EXCLUDE TARGET COLUMN
    numeric_cols_to_scale = [col for col in numeric_cols if col != target_col]
    if numeric_cols_to_scale:
        scaler = StandardScaler()
        df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    
    return df, df.drop(columns=[target_col]), df[target_col]