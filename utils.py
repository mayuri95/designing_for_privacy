import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import Ridge
from typing import Tuple, Optional, Sequence

def load_wine_quality_red(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")
    if "quality" not in df.columns:
        raise ValueError("Expected a 'quality' column in winequality-red.csv")
    y = df["quality"].to_numpy(dtype=float)
    X = df.drop(columns=["quality"]).to_numpy(dtype=float)
    return X, y

def load_adult_census(
    csv_path: str,
    *,
    label_candidates: Sequence[str] = ("income", "class", "target", "salary"),
    positive_labels: Sequence[str] = (">50K", ">50K.", "1", " >50K", " >50K.", ">=50K"),
    negative_labels: Sequence[str] = ("<=50K", "<=50K.", "0", " <=50K", " <=50K."),
    assume_missing_headers: bool = True,
    drop_na: bool = True,
    one_hot_drop_first: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Adult (Census Income) dataset from a CSV and return (X, y).

    - Handles typical 'income' label variations (>50K vs <=50K; trailing periods).
    - Replaces '?' with NaN; optionally drops rows with any NaNs.
    - One-hot encodes categorical columns so X is numeric (float).
    - If the CSV has no headers, sets the canonical Adult column names.

    Returns:
        X: np.ndarray[float]  (one-hot encoded features)
        y: np.ndarray[float]  (binary: 1 for >50K, 0 otherwise)
    """
    # Canonical Adult column names (UCI order)
    canonical_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",  # label
    ]

    # Try reading with headers first
    try:
        df = pd.read_csv(csv_path, na_values="?", skipinitialspace=True)
    except Exception:
        # Fallback to semicolon or other separators if needed
        df = pd.read_csv(csv_path, sep=",", na_values="?", skipinitialspace=True)

    # If the file likely lacks headers, set canonical ones (when column count matches)
    if assume_missing_headers and df.columns.dtype == "int64":
        # Pandas may assign integer headers if none were present
        if df.shape[1] in (14, 15):  # 15 incl. label; 14 when label separated/unknown
            cols = canonical_cols[: df.shape[1]]
            df.columns = cols
    elif assume_missing_headers and not set(df.columns).intersection(label_candidates):
        # If no known label name but col count matches, apply canonical names
        if df.shape[1] in (14, 15):
            df.columns = canonical_cols[: df.shape[1]]

    # Normalize whitespace in string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"?": np.nan})

    # Find the label column
    label_col: Optional[str] = None
    for cand in label_candidates:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(
            f"Could not find a label column. Looked for {label_candidates}. "
            "Pass a CSV with an 'income' (or similar) column."
        )

    # Build y (binary 0/1)
    y_raw = df[label_col].astype(str).str.strip()
    y = y_raw.apply(lambda v: 1.0 if v in positive_labels else (0.0 if v in negative_labels else np.nan))

    # Drop rows with unknown/NaN label
    mask_valid_y = ~y.isna()
    df = df.loc[mask_valid_y].copy()
    y = y.loc[mask_valid_y].astype(float).to_numpy()

    # Features = everything except label
    X_df = df.drop(columns=[label_col])

    # Handle missing values before encoding (optional)
    # (Adult has '?' placeholders—already mapped to NaN above.)
    if drop_na:
        # If dropping, make sure to keep y aligned
        before = len(X_df)
        keep_mask = ~X_df.isna().any(axis=1)
        X_df = X_df.loc[keep_mask]
        y = y[keep_mask.to_numpy()]
        # If you want to know how many dropped:
        # print(f"Dropped {before - len(X_df)} rows due to NaNs")

    # One-hot encode categoricals, keep numeric as-is
    cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=one_hot_drop_first)

    # Ensure float dtype
    X = X_df.to_numpy(dtype=float)

    return X, y

def load_wine_quality_red(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")
    if "quality" not in df.columns:
        raise ValueError("Expected a 'quality' column in winequality-red.csv")
    y = df["quality"].to_numpy(dtype=float)
    X = df.drop(columns=["quality"]).to_numpy(dtype=float)
    return X, y

def load_bank():
    bank_marketing = fetch_ucirepo(id=222) 
      
    # data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets.iloc[:, 0]

    y = (y.astype(str).str.lower() == "yes").astype(int).to_numpy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.columns.difference(cat_cols)

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), list(num_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
        ],
        remainder="drop",
    )
    X = ct.fit_transform(X)
    return X, y


def ridge_1d(x, y, lam):
    h_s = sum([x[i]**2 for i in range(len(x))])
    xy = sum([x[i]*y[i] for i in range(len(x))])
    return xy / (h_s + lam)
def ridge_pred(x, w):
    return [sum(
        [x[ind][i]*w[i] for i in range(len(x[ind]))
    ]) for ind in range(len(x))]

def poisson_sample(X):
    num_pts = len(X)
    pts = []
    for i in range(num_pts):
        if np.random.rand() < 0.5:
            pts.append(i)
    return pts