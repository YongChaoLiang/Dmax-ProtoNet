# utils.py
import numpy as np
import pandas as pd
import torch
import re

NON_FEATURE_COLUMNS = {
    "Dmax", "dmax", "D_max", "target",
    "index", "Index", "Unnamed: 0", "unnamed: 0"
}
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": float(r2), "rmse": rmse, "mae": mae}

def guess_feature_columns(df: pd.DataFrame, target: str = "Dmax", n_feat: int = 45):
    """
    Select element composition columns as model input features.

    Rules:
    1. Exclude target column.
    2. Exclude non-physical columns such as index and Unnamed columns.
    3. Keep only chemical-element-like column names, e.g., Al, Cu, Zr.
    4. Require the number of selected feature columns to be exactly n_feat.
    """

    element_pattern = re.compile(r"^[A-Z][a-z]?$")

    exclude_cols = {
        target,
        str(target).lower(),
        "Dmax",
        "dmax",
        "D_max",
        "target",
        "index",
        "Index",
        "Unnamed: 0",
        "unnamed: 0",
    }

    feat_cols = []

    for col in df.columns:
        col_str = str(col)

        # Exclude the target column and non-physical columns
        if col_str in exclude_cols:
            continue

        # Exclude all index-like columns starting with "Unnamed"
        if col_str.startswith("Unnamed"):
            continue

        # Keep only numeric columns with chemical element-symbol-like names
        if element_pattern.match(col_str) and pd.api.types.is_numeric_dtype(df[col]):
            feat_cols.append(col_str)

    if len(feat_cols) != n_feat:
        raise ValueError(
            f"Expected {n_feat} element composition columns, "
            f"but found {len(feat_cols)}.\n"
            f"Selected columns: {feat_cols}\n"
            f"All columns in dataframe: {list(df.columns)}"
        )

    print(f"[INFO] Selected {len(feat_cols)} feature columns:")
    print(feat_cols)

    return feat_cols
