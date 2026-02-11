# utils.py
import numpy as np
import pandas as pd
import torch


# Element symbols used to identify composition feature columns
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
]


def set_seed(seed: int = 42):
    """Set random seeds for Python/NumPy/PyTorch to improve reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics (R2 / RMSE / MAE) for 1D targets."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": float(r2), "rmse": rmse, "mae": mae}


def guess_feature_columns(df: pd.DataFrame, target: str, n_feat: int = 45):
    """Infer feature columns for composition vectors, prioritizing element-symbol columns."""
    cols = [c for c in df.columns if c != target]

    symbol_cols = [c for c in cols if c in ELEMENT_SYMBOLS]
    if len(symbol_cols) >= n_feat:
        return symbol_cols[:n_feat]

    e_cols = [f"E{i}" for i in range(1, n_feat + 1)]
    if all(c in df.columns for c in e_cols):
        return e_cols

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < n_feat:
        raise ValueError("Not enough numeric feature columns found. Please check the CSV file.")
    return num_cols[:n_feat]
