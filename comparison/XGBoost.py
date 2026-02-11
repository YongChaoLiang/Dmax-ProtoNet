import argparse
import re
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost is not installed: please run `pip install xgboost` first.") from e


def renorm_simplex(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip to non-negative values and renormalize each row to sum to 1 (simplex)."""
    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)
    safe = s > eps
    Xn = X.copy()
    Xn[safe[:, 0]] = X[safe[:, 0]] / (s[safe[:, 0]] + 1e-8)
    if np.any(~safe[:, 0]):
        Xn[~safe[:, 0]] = 1.0 / X.shape[1]
    return Xn


def pick_target_col(df: pd.DataFrame, target: str) -> str:
    """Resolve the target column name with case-insensitive fallback."""
    if target in df.columns:
        return target
    low = {c.lower(): c for c in df.columns}
    if target.lower() in low:
        return low[target.lower()]
    raise KeyError(f"Target column '{target}' was not found. Available columns: {list(df.columns)}")


def pick_element_cols(df: pd.DataFrame, target_col: str):
    """
    Keep only element-symbol-like columns: 1 uppercase letter + optional 1 lowercase letter (e.g., Al, Cu, Zr),
    and require numeric dtype.
    """
    pat = re.compile(r"^[A-Z][a-z]?$")
    cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if pat.match(str(c)) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if len(cols) == 0:
        raise ValueError(
            "No element composition columns were detected (element-symbol columns). "
            "Please ensure feature columns are named as chemical element symbols."
        )
    return cols


def load_xy(csv_path: str, target: str):
    """Load X/y from a CSV and drop rows containing non-finite values."""
    df = pd.read_csv(csv_path)
    target_col = pick_target_col(df, target)
    feat_cols = pick_element_cols(df, target_col)

    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float).reshape(-1)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_cols


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost regression for Dmax prediction and test-set R2/RMSE/MAE (evaluation only)."
    )
    parser.add_argument("--train_csv", type=str, default="../dataset/train_data2.csv",
                        help="Path to the training CSV.")
    parser.add_argument("--test_csv", type=str, default="../dataset/test_data2.csv",
                        help="Path to the test CSV.")
    parser.add_argument("--target", type=str, default="Dmax",
                        help="Target column name (default: Dmax).")

    parser.add_argument("--simplex", action="store_true",
                        help="Project compositions onto the simplex (non-negative and row-sum=1).")

    parser.add_argument("--n_estimators", type=int, default=1500,
                        help="Number of boosting rounds / trees.")
    parser.add_argument("--learning_rate", type=float, default=0.02,
                        help="Learning rate.")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum tree depth.")
    parser.add_argument("--min_child_weight", type=float, default=1.0,
                        help="Minimum sum of instance weight (hessian) needed in a child.")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Row subsampling ratio.")
    parser.add_argument("--colsample_bytree", type=float, default=0.8,
                        help="Column subsampling ratio.")
    parser.add_argument("--reg_alpha", type=float, default=0.0,
                        help="L1 regularization.")
    parser.add_argument("--reg_lambda", type=float, default=1.0,
                        help="L2 regularization.")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Minimum loss reduction required to make a further partition on a leaf node.")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel threads.")
    args = parser.parse_args()

    Xtr, ytr, feat_tr = load_xy(args.train_csv, args.target)
    Xte, yte, feat_te = load_xy(args.test_csv, args.target)

    # Align train/test features using the training column order
    if feat_tr != feat_te:
        te_set = set(feat_te)
        common = [c for c in feat_tr if c in te_set]
        if len(common) == 0:
            raise ValueError(
                "Train and test sets do not share any element columns with the same names; cannot align features."
            )

        df_tr = pd.read_csv(args.train_csv)
        df_te = pd.read_csv(args.test_csv)
        tr_target = pick_target_col(df_tr, args.target)
        te_target = pick_target_col(df_te, args.target)

        Xtr = df_tr[common].to_numpy(dtype=float)
        ytr = df_tr[tr_target].to_numpy(dtype=float).reshape(-1)
        Xte = df_te[common].to_numpy(dtype=float)
        yte = df_te[te_target].to_numpy(dtype=float).reshape(-1)

        mtr = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
        mte = np.isfinite(yte) & np.all(np.isfinite(Xte), axis=1)
        Xtr, ytr = Xtr[mtr], ytr[mtr]
        Xte, yte = Xte[mte], yte[mte]

    if args.simplex:
        Xtr = renorm_simplex(Xtr)
        Xte = renorm_simplex(Xte)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
        verbosity=0,
    )
    model.fit(Xtr, ytr)

    y_pred = model.predict(Xte)

    r2 = float(r2_score(yte, y_pred))
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae = float(mean_absolute_error(yte, y_pred))

    print("[XGBoost]")
    print(f"n_estimators={args.n_estimators} lr={args.learning_rate} max_depth={args.max_depth} "
          f"subsample={args.subsample} colsample_bytree={args.colsample_bytree}")
    print(f"Test R2   = {r2:.6f}")
    print(f"Test RMSE = {rmse:.6f}")
    print(f"Test MAE  = {mae:.6f}")


if __name__ == "__main__":
    main()
