# eval_models.py
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import metrics, guess_feature_columns
from models_v2 import MatPhys_GA
# from modelv2_without_se import MatPhys_GA
# from modelv2_without_token0 import MatPhys_GA


class DmaxDataset(Dataset):
    """Torch Dataset for Dmax regression (optionally log1p-transforming the target)."""

    def __init__(self, X, y, use_log1p: bool = False):
        self.X = X.astype(np.float32)
        self.y_raw = y.astype(np.float32).reshape(-1, 1)
        self.use_log1p = use_log1p
        self.y = np.log1p(self.y_raw) if use_log1p else self.y_raw

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_to_simplex(X: np.ndarray) -> np.ndarray:
    """Clamp to non-negative and renormalize each row to sum to 1 (simplex)."""
    X = np.clip(X, 0, None)
    X = X / (X.sum(axis=1, keepdims=True) + 1e-8)
    return X


def load_test_df(test_path: str) -> pd.DataFrame:
    """Load test data from a CSV file or from a directory containing multiple CSVs."""
    if os.path.isdir(test_path):
        files = sorted(glob.glob(os.path.join(test_path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in: {test_path}")
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, axis=0, ignore_index=True)

    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"test_path not found: {test_path}")
    return pd.read_csv(test_path)


def find_checkpoints(model_dir: str):
    """Recursively scan a directory for checkpoint files (.pt/.pth/.ckpt)."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    pats = ["**/*.pt", "**/*.pth", "**/*.ckpt"]
    files = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(model_dir, pat), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under: {model_dir}")
    return files


@torch.no_grad()
def evaluate_model(model, loader, device, use_log1p: bool):
    """Run inference on a DataLoader and compute regression metrics (optionally inverting log1p)."""
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        ys.append(y.numpy())
        ps.append(pred.cpu().numpy())

    y = np.vstack(ys)
    p = np.vstack(ps)

    if use_log1p:
        y = np.expm1(y)
        p = np.expm1(p)

    return metrics(y, p)


def build_model_from_cfg(cfg: dict, feat_cols: list, fallback):
    """Build MatPhys_GA from checkpoint cfg with argparse fallbacks."""
    model_dim = int(cfg.get("model_dim", fallback.model_dim))
    n_heads = int(cfg.get("n_heads", fallback.n_heads))
    dropout = float(cfg.get("dropout", fallback.dropout))
    K_proto = int(cfg.get("K_proto", fallback.K_proto))
    lambda_lap = float(cfg.get("lambda_lap", fallback.lambda_lap))

    return MatPhys_GA(
        num_tokens=len(feat_cols),
        feat_names=feat_cols,
        model_dim=model_dim,
        n_heads=n_heads,
        dropout=dropout,
        K_proto=K_proto,
        lambda_lap=lambda_lap,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        default="../dataset/test_data2.csv",
        help="Test CSV file or a directory containing multiple CSV files (all CSVs will be loaded).",
    )
    parser.add_argument("--target", type=str, default="Dmax")
    parser.add_argument(
        "--n_feat",
        type=int,
        default=45,
        help="Used by guess_feature_columns when feat_cols is missing in a checkpoint.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../saved_models/20260110_version2/seed=2/argument/smogn",
        help="Directory containing model checkpoint files (recursive scan).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="../predict_result/smogn.csv",
        help="Output CSV path for the aggregated evaluation results.",
    )
    parser.add_argument("--batch_size", type=int, default=256)

    # Fallback model hyperparameters (used if a checkpoint does not contain cfg)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K_proto", type=int, default=16)
    parser.add_argument("--lambda_lap", type=float, default=1e-3)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_test = load_test_df(args.test_path)
    ckpt_files = find_checkpoints(args.model_dir)
    print(f"Found {len(ckpt_files)} checkpoints.")

    rows = []
    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[SKIP] Failed to load: {ckpt_path} | {e}")
            continue

        # Supported checkpoint formats:
        # 1) {"model": state_dict, "feat_cols": ..., "cfg": ...}
        # 2) raw state_dict
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            feat_cols = ckpt.get("feat_cols", None)
            cfg = ckpt.get("cfg", {})
        else:
            state = ckpt
            feat_cols = None
            cfg = {}

        if feat_cols is None:
            feat_cols = guess_feature_columns(df_test, args.target, n_feat=args.n_feat)

        missing = [c for c in feat_cols + [args.target] if c not in df_test.columns]
        if missing:
            print(f"[SKIP] {ckpt_path} | Missing columns in test data: {missing[:10]} ...")
            continue

        X = normalize_to_simplex(df_test[feat_cols].values)
        y = df_test[args.target].values

        use_log1p = bool(cfg.get("use_log1p", False))
        ds = DmaxDataset(X, y, use_log1p=use_log1p)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        model = build_model_from_cfg(cfg, feat_cols, args).to(device)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"[SKIP] {ckpt_path} | state_dict mismatch: {e}")
            continue

        m = evaluate_model(model, loader, device, use_log1p=use_log1p)
        row = {
            "ckpt_path": ckpt_path,
            "r2": m["r2"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "use_log1p": use_log1p,
            "model_dim": cfg.get("model_dim", args.model_dim),
            "n_heads": cfg.get("n_heads", args.n_heads),
            "dropout": cfg.get("dropout", args.dropout),
            "K_proto": cfg.get("K_proto", args.K_proto),
            "lambda_lap": cfg.get("lambda_lap", args.lambda_lap),
        }
        rows.append(row)

        print(
            f"[OK] {os.path.basename(ckpt_path)} | "
            f"R2={row['r2']:.4f} RMSE={row['rmse']:.4f} MAE={row['mae']:.4f}"
        )

    if not rows:
        raise RuntimeError("No checkpoints were successfully evaluated.")

    df_out = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
