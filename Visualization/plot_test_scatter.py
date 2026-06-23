#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test-set inference and example-style scatter plot with a unified colorbar range:

- x-axis: Measured Dmax
- y-axis: Predicted Dmax
- Black solid line: y = x
- Red dashed line: fitted line y = a*x + b
- Scatter color: |y_pred - y_true|, where smaller errors are blue and larger errors are red
- Upper-left text box: model name, R2/RMSE/MAE, and fitted equation
- Colorbar: supports --vmax_fixed to use a fixed upper limit, so that the deepest red
  corresponds to the same error magnitude across different figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter

# Import the model. Modify the import path according to your project structure.
from DPN.models_v2 import MatPhys_GA
# from DPN.modelv2_without_se import MatPhys_GA
# from DPN.modelv2_without_token0 import MatPhys_GA

# Global plotting style, designed to be close to the reference figure.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c

    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]

    raise KeyError(
        f"Could not find column from candidates: {candidates}\n"
        f"Existing columns: {list(df.columns)}"
    )


def renorm_simplex(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip values to be non-negative and normalize each row to sum to one."""
    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)

    safe = s > eps
    Xn = X.copy()

    Xn[safe[:, 0]] = X[safe[:, 0]] / (s[safe[:, 0]] + 1e-8)

    if np.any(~safe[:, 0]):
        Xn[~safe[:, 0]] = 1.0 / X.shape[1]

    return Xn


def fit_line(x: np.ndarray, y: np.ndarray):
    """
    Least-squares fitting for y = a*x + b.

    Here, x is the measured value and y is the predicted value.
    """
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def build_xy_from_ckpt_featcols(df: pd.DataFrame, feat_cols: list, target_col: str):
    """
    Align feature columns according to ckpt['feat_cols'].

    Missing columns are filled with zeros, and extra columns are ignored.
    """
    y = df[target_col].to_numpy(dtype=float).reshape(-1)

    X = np.zeros((len(df), len(feat_cols)), dtype=np.float32)
    name2i = {n: i for i, n in enumerate(feat_cols)}

    for col in df.columns:
        if col == target_col:
            continue

        if col in name2i:
            X[:, name2i[col]] = df[col].to_numpy(dtype=np.float32)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    return X, y, mask


@torch.no_grad()
def infer_dmax(
    model,
    X,
    device="cpu",
    batch_size=512,
    use_log1p=False,
    clamp_nonneg=True,
):
    model.eval()
    model.to(device)

    preds = []
    n = X.shape[0]

    for s in range(0, n, batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        out = model(xb).view(-1).detach().cpu().numpy().astype(np.float64)

        if use_log1p:
            out = np.expm1(out)

        if clamp_nonneg:
            out = np.maximum(out, 0.0)

        preds.append(out)

    return np.concatenate(preds, axis=0)


def pick_cbar_step(vmax: float) -> float:
    """
    Select the colorbar step size.

    This is designed to resemble the reference figure, usually using 0.5 intervals
    and one decimal place when appropriate.
    """
    vmax = float(vmax)

    if vmax <= 2.0:
        return 0.2
    if vmax <= 6.0:
        return 0.5
    if vmax <= 12.0:
        return 1.0
    if vmax <= 25.0:
        return 2.0

    return 5.0


def main():
    parser = argparse.ArgumentParser(
        description="Test-set inference and example-style scatter plot with an optional unified colorbar range"
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        default="../dataset/test_data2.csv",
        help="Test-set CSV file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Dmax",
        help="Target column name, default: Dmax",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="../saved_models/20260110_version2/seed=42/argument/smogn/r2=0_8871.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cpu or cuda",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size",
    )
    parser.add_argument(
        "--simplex",
        action="store_true",
        help="Whether to project compositions onto the simplex",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="DPN+SMOTE",
        help="Model name shown in the figure",
    )
    parser.add_argument(
        "--panel_tag",
        type=str,
        default="",
        help="Panel tag shown in the lower-right corner, e.g., (b)",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper limit of the colorbar for absolute error. If not set, the 95th percentile is used.",
    )
    parser.add_argument(
        "--vmax_fixed",
        type=float,
        default=5,
        help="Fixed upper limit of the colorbar for cross-figure comparison. Use the same value for all figures.",
    )

    parser.add_argument(
        "--save_pred_csv",
        type=str,
        default="",
        help="Optional path to save CSV with prediction results",
    )
    parser.add_argument(
        "--out_fig",
        type=str,
        default="./figures/smote_scatter.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure DPI",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)

    if args.save_pred_csv:
        os.makedirs(os.path.dirname(args.save_pred_csv), exist_ok=True)

    # Step 1: Load model checkpoint.
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    feat_cols = ckpt.get("feat_cols", None)

    if feat_cols is None:
        raise KeyError(
            "The checkpoint does not contain 'feat_cols', so input feature alignment cannot be performed."
        )

    use_log1p = bool(cfg.get("use_log1p", False))

    model = MatPhys_GA(
        num_tokens=len(feat_cols),
        feat_names=feat_cols,
        model_dim=cfg.get("model_dim", 128),
        n_heads=cfg.get("n_heads", 4),
        dropout=cfg.get("dropout", 0.1),
        K_proto=cfg.get("K_proto", 16),
        lambda_lap=cfg.get("lambda_lap", 1e-3),
    )
    model.load_state_dict(ckpt["model"], strict=True)

    show_name = args.model_name.strip()
    if not show_name:
        show_name = os.path.splitext(os.path.basename(args.ckpt))[0]

    # Step 2: Load test set and align input features.
    df = pd.read_csv(args.test_csv)
    target_col = pick_col(df, [args.target, "Dmax", "dmax", "y", "label"])
    X, y_true, row_mask = build_xy_from_ckpt_featcols(df, feat_cols, target_col)

    if args.simplex:
        X = renorm_simplex(X)

    # Step 3: Inference.
    y_pred = infer_dmax(
        model=model,
        X=X,
        device=args.device,
        batch_size=args.batch_size,
        use_log1p=use_log1p,
        clamp_nonneg=True,
    )

    # Step 4: Evaluation metrics.
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    a, b = fit_line(y_true, y_pred)

    # Step 5: Color points by absolute error.
    abs_err = np.abs(y_pred - y_true)

    if args.vmax_fixed is not None:
        vmax = float(args.vmax_fixed)
    elif args.vmax is not None:
        vmax = float(args.vmax)
    else:
        vmax = float(np.percentile(abs_err, 95))
        vmax = max(vmax, 1e-8)

    # Step 6: Plot the example-style scatter figure.
    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    vmin_xy = float(min(y_true.min(), y_pred.min()))
    vmax_xy = float(max(y_true.max(), y_pred.max()))
    pad = 0.04 * (vmax_xy - vmin_xy + 1e-12)
    lo = vmin_xy - pad
    hi = vmax_xy + pad

    sc = ax.scatter(
        y_true,
        y_pred,
        c=np.clip(abs_err, 0, vmax),
        cmap="coolwarm",
        vmin=0.0,
        vmax=vmax,
        s=28,
        alpha=0.85,
        edgecolors="none",
    )

    # Black solid y=x line.
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=2.2)

    # Red dashed fitted line.
    ax.plot(
        [lo, hi],
        [a * lo + b, a * hi + b],
        color="red",
        linestyle="--",
        linewidth=2.0,
    )

    ax.set_xlabel("Measured Dmax", fontsize=12)
    ax.set_ylabel("Predicted Dmax", fontsize=12)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    step = pick_cbar_step(vmax)

    ticks = np.arange(0.0, vmax + 0.5 * step, step)
    ticks = ticks[ticks <= vmax + 1e-9]

    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.update_ticks()

    text = (
        f"{show_name}\n"
        f"RMSE: {rmse:.3f}\n"
        f"MAE:  {mae:.3f}\n"
        f"$R^2$:  {r2:.3f}\n"
        f"Fit: $y={a:.3f}x{b:+.3f}$"
    )

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            linewidth=0.6,
        ),
    )

    if args.panel_tag.strip():
        ax.text(
            0.97,
            0.06,
            args.panel_tag.strip(),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
        )

    ax.grid(True, alpha=0.18)

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=args.dpi, bbox_inches="tight")

    print(f"Saved figure: {args.out_fig}")
    print(f"Test R2={r2:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f}")

    if args.vmax_fixed is not None:
        print(f"[Colorbar] Using fixed vmax = {vmax:.3f} for cross-figure comparison")
    else:
        print(f"[Colorbar] Using vmax = {vmax:.3f}")

    if args.save_pred_csv:
        out_df = df.loc[row_mask].copy()
        out_df["Dmax_pred"] = y_pred
        out_df.to_csv(args.save_pred_csv, index=False)
        print(f"Saved prediction CSV: {args.save_pred_csv}")

    plt.show()


if __name__ == "__main__":
    main()
