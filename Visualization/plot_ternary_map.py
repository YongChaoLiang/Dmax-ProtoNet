import os
import torch
import pandas as pd
import numpy as np

from DPN.models_v2 import MatPhys_GA


# ====== 1. Basic settings ======
CKPT_PATH = "../saved_models/20260110_version2/seed=0/argument/mixup/r2=0_8941.pt"
OUT_CSV = "./predict_result/seed=0_mixup/ternary_Ca_Mg_Cu_pred.csv"

# Elements to scan in the ternary composition space
A, B, C = "Ca", "Mg", "Cu"

# Composition step size. For example, 0.01 means 1%.
STEP = 0.01


def ensure_parent_dir(file_path: str):
    """Create the parent directory of the output file if it does not exist."""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


# ====== 2. Load model checkpoint ======
ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg = ckpt["cfg"]
feat_cols = ckpt["feat_cols"]

use_log1p = cfg.get("use_log1p", False)

model = MatPhys_GA(
    num_tokens=len(feat_cols),
    feat_names=feat_cols,
    model_dim=cfg.get("model_dim", 128),
    n_heads=cfg.get("n_heads", 4),
    dropout=cfg.get("dropout", 0.1),
    K_proto=cfg.get("K_proto", 16),
    lambda_lap=cfg.get("lambda_lap", 1e-3),
)

model.load_state_dict(ckpt["model"])
model.eval()

name2idx = {name: i for i, name in enumerate(feat_cols)}

# Check whether the selected ternary elements are included in the model feature list
missing_elements = [elem for elem in [A, B, C] if elem not in name2idx]
if missing_elements:
    raise ValueError(
        f"The following elements are not found in feat_cols: {missing_elements}\n"
        f"Available feat_cols: {feat_cols}"
    )


# ====== 3. Generate ternary grid and make predictions ======
rows = []
steps = np.arange(0.0, 1.0 + 1e-6, STEP)

with torch.no_grad():
    for a in steps:
        for b in steps:
            c = 1.0 - a - b

            if c < -1e-8:
                continue

            if c < 0:
                c = 0.0

            total = a + b + c
            a_, b_, c_ = a / total, b / total, c / total

            x = np.zeros(len(feat_cols), dtype=np.float32)
            x[name2idx[A]] = a_
            x[name2idx[B]] = b_
            x[name2idx[C]] = c_

            t = torch.from_numpy(x).unsqueeze(0)
            pred = model(t).item()

            # Post-processing
            if use_log1p:
                # If the model was trained on log(1 + Dmax), convert it back to Dmax.
                dmax = np.expm1(pred)
            else:
                # If the model directly regresses Dmax, clamp negative predictions to zero.
                dmax = max(pred, 0.0)

            rows.append({
                A: a_,
                B: b_,
                C: c_,
                "Dmax_pred": dmax
            })


# ====== 4. Save prediction results ======
df = pd.DataFrame(rows)

ensure_parent_dir(OUT_CSV)
df.to_csv(OUT_CSV, index=False)

print(f"Saved to: {OUT_CSV}")
print(f"Number of grid points: {len(df)}")
