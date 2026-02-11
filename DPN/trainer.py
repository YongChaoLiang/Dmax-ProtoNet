# trainer.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import set_seed, metrics, guess_feature_columns
from models_v2 import MatPhys_GA
# from modelv2_without_se import MatPhys_GA
# from modelv2_without_token0 import MatPhys_GA
# from models_v3 import MatPhys_GA
from augment import apply_augment, smogn_preprocess


class DmaxDataset(Dataset):
    """Torch Dataset for composition vectors and regression targets (optionally log1p-transformed)."""

    def __init__(self, X, y, use_log1p=False):
        self.X = X.astype(np.float32)
        self.y_raw = y.astype(np.float32).reshape(-1, 1)
        self.use_log1p = use_log1p
        self.y = np.log1p(self.y_raw) if use_log1p else self.y_raw

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_one_epoch(model, loader, optimizer, device, criterion, aug_conf):
    """Run one training epoch with optional online augmentation."""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        X, y = apply_augment(
            X, y,
            use_mixup=aug_conf.get("mixup", False),
            mixup_alpha=aug_conf.get("mixup_alpha", 0.4),
            use_gn=aug_conf.get("gn", False),
            gn_std=aug_conf.get("gn_std", 0.01),
            use_smote=aug_conf.get("smote", False),
            smote_frac=aug_conf.get("smote_frac", 0.0),
        )

        pred = model(X)
        loss = criterion(pred, y)
        if hasattr(model, "reg_loss"):
            loss = loss + model.reg_loss()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device, use_log1p: bool):
    """Evaluate on a dataloader and return metric dict (optionally inverse log1p)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
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


def parse_aug(arg: str):
    """Parse --aug string into an augmentation config dict."""
    conf = {
        "mixup": False,
        "mixup_alpha": 0.4,
        "gn": False,
        "gn_std": 0.01,
        "smote": False,
        "smote_frac": 0.3,
    }
    if not arg or arg.lower() == "none":
        return conf
    for p in arg.split(","):
        p = p.strip().lower()
        if p == "mixup":
            conf["mixup"] = True
        elif p == "gn":
            conf["gn"] = True
        elif p == "smote":
            conf["smote"] = True
    return conf


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_csv", type=str, default="../dataset/train_data2.csv")
    parser.add_argument("--val_csv", type=str, default="../dataset/test_data2.csv")
    parser.add_argument("--target", type=str, default="Dmax")

    # Model
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K_proto", type=int, default=16)
    parser.add_argument("--lambda_lap", type=float, default=1e-3)

    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use_log1p", action="store_true")
    parser.add_argument(
        "--save",
        type=str,
        default="../saved_models/20260110_version2/seed=2/ablation/without_token0/best.pt",
    )
    parser.add_argument("--patience", type=int, default=100)

    # Online augmentation
    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        help="none | mixup | gn | smote | mixup,gn | mixup,gn,smote",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--gn_std", type=float, default=0.0005)
    parser.add_argument("--smote_frac", type=float, default=0.3)

    # Offline preprocessing augmentation
    parser.add_argument(
        "--pre_smogn_frac",
        type=float,
        default=0.0,
        help="Fraction of new samples to generate in preprocessing (e.g., 0.3 means +30%).",
    )
    parser.add_argument("--pre_smogn_q", type=float, default=0.0)
    parser.add_argument(
        "--pre_smogn_tail",
        type=str,
        default="high",
        help="low | high | both",
    )

    args = parser.parse_args()
    set_seed(2)

    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)
    df_all = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
    feat_cols = guess_feature_columns(df_all, args.target, n_feat=45)

    X_train = df_tr[feat_cols].values
    y_train = df_tr[args.target].values
    X_val = df_va[feat_cols].values
    y_val = df_va[args.target].values

    # Normalize compositions to the simplex
    X_train = np.clip(X_train, 0, None)
    X_train = X_train / (X_train.sum(axis=1, keepdims=True) + 1e-8)
    X_val = np.clip(X_val, 0, None)
    X_val = X_val / (X_val.sum(axis=1, keepdims=True) + 1e-8)

    if args.pre_smogn_frac > 0:
        X_train, y_train = smogn_preprocess(
            X_train,
            y_train,
            frac=args.pre_smogn_frac,
            q=args.pre_smogn_q,
            tail=args.pre_smogn_tail,
            seed=42,
        )

    train_ds = DmaxDataset(X_train, y_train, use_log1p=args.use_log1p)
    val_ds = DmaxDataset(X_val, y_val, use_log1p=args.use_log1p)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MatPhys_GA(
        num_tokens=len(feat_cols),
        feat_names=feat_cols,
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        K_proto=args.K_proto,
        lambda_lap=args.lambda_lap,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)

    aug_conf = parse_aug(args.aug)
    aug_conf["mixup_alpha"] = args.mixup_alpha
    aug_conf["gn_std"] = args.gn_std
    aug_conf["smote_frac"] = args.smote_frac

    best_r2 = -1e9
    no_improve = 0
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, aug_conf)
        val_metrics = evaluate(model, val_loader, device, use_log1p=args.use_log1p)

        print(
            f"[{epoch:03d}] loss={train_loss:.4f} | "
            f"val R2={val_metrics['r2']:.4f} RMSE={val_metrics['rmse']:.4f} MAE={val_metrics['mae']:.4f}"
        )

        if val_metrics["r2"] > best_r2 + 1e-5:
            best_r2 = val_metrics["r2"]
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "feat_cols": feat_cols, "cfg": vars(args)},
                args.save,
            )
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    print(f"Best R2 = {best_r2:.4f}")


if __name__ == "__main__":
    main()
