# utils_augment.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import math
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable

import numpy as np

try:
    import torch
except Exception:
    torch = None  # Allow offline-only augmentation in pure NumPy environments


def _dt_tag():
    """Return a timestamp tag for output file naming."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_npz(X: np.ndarray, y: np.ndarray, out_dir: str, prefix: str, meta: Dict[str, Any]):
    """Save (X, y) to a compressed NPZ file plus a JSON metadata sidecar."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{prefix}_{_dt_tag()}"
    npz_path = Path(out_dir) / f"{tag}.npz"
    json_path = Path(out_dir) / f"{tag}.json"
    np.savez_compressed(npz_path, X=X.astype(np.float32), y=y.astype(np.float32))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(npz_path), str(json_path)


def save_csv(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    prefix: str,
    meta: Dict[str, Any],
    columns: Optional[list] = None,
):
    """Save (X, y) to CSV plus a JSON metadata sidecar."""
    import pandas as pd
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{prefix}_{_dt_tag()}"
    csv_path = Path(out_dir) / f"{tag}.csv"
    json_path = Path(out_dir) / f"{tag}.json"
    if columns is None:
        columns = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(csv_path), str(json_path)


class AugBuffer:
    """Collect online-augmented samples during training and flush to disk on demand."""

    def __init__(
        self,
        out_dir: str,
        prefix: str,
        fmt: str = "npz",
        columns: Optional[list] = None,
        max_hold: int = 10000,
    ):
        self.out_dir = out_dir
        self.prefix = prefix
        self.fmt = fmt
        self.columns = columns
        self.max_hold = int(max_hold)
        self._xs = []
        self._ys = []

    def add(self, xs: np.ndarray, ys: np.ndarray):
        """Append samples to the buffer; auto-flush when max_hold is reached."""
        if xs is None or len(xs) == 0:
            return
        self._xs.append(np.asarray(xs))
        self._ys.append(np.asarray(ys))
        if sum(arr.shape[0] for arr in self._xs) >= self.max_hold:
            self.flush()

    def flush(self):
        """Write buffered samples to disk and clear the buffer."""
        if not self._xs:
            return None
        X = np.concatenate(self._xs, axis=0)
        y = np.concatenate(self._ys, axis=0)
        meta = {"type": "online", "N": int(X.shape[0]), "note": "collected_during_training"}
        if self.fmt.lower() == "csv":
            path, meta_path = save_csv(X, y, self.out_dir, self.prefix, meta, columns=self.columns)
        else:
            path, meta_path = save_npz(X, y, self.out_dir, self.prefix, meta)
        self._xs.clear()
        self._ys.clear()
        return path, meta_path


AUG_METHODS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {}


def register_aug(name: str):
    """Decorator to register an offline augmentation function under a given name."""
    def _wrap(fn):
        AUG_METHODS[name] = fn
        return fn
    return _wrap


@register_aug("smogn_preprocess")
def smogn_preprocess(
    X: np.ndarray,
    y: np.ndarray,
    frac: float = 0.0,
    k: int = 5,
    q: float = 0.2,
    tail: str = "both",
    seed: int = 42,
    y_space: str = "raw",          # "raw" | "log"
    shift: float = 0.0,            # used when y = ln(D + shift)
    renorm_simplex: bool = True,   # whether to renormalize X to row-sum=1 (compositional data)
) -> Tuple[np.ndarray, np.ndarray]:
    """Tail-focused SMOGN-style oversampling with kNN-restricted interpolation in the rare subset."""
    if frac <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    X = X.astype(np.float32, copy=True)
    y = y.reshape(-1).astype(np.float32, copy=True)
    N, D = X.shape

    q_low, q_high = np.quantile(y, q), np.quantile(y, 1 - q)
    if tail == "low":
        idx = np.where(y <= q_low)[0]
    elif tail == "high":
        idx = np.where(y >= q_high)[0]
    else:
        idx = np.where((y <= q_low) | (y >= q_high))[0]
    if idx.size < 2:
        return X, y

    Xm, ym = X[idx], y[idx]

    diff = Xm[:, None, :] - Xm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2, dtype=np.float64))
    np.fill_diagonal(dist, np.inf)

    k_eff = int(min(k, idx.size - 1))
    nn_idx = np.argpartition(dist, kth=k_eff, axis=1)[:, :k_eff]

    n_new = int(frac * N)
    X_new = np.zeros((n_new, D), dtype=np.float32)
    y_new = np.zeros((n_new,), dtype=np.float32)

    for t in range(n_new):
        i = rng.integers(0, idx.size)
        j = nn_idx[i, rng.integers(0, k_eff)]
        lam = rng.random()

        xi, xj = Xm[i], Xm[j]
        yi, yj = ym[i], ym[j]

        xs = lam * xi + (1 - lam) * xj
        if renorm_simplex:
            xs = np.clip(xs, 0.0, None)
            s = xs.sum()
            xs[:] = (xs / s) if s >= 1e-8 else (1.0 / D)

        if y_space == "log":
            Di = math.exp(yi) - shift
            Dj = math.exp(yj) - shift
            Ds = lam * Di + (1 - lam) * Dj
            ys = math.log(max(Ds + shift, 1e-9))
        else:
            ys = lam * yi + (1 - lam) * yj

        X_new[t] = xs
        y_new[t] = np.float32(ys)

    X_aug = np.concatenate([X, X_new], axis=0)
    y_aug = np.concatenate([y, y_new], axis=0)
    return X_aug, y_aug


def smogn_batch_torch(
    x,
    y,
    frac: float = 0.0,
    renorm_simplex: bool = True,
    y_space: str = "raw",
    shift: float = 0.0,
    buffer: Optional[AugBuffer] = None,
):
    """Training-time augmentation; optionally collect synthesized samples into an AugBuffer."""
    assert torch is not None, "PyTorch is required for smogn_batch_torch."
    if frac <= 0:
        return x, y

    B = x.size(0)
    n_new = int(B * frac)
    if n_new <= 0:
        return x, y

    perm1 = torch.randint(0, B, (n_new,), device=x.device)
    perm2 = torch.randint(0, B, (n_new,), device=x.device)
    lam = torch.rand(n_new, 1, device=x.device)

    xs = lam * x[perm1] + (1 - lam) * x[perm2]
    if renorm_simplex:
        xs = torch.clamp(xs, min=0.0)
        xs = xs / (xs.sum(dim=1, keepdim=True) + 1e-8)

    if y.dim() == 1:
        y = y.unsqueeze(1)

    if y_space == "log":
        Di = torch.exp(y[perm1]) - shift
        Dj = torch.exp(y[perm2]) - shift
        Ds = lam * Di + (1 - lam) * Dj
        ys = torch.log(torch.clamp(Ds + shift, min=1e-9))
    else:
        ys = lam * y[perm1] + (1 - lam) * y[perm2]

    if buffer is not None:
        buffer.add(xs.detach().cpu().numpy(), ys.detach().cpu().numpy().reshape(-1))

    x_out = torch.cat([x, xs], dim=0)
    y_out = torch.cat([y, ys], dim=0)
    if y_out.size(-1) == 1:
        y_out = y_out.squeeze(-1)
    return x_out, y_out


def apply_and_save_offline(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    filename_prefix: str,
    fmt: str = "npz",
    columns: Optional[list] = None,
    **kwargs,
) -> Tuple[str, str]:
    """Run a registered offline augmentation method and immediately save the augmented dataset."""
    if method not in AUG_METHODS:
        raise KeyError(f"Unknown augmentation method: {method}. Available: {list(AUG_METHODS.keys())}")

    X_aug, y_aug = AUG_METHODS[method](X, y, **kwargs)
    meta = {
        "method": method,
        "kwargs": kwargs,
        "N_orig": int(X.shape[0]),
        "N_aug": int(X_aug.shape[0]),
        "fmt": fmt,
    }

    if fmt.lower() == "csv":
        return save_csv(X_aug, y_aug, out_dir, filename_prefix, meta, columns=columns)
    return save_npz(X_aug, y_aug, out_dir, filename_prefix, meta)
