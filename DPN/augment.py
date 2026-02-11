import numpy as np
import torch


def _renorm_simplex(x: torch.Tensor) -> torch.Tensor:
    """Project to the simplex: clamp to non-negative and renormalize each row to sum to 1."""
    x = torch.clamp(x, min=0.0)
    s = x.sum(dim=1, keepdim=True)
    mask = (s <= 1e-12)
    if mask.any():
        x[mask] = 1.0 / x.size(1)
        s = x.sum(dim=1, keepdim=True)
    return x / (s + 1e-8)


def mixup_simplex(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Online Mixup for simplex inputs (non-negative, row-sum=1)."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[perm], y[perm]
    xm = lam * x + (1 - lam) * x2
    xm = torch.clamp(xm, min=0.0)
    xm = xm / (xm.sum(dim=1, keepdim=True) + 1e-8)
    ym = lam * y + (1 - lam) * y2
    return xm, ym


def gaussian_noise_simplex_masked(
    x: torch.Tensor,
    std: float = 0.005,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Add Gaussian noise to a composition vector and project back to the simplex.
    Noise is applied only on "active" dimensions (x > eps) to avoid introducing spurious non-zero elements.
    """
    if std <= 0:
        return x

    mask = (x > eps).to(x.dtype)  # [B, N]
    noise = torch.randn_like(x) * std
    xn = x + noise * mask

    xn = torch.clamp(xn, min=0.0)
    denom = xn.sum(dim=1, keepdim=True)

    safe = denom > 1e-12
    xn = torch.where(safe, xn / (denom + 1e-8), x)
    return xn


def smogn_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    frac: float = 0.3,
    tail: str = "high",      # "high" | "low" | "both"
    q: float = 0.9,
    k: int = 5,
    p_gn: float = 0.3,
    gn_std: float = 0.005,
    eps: float = 1e-6,
):
    """
    Online approximate SMOGN for regression with simplex composition inputs.

    Inputs:
      x: [B, N] composition vectors (assumed non-negative and row-sum=1)
      y: [B, 1] regression targets (e.g., Dmax)
      frac: fraction of new samples to generate (relative to batch size B)
      tail: which tail to treat as rare ("high", "low", or "both")
      q: quantile threshold (e.g., 0.9 means top 10% are rare for "high")
      k: number of neighbors (within the current batch)
      p_gn: probability of using the Gaussian-noise branch when synthesizing
      gn_std: Gaussian noise std for composition perturbation
      eps: threshold for "active" composition dimensions

    Outputs:
      x_out: [B + n_new, N]
      y_out: [B + n_new, 1]

    Note:
      This is a batch-local approximation of SMOGN ideas (rare-region oversampling + interpolation/GN).
      A full SMOGN implementation typically uses dataset-level rarity modeling and resampling ratios offline.
    """
    if frac <= 0:
        return x, y

    B, N = x.shape
    n_new = int(B * frac)
    if n_new <= 0:
        return x, y

    # 1) Identify rare samples in the batch using quantiles of y
    y1 = y.view(-1)
    tail = tail.lower().strip()

    q_hi = torch.quantile(y1, q)
    q_lo = torch.quantile(y1, 1.0 - q)

    if tail == "high":
        rare_mask = (y1 >= q_hi)
    elif tail == "low":
        rare_mask = (y1 <= q_lo)
    elif tail == "both":
        rare_mask = (y1 >= q_hi) | (y1 <= q_lo)
    else:
        raise ValueError("tail must be one of: 'high', 'low', 'both'")

    rare_idx = torch.where(rare_mask)[0]
    n_rare = rare_idx.numel()
    if n_rare < 2:
        return x, y

    # 2) Find neighbors among rare samples (Euclidean distance in x-space)
    xr = x[rare_idx]  # [R, N]
    yr = y[rare_idx]  # [R, 1]

    dist = torch.cdist(xr, xr, p=2)
    dist.fill_diagonal_(float("inf"))

    k_eff = min(k, n_rare - 1)
    nn_idx = dist.topk(k_eff, largest=False).indices  # [R, k_eff]

    # 3) Synthesize samples: interpolation (SMOTER-style) or Gaussian-noise branch
    seed_pos = torch.randint(0, n_rare, (n_new,), device=x.device)
    seed_x = xr[seed_pos]
    seed_y = yr[seed_pos]

    nn_choices = nn_idx[seed_pos]
    pick = torch.randint(0, k_eff, (n_new,), device=x.device)
    nb_pos = nn_choices[torch.arange(n_new, device=x.device), pick]
    nb_x = xr[nb_pos]
    nb_y = yr[nb_pos]

    use_gn = (torch.rand(n_new, device=x.device) < p_gn).view(-1, 1)

    lam = torch.rand(n_new, 1, device=x.device)
    xs_interp = lam * seed_x + (1.0 - lam) * nb_x
    ys_interp = lam * seed_y + (1.0 - lam) * nb_y

    mask = (seed_x > eps).to(seed_x.dtype)
    noise = torch.randn_like(seed_x) * gn_std
    xs_gn = seed_x + noise * mask
    ys_gn = seed_y

    xs = torch.where(use_gn, xs_gn, xs_interp)
    ys = torch.where(use_gn, ys_gn, ys_interp)

    # 4) Project synthesized x back to simplex
    xs = torch.clamp(xs, min=0.0)
    denom = xs.sum(dim=1, keepdim=True)
    safe = denom > 1e-12
    xs = torch.where(safe, xs / (denom + 1e-8), seed_x)

    # 5) Concatenate outputs
    x_out = torch.cat([x, xs], dim=0)
    y_out = torch.cat([y, ys], dim=0)
    return x_out, y_out


def apply_augment(
    x,
    y,
    use_mixup: bool = False,
    mixup_alpha: float = 0.4,
    use_gn: bool = False,
    gn_std: float = 0.01,
    use_smote: bool = False,
    smote_frac: float = 0.0,
):
    """Apply online augmentations in a configurable way (Mixup / Gaussian noise / SMOGN)."""
    if use_mixup:
        x, y = mixup_simplex(x, y, mixup_alpha)
    if use_gn:
        x = gaussian_noise_simplex_masked(x, gn_std)
    if use_smote and smote_frac > 0:
        x, y = smogn_batch(x, y, smote_frac)
    return x, y


def smogn_preprocess(
    X: np.ndarray,
    y: np.ndarray,
    frac: float = 0.0,
    k: int = 5,
    q: float = 0.2,
    tail: str = "both",
    seed: int = 42,
):
    """
    Offline preprocessing variant: enlarge the dataset by convex combinations among tail samples.

    X: (N, D) composition vectors (non-negative, row-sum≈1)
    y: (N,) or (N, 1)
    """
    if frac <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    y = y.reshape(-1).astype(np.float32)

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

    Xm = X[idx]
    ym = y[idx]

    diff = Xm[:, None, :] - Xm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)

    k_eff = min(k, idx.size - 1)
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
        ys = lam * yi + (1 - lam) * yj

        xs = np.clip(xs, 0.0, None)
        s = xs.sum()
        if s < 1e-8:
            xs[:] = 1.0 / D
        else:
            xs /= s

        X_new[t] = xs
        y_new[t] = ys

    X_aug = np.concatenate([X, X_new], axis=0)
    y_aug = np.concatenate([y, y_new], axis=0)
    return X_aug, y_aug
