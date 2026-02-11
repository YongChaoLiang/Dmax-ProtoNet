import numpy as np


def renorm_simplex_np(X: np.ndarray, eps_sum: float = 1e-12) -> np.ndarray:
    """Project composition vectors back to the simplex: non-negative and row-sum=1."""
    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)
    bad = (s <= eps_sum).reshape(-1)
    if np.any(bad):
        X[bad] = 1.0 / X.shape[1]
        s = X.sum(axis=1, keepdims=True)
    return X / (s + 1e-8)


def mixup_simplex_offline(
    X: np.ndarray,
    y: np.ndarray,
    frac: float = 0.0,
    alpha: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Offline Mixup: generate n_new = int(frac * n) synthetic samples over the full dataset."""
    if frac <= 0 or alpha <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_new = int(n * frac)
    if n_new <= 0:
        return X, y

    idx1 = rng.integers(0, n, size=n_new)
    idx2 = rng.integers(0, n, size=n_new)

    lam = rng.beta(alpha, alpha, size=(n_new, 1)).astype(np.float32)

    Xs = lam * X[idx1] + (1.0 - lam) * X[idx2]
    Xs = renorm_simplex_np(Xs)

    ys = (lam[:, 0] * y[idx1] + (1.0 - lam[:, 0]) * y[idx2]).astype(np.float32)

    X_out = np.concatenate([X, Xs], axis=0)
    y_out = np.concatenate([y, ys], axis=0)
    return X_out, y_out


def gaussian_noise_simplex_masked_offline(
    X: np.ndarray,
    frac: float = 0.0,
    std: float = 0.005,
    eps: float = 1e-6,
    seed: int = 42,
) -> np.ndarray:
    """
    Offline Gaussian noise: sample n_new rows, perturb only active dims (x > eps),
    and add them as new samples (labels should be duplicated outside this function).
    """
    if frac <= 0 or std <= 0:
        return X

    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_new = int(n * frac)
    if n_new <= 0:
        return X

    idx = rng.integers(0, n, size=n_new)
    X_seed = X[idx].copy()

    mask = (X_seed > eps).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=std, size=X_seed.shape).astype(np.float32)

    X_new = X_seed + noise * mask
    X_new = renorm_simplex_np(X_new)

    return np.concatenate([X, X_new], axis=0)


def smogn_preprocess_offline(
    X: np.ndarray,
    y: np.ndarray,
    frac: float = 0.3,
    q: float = 0.9,
    tail: str = "high",   # "high" | "low" | "both"
    k: int = 5,
    p_gn: float = 0.3,
    gn_std: float = 0.005,
    eps: float = 1e-6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Offline (dataset-level) approximate SMOGN:
      - define rare regions by global y quantiles
      - oversample around rare samples only
      - synthesize by interpolation (SMOTER-style) or Gaussian-noise perturbation
      - project synthesized compositions back to the simplex

    Inputs:
      X: [n, d] composition (preferably already normalized to simplex)
      y: [n] or [n,1] regression targets

    Outputs:
      X_aug: [n+n_new, d]
      y_aug: [n+n_new]
    """
    if frac <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    y1 = y.reshape(-1).astype(np.float32)

    n, d = X.shape
    n_new = int(n * frac)
    if n_new <= 0:
        return X, y

    tail = tail.lower().strip()
    q_hi = np.quantile(y1, q)
    q_lo = np.quantile(y1, 1.0 - q)

    if tail == "high":
        rare_mask = (y1 >= q_hi)
    elif tail == "low":
        rare_mask = (y1 <= q_lo)
    elif tail == "both":
        rare_mask = (y1 >= q_hi) | (y1 <= q_lo)
    else:
        raise ValueError("tail must be one of: 'high', 'low', 'both'")

    rare_idx = np.where(rare_mask)[0]
    R = rare_idx.size
    if R < 2:
        return X, y

    Xr = X[rare_idx].astype(np.float32)
    yr = y1[rare_idx].astype(np.float32)

    # Batch-local kNN within the rare subset (Euclidean distance)
    diff = Xr[:, None, :] - Xr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
    np.fill_diagonal(dist, np.inf)

    k_eff = min(k, R - 1)
    nn_idx = np.argpartition(dist, kth=k_eff, axis=1)[:, :k_eff]  # [R, k_eff]

    seed_pos = rng.integers(0, R, size=n_new)
    seed_x = Xr[seed_pos]
    seed_y = yr[seed_pos]

    pick = rng.integers(0, k_eff, size=n_new)
    nb_pos = nn_idx[seed_pos, pick]
    nb_x = Xr[nb_pos]
    nb_y = yr[nb_pos]

    use_gn = rng.random(size=n_new) < p_gn

    lam = rng.random(size=(n_new, 1)).astype(np.float32)
    xs_interp = lam * seed_x + (1.0 - lam) * nb_x
    ys_interp = (lam[:, 0] * seed_y + (1.0 - lam[:, 0]) * nb_y).astype(np.float32)

    mask = (seed_x > eps).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=gn_std, size=seed_x.shape).astype(np.float32)
    xs_gn = seed_x + noise * mask
    ys_gn = seed_y.astype(np.float32)

    xs = xs_interp.copy()
    ys = ys_interp.copy()
    xs[use_gn] = xs_gn[use_gn]
    ys[use_gn] = ys_gn[use_gn]

    xs = renorm_simplex_np(xs)

    X_out = np.concatenate([X, xs], axis=0)
    y_out = np.concatenate([y1, ys], axis=0)
    return X_out, y_out


def augment_offline_dataset(
    X: np.ndarray,
    y: np.ndarray,
    use_mixup: bool = False,
    mixup_frac: float = 0.0,
    mixup_alpha: float = 0.1,
    use_gn: bool = False,
    gn_frac: float = 0.0,
    gn_std: float = 0.005,
    use_smogn: bool = False,
    smogn_frac: float = 0.3,
    smogn_q: float = 0.9,
    smogn_tail: str = "high",
    smogn_k: int = 5,
    smogn_p_gn: float = 0.3,
    smogn_gn_std: float = 0.005,
    eps: float = 1e-6,
    seed: int = 42,
):
    """
    Offline augmentation controller: compose SMOGN / Gaussian noise / Mixup and return an augmented dataset.
    Recommended order:
      1) SMOGN (target rare regions)
      2) Gaussian noise (local perturbation)
      3) Mixup (global smoothing regularization)
    """
    X = renorm_simplex_np(X.astype(np.float32))
    y = y.reshape(-1).astype(np.float32)

    cur_seed = seed

    if use_smogn and smogn_frac > 0:
        X, y = smogn_preprocess_offline(
            X, y,
            frac=smogn_frac,
            q=smogn_q,
            tail=smogn_tail,
            k=smogn_k,
            p_gn=smogn_p_gn,
            gn_std=smogn_gn_std,
            eps=eps,
            seed=cur_seed,
        )
        cur_seed += 1

    if use_gn and gn_frac > 0:
        X = gaussian_noise_simplex_masked_offline(
            X,
            frac=gn_frac,
            std=gn_std,
            eps=eps,
            seed=cur_seed,
        )

        # Duplicate labels for the newly added noisy samples (y is kept from the sampled seed rows)
        n0 = y.shape[0]
        n_new = int(n0 * gn_frac)
        if n_new > 0:
            rng = np.random.default_rng(cur_seed)
            idx = rng.integers(0, n0, size=n_new)
            y = np.concatenate([y, y[idx]], axis=0)

        cur_seed += 1

    if use_mixup and mixup_frac > 0 and mixup_alpha > 0:
        X, y = mixup_simplex_offline(
            X, y,
            frac=mixup_frac,
            alpha=mixup_alpha,
            seed=cur_seed,
        )
        cur_seed += 1

    return X, y
