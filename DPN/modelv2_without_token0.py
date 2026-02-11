# models.py
import torch
import torch.nn as nn


# Period/group lookup for selected elements: (period, group)
PERIOD = {
    "H": (1, 1), "He": (1, 18), "Li": (2, 1), "Be": (2, 2), "B": (2, 13), "C": (2, 14), "N": (2, 15), "O": (2, 16), "F": (2, 17), "Ne": (2, 18),
    "Na": (3, 1), "Mg": (3, 2), "Al": (3, 13), "Si": (3, 14), "P": (3, 15), "S": (3, 16), "Cl": (3, 17), "Ar": (3, 18),
    "K": (4, 1), "Ca": (4, 2), "Sc": (4, 3), "Ti": (4, 4), "V": (4, 5), "Cr": (4, 6), "Mn": (4, 7), "Fe": (4, 8), "Co": (4, 9), "Ni": (4, 10),
    "Cu": (4, 11), "Zn": (4, 12), "Ga": (4, 13), "Ge": (4, 14), "As": (4, 15), "Se": (4, 16), "Br": (4, 17), "Kr": (4, 18),
    "Rb": (5, 1), "Sr": (5, 2), "Y": (5, 3), "Zr": (5, 4), "Nb": (5, 5), "Mo": (5, 6), "Tc": (5, 7), "Ru": (5, 8), "Rh": (5, 9), "Pd": (5, 10),
    "Ag": (5, 11), "Cd": (5, 12), "In": (5, 13), "Sn": (5, 14), "Sb": (5, 15), "Te": (5, 16), "I": (5, 17), "Xe": (5, 18),
    "Cs": (6, 1), "Ba": (6, 2), "La": (6, 3), "Ce": (6, 3), "Pr": (6, 3), "Nd": (6, 3),
}


def build_period_adj(feat_names):
    """Build adjacency lists by connecting elements that share the same period or the same group."""
    def name2_pg(name):
        return PERIOD.get(name, None)

    N = len(feat_names)
    buckets = {}
    for i, name in enumerate(feat_names):
        pg = name2_pg(name)
        if pg is None:
            continue
        p, g = pg
        buckets.setdefault(("row", p), []).append(i)
        buckets.setdefault(("col", g), []).append(i)

    adj = [[] for _ in range(N)]
    for _, idxs in buckets.items():
        for u in idxs:
            for v in idxs:
                if u != v and v not in adj[u]:
                    adj[u].append(v)
    return adj


class MatProtoFeaturizer(nn.Module):
    """Learnable property-primitive matrix + derived pairwise bias and Laplacian regularization."""

    def __init__(self, N, K=8, lambda_lap=1e-3):
        super().__init__()
        self.N = N
        self.K = K
        self.lambda_lap = lambda_lap
        self.P = nn.Parameter(torch.randn(N, K) * 0.1)
        self.bias_w = nn.Parameter(torch.randn(K) * 0.1)
        self.bias_b = nn.Parameter(torch.zeros(1))
        self.register_buffer("_adj_mask", torch.zeros(N, N, dtype=torch.bool))

    def set_adjacency(self, adj_lists):
        """Set the adjacency mask used by Laplacian regularization (from adjacency lists)."""
        N = self.N
        mask = torch.zeros(N, N, dtype=torch.bool)
        for i, nbrs in enumerate(adj_lists):
            if len(nbrs) > 0:
                mask[i, torch.tensor(nbrs, dtype=torch.long)] = True
        self._adj_mask = mask

    def global_stats(self, frac):
        """Compute composition-weighted mean/variance of primitives, plus entropy and sparsity features."""
        P = self.P.to(frac.device)  # [N, K]
        w = frac                    # [B, N]

        mean = w @ P  # [B, K]
        diff = P.unsqueeze(0) - mean.unsqueeze(1)          # [B, N, K]
        var = torch.sum(w.unsqueeze(-1) * diff.pow(2), 1)  # [B, K]

        eps = 1e-12
        H = -torch.sum(w * torch.log(w + eps), 1, keepdim=True)          # [B, 1]
        nnz = (w > 1e-6).float().sum(1, keepdim=True) / w.size(1)        # [B, 1]

        return torch.cat([mean, var, H, nnz], 1)  # [B, 2K+2]

    def pair_bias(self):
        """Compute pairwise bias matrix from primitive-space squared distances."""
        P = self.P
        dij2 = (P.unsqueeze(1) - P.unsqueeze(0)).pow(2)  # [N, N, K]
        B_ij = dij2 @ self.bias_w                        # [N, N]
        return B_ij + self.bias_b

    def laplacian_reg(self):
        """Laplacian smoothness regularizer over the primitive matrix under the provided adjacency mask."""
        if self._adj_mask.numel() == 0:
            return self.lambda_lap * self.P.new_tensor(0.0)
        Pi = self.P.unsqueeze(1)
        Pj = self.P.unsqueeze(0)
        diff2 = (Pi - Pj).pow(2).sum(-1)
        masked = diff2[self._adj_mask]
        if masked.numel() == 0:
            return self.lambda_lap * self.P.new_tensor(0.0)
        return self.lambda_lap * masked.mean()


class MatBiasSelfAttention(nn.Module):
    """Multi-head self-attention with an additive pairwise bias term (broadcasted across heads)."""

    def __init__(self, dim: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.nh = n_heads
        self.dh = dim // n_heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, bias_ij: torch.Tensor, key_mask: torch.Tensor = None):
        """
        Args:
            x:        [B, N, D]
            bias_ij:  [N, N]  (pairwise bias matrix)
            key_mask: [B, N]  True=valid key; False=masked out
        """
        B, N, D = x.shape
        H = self.nh
        Dh = self.dh

        q = self.Wq(x).view(B, N, H, Dh).transpose(1, 2)  # [B, H, N, Dh]
        k = self.Wk(x).view(B, N, H, Dh).transpose(1, 2)  # [B, H, N, Dh]
        v = self.Wv(x).view(B, N, H, Dh).transpose(1, 2)  # [B, H, N, Dh]

        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)  # [B, H, N, N]
        scores = scores + self.alpha * bias_ij.view(1, 1, N, N)

        if key_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~key_mask.view(B, 1, 1, N), mask_value)

        attn = scores.softmax(dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(self.drop(y))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with optional masked pooling over tokens."""

    def __init__(self, dim: int, r: int = 4):
        super().__init__()
        hidden = max(dim // r, 8)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

    def forward(self, x, mask: torch.Tensor = None):
        """
        Args:
            x:    [B, N, D]
            mask: [B, N] True indicates valid tokens (optional)
        """
        if mask is None:
            z = x.mean(dim=1)  # [B, D]
        else:
            m = mask.to(x.dtype).unsqueeze(-1)            # [B, N, 1]
            denom = m.sum(dim=1).clamp_min(1e-8)          # [B, 1]
            z = (x * m).sum(dim=1) / denom                # [B, D]

        s = self.fc2(self.act(self.fc1(z)))               # [B, D]
        s = self.sig(s).unsqueeze(1)                      # [B, 1, D]
        return x * s


class MatPhys_GA(nn.Module):
    """Composition-only regression model with primitive-induced pairwise attention bias."""

    def __init__(
        self,
        num_tokens,
        feat_names,
        model_dim=96,
        n_heads=2,
        dropout=0.2,
        K_proto=8,
        lambda_lap=1e-3,
        positive_output: bool = True,
    ):
        super().__init__()
        self.N = num_tokens
        self.positive_output = positive_output

        self.elem = nn.Embedding(self.N, model_dim)
        self.frac = nn.Linear(1, model_dim)

        self.proto = MatProtoFeaturizer(N=self.N, K=K_proto, lambda_lap=lambda_lap)
        adj = build_period_adj(feat_names)
        self.proto.set_adjacency(adj)

        self.norm = nn.LayerNorm(model_dim)
        self.attn = MatBiasSelfAttention(model_dim, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
        )

        self.se = SEBlock(model_dim)

        head_layers = [
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        ]
        if positive_output:
            head_layers.append(nn.Softplus())
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        """
        Args:
            x: [B, N] composition fractions (already normalized to the simplex)
        """
        B, Nin = x.shape
        if Nin != self.N:
            raise ValueError(f"Input feature dim mismatch: got {Nin}, expected {self.N}")

        idx = torch.arange(self.N, device=x.device).long().unsqueeze(0).expand(B, -1)  # [B, N]
        tok = self.elem(idx) + self.frac(x.unsqueeze(-1))                               # [B, N, D]

        bias_ij = self.proto.pair_bias().to(x.device)                                   # [N, N]

        eps = 1e-6
        key_mask = (x > eps)                                                            # [B, N]

        tok = tok + self.attn(self.norm(tok), bias_ij, key_mask=key_mask)
        tok = tok + self.ffn(tok)
        tok = self.se(tok, mask=key_mask)

        m = key_mask.to(tok.dtype).unsqueeze(-1)                                        # [B, N, 1]
        denom = m.sum(dim=1).clamp_min(1e-8)                                            # [B, 1]
        pooled_masked = (tok * m).sum(dim=1) / denom                                    # [B, D]

        safe = (key_mask.sum(dim=1, keepdim=True) > 0)                                  # [B, 1]
        pooled = torch.where(safe, pooled_masked, tok.mean(dim=1))

        return self.head(pooled)

    def reg_loss(self):
        """Return the Laplacian regularization term for the property-primitive matrix."""
        return self.proto.laplacian_reg()
