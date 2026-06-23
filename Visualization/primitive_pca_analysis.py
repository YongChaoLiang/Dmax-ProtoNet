# primitive_pca_analysis.py
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


ELEMENT_DESCRIPTOR = {
    "Ag": {"atomic_number": 47, "period": 5, "group": 11, "atomic_radius_pm": 144, "electronegativity": 1.93},
    "Al": {"atomic_number": 13, "period": 3, "group": 13, "atomic_radius_pm": 143, "electronegativity": 1.61},
    "Au": {"atomic_number": 79, "period": 6, "group": 11, "atomic_radius_pm": 144, "electronegativity": 2.54},
    "B":  {"atomic_number": 5,  "period": 2, "group": 13, "atomic_radius_pm": 87,  "electronegativity": 2.04},
    "Ba": {"atomic_number": 56, "period": 6, "group": 2,  "atomic_radius_pm": 222, "electronegativity": 0.89},
    "Be": {"atomic_number": 4,  "period": 2, "group": 2,  "atomic_radius_pm": 112, "electronegativity": 1.57},
    "C":  {"atomic_number": 6,  "period": 2, "group": 14, "atomic_radius_pm": 67,  "electronegativity": 2.55},
    "Ca": {"atomic_number": 20, "period": 4, "group": 2,  "atomic_radius_pm": 197, "electronegativity": 1.00},
    "Ce": {"atomic_number": 58, "period": 6, "group": 3,  "atomic_radius_pm": 182, "electronegativity": 1.12},
    "Co": {"atomic_number": 27, "period": 4, "group": 9,  "atomic_radius_pm": 125, "electronegativity": 1.88},
    "Cr": {"atomic_number": 24, "period": 4, "group": 6,  "atomic_radius_pm": 128, "electronegativity": 1.66},
    "Cu": {"atomic_number": 29, "period": 4, "group": 11, "atomic_radius_pm": 128, "electronegativity": 1.90},
    "Dy": {"atomic_number": 66, "period": 6, "group": 3,  "atomic_radius_pm": 178, "electronegativity": 1.22},
    "Er": {"atomic_number": 68, "period": 6, "group": 3,  "atomic_radius_pm": 176, "electronegativity": 1.24},
    "Fe": {"atomic_number": 26, "period": 4, "group": 8,  "atomic_radius_pm": 126, "electronegativity": 1.83},
    "Ga": {"atomic_number": 31, "period": 4, "group": 13, "atomic_radius_pm": 135, "electronegativity": 1.81},
    "Gd": {"atomic_number": 64, "period": 6, "group": 3,  "atomic_radius_pm": 180, "electronegativity": 1.20},
    "Ge": {"atomic_number": 32, "period": 4, "group": 14, "atomic_radius_pm": 125, "electronegativity": 2.01},
    "Hf": {"atomic_number": 72, "period": 6, "group": 4,  "atomic_radius_pm": 159, "electronegativity": 1.30},
    "Ho": {"atomic_number": 67, "period": 6, "group": 3,  "atomic_radius_pm": 177, "electronegativity": 1.23},
    "In": {"atomic_number": 49, "period": 5, "group": 13, "atomic_radius_pm": 167, "electronegativity": 1.78},
    "La": {"atomic_number": 57, "period": 6, "group": 3,  "atomic_radius_pm": 187, "electronegativity": 1.10},
    "Li": {"atomic_number": 3,  "period": 2, "group": 1,  "atomic_radius_pm": 152, "electronegativity": 0.98},
    "Lu": {"atomic_number": 71, "period": 6, "group": 3,  "atomic_radius_pm": 174, "electronegativity": 1.27},
    "Mg": {"atomic_number": 12, "period": 3, "group": 2,  "atomic_radius_pm": 160, "electronegativity": 1.31},
    "Mn": {"atomic_number": 25, "period": 4, "group": 7,  "atomic_radius_pm": 127, "electronegativity": 1.55},
    "Mo": {"atomic_number": 42, "period": 5, "group": 6,  "atomic_radius_pm": 139, "electronegativity": 2.16},
    "Nb": {"atomic_number": 41, "period": 5, "group": 5,  "atomic_radius_pm": 146, "electronegativity": 1.60},
    "Nd": {"atomic_number": 60, "period": 6, "group": 3,  "atomic_radius_pm": 181, "electronegativity": 1.14},
    "Ni": {"atomic_number": 28, "period": 4, "group": 10, "atomic_radius_pm": 124, "electronegativity": 1.91},
    "P":  {"atomic_number": 15, "period": 3, "group": 15, "atomic_radius_pm": 98,  "electronegativity": 2.19},
    "Pb": {"atomic_number": 82, "period": 6, "group": 14, "atomic_radius_pm": 175, "electronegativity": 2.33},
    "Pd": {"atomic_number": 46, "period": 5, "group": 10, "atomic_radius_pm": 137, "electronegativity": 2.20},
    "Pr": {"atomic_number": 59, "period": 6, "group": 3,  "atomic_radius_pm": 182, "electronegativity": 1.13},
    "Pt": {"atomic_number": 78, "period": 6, "group": 10, "atomic_radius_pm": 139, "electronegativity": 2.28},
    "Sc": {"atomic_number": 21, "period": 4, "group": 3,  "atomic_radius_pm": 162, "electronegativity": 1.36},
    "Si": {"atomic_number": 14, "period": 3, "group": 14, "atomic_radius_pm": 111, "electronegativity": 1.90},
    "Sm": {"atomic_number": 62, "period": 6, "group": 3,  "atomic_radius_pm": 180, "electronegativity": 1.17},
    "Sn": {"atomic_number": 50, "period": 5, "group": 14, "atomic_radius_pm": 158, "electronegativity": 1.96},
    "Ta": {"atomic_number": 73, "period": 6, "group": 5,  "atomic_radius_pm": 146, "electronegativity": 1.50},
    "Tb": {"atomic_number": 65, "period": 6, "group": 3,  "atomic_radius_pm": 178, "electronegativity": 1.10},
    "Ti": {"atomic_number": 22, "period": 4, "group": 4,  "atomic_radius_pm": 147, "electronegativity": 1.54},
    "Tm": {"atomic_number": 69, "period": 6, "group": 3,  "atomic_radius_pm": 176, "electronegativity": 1.25},
    "V":  {"atomic_number": 23, "period": 4, "group": 5,  "atomic_radius_pm": 134, "electronegativity": 1.63},
    "W":  {"atomic_number": 74, "period": 6, "group": 6,  "atomic_radius_pm": 139, "electronegativity": 2.36},
    "Y":  {"atomic_number": 39, "period": 5, "group": 3,  "atomic_radius_pm": 180, "electronegativity": 1.22},
    "Zn": {"atomic_number": 30, "period": 4, "group": 12, "atomic_radius_pm": 134, "electronegativity": 1.65},
    "Zr": {"atomic_number": 40, "period": 5, "group": 4,  "atomic_radius_pm": 160, "electronegativity": 1.33},
}


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        feat_cols = ckpt.get("feat_cols", None)
        cfg = ckpt.get("cfg", {})
    else:
        state = ckpt
        feat_cols = None
        cfg = {}

    return state, feat_cols, cfg


def find_primitive_key(state_dict, user_key=None):
    if user_key is not None:
        if user_key not in state_dict:
            raise KeyError(f"User-specified P key '{user_key}' not found in checkpoint.")
        return user_key

    preferred_keys = [
        "proto.P",
        "module.proto.P",
    ]

    for key in preferred_keys:
        if key in state_dict:
            return key

    candidates = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        if value.ndim == 2 and (key.endswith(".P") or key.endswith("P") or "proto" in key.lower()):
            candidates.append((key, tuple(value.shape)))

    if len(candidates) == 1:
        return candidates[0][0]

    print("\nCandidate 2D tensor keys possibly related to primitive P:")
    for key, shape in candidates:
        print(f"  {key}: {shape}")

    raise KeyError(
        "Could not automatically identify primitive matrix P. "
        "Please specify it with --p_key, e.g., --p_key proto.P"
    )


def load_descriptor_table(feat_cols, descriptor_csv=None):
    if descriptor_csv is not None:
        desc = pd.read_csv(descriptor_csv)
        if "Element" not in desc.columns:
            raise ValueError("descriptor_csv must contain a column named 'Element'.")
        return desc

    rows = []
    for elem in feat_cols:
        if elem in ELEMENT_DESCRIPTOR:
            row = {"Element": elem}
            row.update(ELEMENT_DESCRIPTOR[elem])
            rows.append(row)
        else:
            rows.append({
                "Element": elem,
                "atomic_number": np.nan,
                "period": np.nan,
                "group": np.nan,
                "atomic_radius_pm": np.nan,
                "electronegativity": np.nan,
            })

    return pd.DataFrame(rows)


def parse_key_elements(s):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def add_element_labels(
    ax,
    df,
    fontsize=7.5,
    label_mode="all",
    key_elements=None,
):
    if key_elements is None:
        key_elements = []

    if label_mode == "none":
        return

    texts = []

    x = df["PC1"].values
    y = df["PC2"].values

    x_range = np.nanmax(x) - np.nanmin(x)
    y_range = np.nanmax(y) - np.nanmin(y)

    dx = 0.010 * x_range if x_range > 0 else 0.03
    dy = 0.010 * y_range if y_range > 0 else 0.03

    for _, row in df.iterrows():
        elem = row["Element"]

        if label_mode == "key" and elem not in key_elements:
            continue

        text = ax.text(
            row["PC1"] + dx,
            row["PC2"] + dy,
            elem,
            fontsize=fontsize,
            ha="left",
            va="bottom",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.10",
                fc="white",
                ec="none",
                alpha=0.68
            )
        )
        texts.append(text)

    if HAS_ADJUST_TEXT and len(texts) > 0:
        try:
            adjust_text(
                texts,
                x=x,
                y=y,
                ax=ax,
                expand_text=(1.10, 1.25),
                expand_points=(1.25, 1.45),
                force_text=(0.20, 0.35),
                force_points=(0.12, 0.25),
                lim=500,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.35,
                    alpha=0.60
                )
            )
        except TypeError:
            adjust_text(
                texts,
                x=x,
                y=y,
                ax=ax,
                expand=(1.10, 1.25),
                force_text=(0.20, 0.35),
                force_static=(0.12, 0.25),
                lim=500,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.35,
                    alpha=0.60
                )
            )


def plot_pca_2x2(
    df,
    descriptor_cols,
    explained,
    out_path,
    label_mode="all",
    key_elements=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    axes = axes.ravel()

    xlab = f"PC1 ({explained[0] * 100:.2f}%)"
    ylab = f"PC2 ({explained[1] * 100:.2f}%)"

    title_map = {
        "atomic_radius_pm": "Atomic radius / pm",
        "electronegativity": "Electronegativity",
        "period": "Period",
        "group": "Group",
        "atomic_number": "Atomic number",
    }

    for idx, ax in enumerate(axes):
        if idx >= len(descriptor_cols):
            ax.axis("off")
            continue

        desc = descriptor_cols[idx]
        valid = np.isfinite(df[desc].values)

        if valid.any():
            sc = ax.scatter(
                df.loc[valid, "PC1"],
                df.loc[valid, "PC2"],
                c=df.loc[valid, desc],
                s=62,
                cmap="viridis",
                edgecolor="black",
                linewidth=0.45,
                alpha=0.90,
                zorder=3
            )

            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.scatter(
                df["PC1"],
                df["PC2"],
                s=62,
                c="lightgray",
                edgecolor="black",
                linewidth=0.45,
                alpha=0.75,
                zorder=3
            )

        missing = ~valid
        if missing.any():
            ax.scatter(
                df.loc[missing, "PC1"],
                df.loc[missing, "PC2"],
                s=62,
                c="lightgray",
                edgecolor="black",
                linewidth=0.45,
                alpha=0.75,
                label="Missing descriptor",
                zorder=3
            )

        add_element_labels(
            ax,
            df,
            fontsize=7.2 if label_mode == "all" else 8.2,
            label_mode=label_mode,
            key_elements=key_elements,
        )

        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title_map.get(desc, desc), fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="../saved_models/20260110_version2/seed=0/argument/mixup/r2=0_8941.pt",
        help="Path to trained DPN checkpoint .pt"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./figures",
        help="Output directory. Only primitive_pca_2x2.png will be saved."
    )
    parser.add_argument(
        "--p_key",
        type=str,
        default=None,
        help="State dict key of primitive matrix P, e.g., proto.P"
    )
    parser.add_argument(
        "--descriptor_csv",
        type=str,
        default=None,
        help="Optional descriptor CSV with columns: Element, atomic_radius_pm, electronegativity, period, group, etc."
    )
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="If set, PCA will be performed on raw P instead of standardized P."
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="all",
        choices=["all", "key", "none"],
        help="Element label mode: all, key, or none."
    )
    parser.add_argument(
        "--key_elements",
        type=str,
        default="Al,Cu,Zr,Ca,Mg,Ni,Ti,Nb,B,P,Pd,Pt,La,Lu,Dy",
        help="Comma-separated key elements to label when --label_mode key is used."
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    state, feat_cols, cfg = load_checkpoint(args.ckpt)

    p_key = find_primitive_key(state, args.p_key)
    P = state[p_key].detach().cpu().numpy()

    if feat_cols is None:
        raise ValueError(
            "feat_cols not found in checkpoint. "
            "Your checkpoint should contain {'model': state_dict, 'feat_cols': feat_cols, 'cfg': cfg}."
        )

    if len(feat_cols) != P.shape[0]:
        raise ValueError(
            f"Number of feat_cols ({len(feat_cols)}) does not match P rows ({P.shape[0]}). "
            "Please check checkpoint consistency."
        )

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Primitive key: {p_key}")
    print(f"P shape: {P.shape}")
    print(f"Number of elements: {len(feat_cols)}")

    if not HAS_ADJUST_TEXT and args.label_mode != "none":
        print("[Warning] adjustText is not installed. Labels may still overlap.")
        print("          Install it with: pip install adjustText")

    if args.no_standardize:
        P_for_pca = P.copy()
        pca_note = "raw_P"
    else:
        P_for_pca = StandardScaler().fit_transform(P)
        pca_note = "standardized_P"

    pca = PCA(n_components=2)
    Z = pca.fit_transform(P_for_pca)
    explained = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "Element": feat_cols,
        "PC1": Z[:, 0],
        "PC2": Z[:, 1],
    })

    desc_df = load_descriptor_table(feat_cols, args.descriptor_csv)
    merged = pca_df.merge(desc_df, on="Element", how="left")

    plot_cols = [
        col for col in ["atomic_radius_pm", "electronegativity", "period", "group"]
        if col in merged.columns
    ]

    if len(plot_cols) == 0:
        raise ValueError(
            "No descriptor columns were found for plotting. "
            "Please check ELEMENT_DESCRIPTOR or descriptor_csv."
        )

    key_elements = parse_key_elements(args.key_elements)

    fig_path = os.path.join(args.out_dir, "primitive_pca_2x2.png")

    plot_pca_2x2(
        merged,
        plot_cols,
        explained,
        fig_path,
        label_mode=args.label_mode,
        key_elements=key_elements,
    )

    print("\n===== PCA Summary =====")
    print(f"PCA input: {pca_note}")
    print(f"PC1 explained variance: {explained[0] * 100:.2f}%")
    print(f"PC2 explained variance: {explained[1] * 100:.2f}%")
    print(f"Total explained variance: {explained[:2].sum() * 100:.2f}%")

    descriptor_cols = [
        col for col in ["atomic_number", "period", "group", "atomic_radius_pm", "electronegativity"]
        if col in merged.columns
    ]

    if len(descriptor_cols) > 0:
        missing = merged[merged[descriptor_cols].isna().any(axis=1)]["Element"].tolist()
        if missing:
            print("\n[Warning] Missing descriptor values for elements:")
            print(missing)

    print("\nSaved file:")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
