#!/usr/bin/env python3
"""
radial_gradient_analysis.py

Multi-rosette radial gradient analysis for MALDI mass spectrometry imaging data.

This script quantifies radial metabolite/lipid gradients in neural rosette structures by:
  - Computing radial distance from lumen for each cell
  - Binning cells into radial shells (fixed quantile or dynamic)
  - Calculating fold-change profiles relative to lumen
  - Statistical testing via Spearman correlation and permutation analysis

INPUT DATA REQUIREMENTS:
------------------------
1. Merged rosette file (--input):
   Required columns:
     - x, y: Spatial coordinates (pixels or microns)
     - rosette_id: Unique identifier for each rosette
     - cell_type: Must include "Lumen" cells (or use preprocessing script)
   Optional columns:
     - Any "mz *" columns (e.g., "mz 885.724") for metabolite intensities

   Compatible with any cell-level spatial data where:
     - Cells are grouped into discrete structures (rosettes, organoids, colonies)
     - Each structure has an identifiable center/lumen region
     - Metabolite/marker intensities are measured per cell

2. Master metrics file (--master):
   Required columns:
     - X, Y (or x, y): Spatial coordinates matching the input file
     - "mz *" columns: Full panel of m/z features for permutation testing

   This file provides the complete feature pool. Can be:
     - MALDI imaging export with all detected m/z peaks
     - Filtered feature list (e.g., no isotopes, specific m/z range)

Usage example (with included example data):
  python radial_gradient_analysis.py \\
    --input  "example_data/merged_rosette_data_v1.csv" \\
    --master "example_data/no_isotopes_400to1100_daTol0p01.csv" \\
    --outdir "results/" \\
    --good_mz "mz 917.739, mz 885.724, mz 888.74, mz 911.724" \\
    --bad_mz "" \\
    --n_perm 200 \\
    --normalization log2 \\
    --n_bins 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA


LUMEN_LABEL = "Lumen"


# =========================================================================
# Publication style
# =========================================================================
def set_publication_style():
    """Configure matplotlib for publication-quality figures (Wiley journals)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


PUBLICATION_DPI = 300


# =========================================================================
# Helpers
# =========================================================================
def parse_list_flags(raw: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped, non-empty strings."""
    return [p.strip() for p in raw.split(",") if p.strip()]


def get_mz_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns whose name starts with 'mz '."""
    return [c for c in df.columns if str(c).startswith("mz ")]


def safe_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def normalized_bin_count(
    n_cells: int, max_cells: int, max_bins: int = 10, min_bins: int = 2
) -> int:
    """
    Dynamic bin count normalised by the largest rosette's cell count.
    The largest rosette gets max_bins (10).  Smaller rosettes get
    ceil(n_cells / max_cells * max_bins), clamped to [min_bins, max_bins].
    """
    if max_cells < 1 or n_cells < 1:
        return min_bins
    return max(min_bins, min(max_bins, math.ceil(n_cells / max_cells * max_bins)))


# =========================================================================
# Data loading & merging
# =========================================================================
def load_and_merge_master(
    input_path: Path,
    master_path: Path | None,
    needed_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the merged rosette CSV.  If a master file is provided, attach any
    missing m/z columns from it (matched on rounded x, y).

    Returns
    -------
    df : DataFrame  --  the enriched cell-level table
    master_mz_cols : list[str]  --  ALL m/z column names found in the master
    """
    df = pd.read_csv(input_path)
    df["x"] = pd.to_numeric(df["x"], errors="coerce").round(2)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").round(2)

    master_mz_cols: list[str] = []

    if master_path is not None and master_path.exists():
        master = pd.read_csv(master_path)

        # normalise coordinate names
        rename_map = {}
        if "X" in master.columns:
            rename_map["X"] = "x"
        if "Y" in master.columns:
            rename_map["Y"] = "y"
        master = master.rename(columns=rename_map)
        master["x"] = pd.to_numeric(master["x"], errors="coerce").round(2)
        master["y"] = pd.to_numeric(master["y"], errors="coerce").round(2)

        master_mz_cols = get_mz_columns(master)

        # columns to bring in
        missing_in_df = [c for c in master_mz_cols if c not in df.columns]
        if needed_cols:
            for c in needed_cols:
                if c not in df.columns and c in master.columns and c not in missing_in_df:
                    missing_in_df.append(c)
        # also pull ncam only if present
        if "ncam only" in master.columns and "ncam only" not in df.columns:
            missing_in_df.append("ncam only")

        if missing_in_df:
            merge_cols = ["x", "y"] + missing_in_df
            master_sub = master[merge_cols].drop_duplicates(subset=["x", "y"])
            df = pd.merge(df, master_sub, on=["x", "y"], how="left")
            print(f"  Merged {len(missing_in_df)} additional columns from master file.")

    # fallback: if no master, use whatever mz columns are already in the data
    if not master_mz_cols:
        master_mz_cols = get_mz_columns(df)

    return df, master_mz_cols


# =========================================================================
# Core rosette pipeline
# =========================================================================
def full_rosette_pipeline(
    df: pd.DataFrame,
    max_thickness: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per rosette:
      r         -- distance to nearest Lumen cell (KDTree)
      theta     -- angle relative to lumen centroid
      r_norm    -- r / max(r) within each rosette  (independent normalisation)
      aspect_ratio -- lumen elongation via PCA singular-value ratio

    Binning schemes added:
      shell_bin_fixed5     -- fixed 5-bin quantile on r_norm per rosette
      shell_bin_dynamic    -- normalised bin count per rosette (max 10;
                              largest rosette -> 10 bins, others proportional)
      shell_bin_eqwidth10  -- 10 equal-width bins on r_norm (for variability plots)
    """
    required = {"x", "y", "rosette_id", "cell_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    rosette_stats: list[dict] = []

    # First pass: collect valid rosette ids and cell counts to find max_cells
    valid_rosettes: list[tuple] = []  # (rid, idx_mask)
    for rid in df["rosette_id"].dropna().unique():
        idx = df["rosette_id"] == rid
        rosette_cells = df.loc[idx]
        lumen_cells = rosette_cells[rosette_cells["cell_type"] == LUMEN_LABEL]
        if len(lumen_cells) < 5:
            continue
        valid_rosettes.append((rid, idx))

    max_cells = max(int(idx.sum()) for _, idx in valid_rosettes) if valid_rosettes else 1

    # Debug: show which rosette has max_cells for dynamic binning normalization
    if valid_rosettes:
        rosette_sizes = [(rid, int(idx.sum())) for rid, idx in valid_rosettes]
        largest_rid, largest_size = max(rosette_sizes, key=lambda x: x[1])
        print(f"  Dynamic binning: max_cells = {max_cells} from rosette {largest_rid}")
        print(f"  Rosette sizes: {dict(rosette_sizes)}")

    # Second pass: compute features using max_cells for normalised binning
    for rid, idx in valid_rosettes:
        rosette_cells = df.loc[idx]
        lumen_cells = rosette_cells[rosette_cells["cell_type"] == LUMEN_LABEL]

        cx, cy = lumen_cells["x"].mean(), lumen_cells["y"].mean()

        # aspect ratio via PCA
        pca = PCA(n_components=2)
        pca.fit(lumen_cells[["x", "y"]].values)
        s = pca.singular_values_
        aspect_ratio = float(s[0] / s[1]) if s[1] > 0 else float("inf")

        # distance to nearest lumen cell
        tree = KDTree(lumen_cells[["x", "y"]].values)
        r, _ = tree.query(rosette_cells[["x", "y"]].values, k=1)

        # angle
        dx = rosette_cells["x"].values - cx
        dy = rosette_cells["y"].values - cy
        theta = np.degrees(np.arctan2(dy, dx))

        n_cells = int(idx.sum())
        n_dyn = normalized_bin_count(n_cells, max_cells)

        df.loc[idx, "r"] = r.flatten()
        df.loc[idx, "theta"] = theta
        df.loc[idx, "aspect_ratio"] = aspect_ratio
        df.loc[idx, "n_dynamic_bins"] = n_dyn

        rosette_stats.append(
            {
                "rosette_id": rid,
                "aspect_ratio": aspect_ratio,
                "n_cells": n_cells,
                "n_dynamic_bins": n_dyn,
            }
        )

    # keep only rosettes where r was computed
    df = df.dropna(subset=["r"]).copy()

    if max_thickness is not None:
        df = df[df["r"] <= max_thickness].copy()

    # normalise r independently per rosette
    df["r_norm"] = df.groupby("rosette_id")["r"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0.0
    )

    # ---- Binning 1: fixed 5-bin quantile per rosette ----
    df["shell_bin_fixed5"] = (
        df.groupby("rosette_id")["r_norm"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"), q=5, labels=False, duplicates="drop"
            )
        )
        .astype(int)
    )
    df["shell_center_fixed5"] = df.groupby("rosette_id")[
        "shell_bin_fixed5"
    ].transform(
        lambda b: (b.astype(float) + 0.5) / float(b.max() + 1)
        if b.max() >= 0
        else 0.0
    )

    # ---- Binning 1b: fixed 8-bin quantile per rosette ----
    df["shell_bin_fixed8"] = (
        df.groupby("rosette_id")["r_norm"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"), q=8, labels=False, duplicates="drop"
            )
        )
        .astype(int)
    )
    df["shell_center_fixed8"] = df.groupby("rosette_id")[
        "shell_bin_fixed8"
    ].transform(
        lambda b: (b.astype(float) + 0.5) / float(b.max() + 1)
        if b.max() >= 0
        else 0.0
    )

    # ---- Binning 2: dynamic normalised-count per rosette ----
    shell_bin_dynamic = pd.Series(dtype=int, index=df.index)
    for rid, group in df.groupby("rosette_id"):
        n_bins = int(group["n_dynamic_bins"].iloc[0])
        r = group["r_norm"].rank(method="first")
        bins = pd.qcut(r, q=n_bins, labels=False, duplicates="drop").astype(int)
        shell_bin_dynamic.loc[group.index] = bins
    df["shell_bin_dynamic"] = shell_bin_dynamic
    df["shell_center_dynamic"] = df.groupby("rosette_id")[
        "shell_bin_dynamic"
    ].transform(
        lambda b: (b.astype(float) + 0.5) / float(b.max() + 1)
        if b.max() >= 0
        else 0.0
    )

    # ---- Binning 3: equal-width 10 bins on r_norm (common x-axis for variability plots) ----
    N_EQ = 10
    eq_bin = np.clip(np.floor(df["r_norm"].values * N_EQ).astype(int), 0, N_EQ - 1)
    df["shell_bin_eqwidth10"] = eq_bin
    df["shell_center_eqwidth10"] = (eq_bin.astype(float) + 0.5) / float(N_EQ)

    stats_df = pd.DataFrame(rosette_stats)
    return df, stats_df


# =========================================================================
# Delta / TIC normalisation
# =========================================================================
def build_delta_tic_long(
    df_proc: pd.DataFrame,
    targets: list[str],
    center_col: str,
    bin_col: str,
    tic_col: str = "TIC",
    mz_cols_for_tic: list[str] | None = None,
    normalization: str = "tic",
) -> pd.DataFrame:
    """
    Long-form table with columns:
      rosette_id, bin, center, feature, frac_mean, delta

    Normalization options:
      - "tic": frac = feature / TIC, delta = frac(bin) - frac(lumen_bin)
      - "log2": frac = log2(feature + 1), delta = frac(bin) - frac(lumen_bin)
      - "none": frac = feature, delta = frac(bin) - frac(lumen_bin)
    """
    if not targets:
        raise ValueError("No targets provided.")
    missing = [t for t in targets if t not in df_proc.columns]
    if missing:
        raise ValueError(f"Target columns not found in data: {missing}")

    df = df_proc.copy()

    # compute TIC if needed (for "tic" mode only, but compute anyway for compatibility)
    if tic_col not in df.columns:
        if mz_cols_for_tic is None:
            mz_cols_for_tic = get_mz_columns(df)
        if not mz_cols_for_tic:
            raise ValueError("No mz columns found for TIC computation.")
        df[tic_col] = df[mz_cols_for_tic].sum(axis=1)
    df[tic_col] = df[tic_col].replace(0, np.nan)

    frac_wide = pd.DataFrame(
        {
            center_col: df[center_col],
            bin_col: df[bin_col],
            "rosette_id": df["rosette_id"],
        }
    )

    # Apply normalization
    for t in targets:
        if normalization == "tic":
            frac_wide[t] = df[t] / df[tic_col]
        elif normalization == "log2":
            frac_wide[t] = np.log2(df[t] + 1)
        elif normalization == "none":
            frac_wide[t] = df[t]
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

    long_df = frac_wide.melt(
        id_vars=["rosette_id", bin_col, center_col],
        value_vars=targets,
        var_name="feature",
        value_name="frac",
    )

    g = (
        long_df.groupby(
            ["rosette_id", "feature", bin_col, center_col], as_index=False
        )["frac"]
        .mean()
        .rename(columns={"frac": "frac_mean"})
    )
    g = g.sort_values(["rosette_id", "feature", bin_col])
    g["lumen_frac"] = g.groupby(["rosette_id", "feature"])["frac_mean"].transform(
        "first"
    )
    g["delta"] = g["frac_mean"] - g["lumen_frac"]
    g["fold_change"] = g["delta"] / g["lumen_frac"].replace(0, np.nan)
    return g


# =========================================================================
# Direction statistics
# =========================================================================
def per_feature_direction_stats(
    delta_long: pd.DataFrame, center_col: str
) -> pd.DataFrame:
    """Aggregate Spearman rho (delta vs centre) per feature across all rosettes."""
    rows = []
    for (rid, feat), sub in delta_long.groupby(["rosette_id", "feature"]):
        sub = sub.sort_values(center_col)
        x = sub[center_col].values
        y = sub["delta"].values
        if len(np.unique(x)) < 3:
            continue
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        if pd.isna(rho):
            continue
        rows.append(
            {"rosette_id": rid, "feature": feat, "spearman_rho": float(rho)}
        )

    rosette_rhos = pd.DataFrame(rows)
    if rosette_rhos.empty:
        return pd.DataFrame(
            columns=["feature", "mean_rho", "median_rho", "std_rho", "frac_pos", "n_rosettes"]
        )

    feat_stats = (
        rosette_rhos.groupby("feature")["spearman_rho"]
        .agg(mean_rho="mean", median_rho="median", std_rho="std", n_rosettes="count")
        .reset_index()
    )
    feat_stats["frac_pos"] = (
        rosette_rhos.groupby("feature")["spearman_rho"]
        .apply(lambda s: float((s > 0).mean()))
        .values
    )
    return feat_stats


def per_rosette_direction_stats(
    delta_long: pd.DataFrame, center_col: str
) -> pd.DataFrame:
    """Per-rosette, per-feature Spearman rho of delta vs centre."""
    rows = []
    for (rid, feat), sub in delta_long.groupby(["rosette_id", "feature"]):
        sub = sub.sort_values(center_col)
        x = sub[center_col].values
        y = sub["delta"].values
        if len(np.unique(x)) < 3:
            continue
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        if pd.isna(rho):
            continue
        rows.append(
            {"rosette_id": rid, "feature": feat, "spearman_rho": float(rho)}
        )
    return pd.DataFrame(rows)


# =========================================================================
# Cell-level Spearman (continuous rho values)
# =========================================================================
def _cell_level_rho_rows(
    df_proc: pd.DataFrame,
    targets: list[str],
    bin_col: str,
    mz_cols_for_tic: list[str],
    normalization: str = "tic",
) -> list[dict]:
    """Compute per-(rosette, feature) Spearman rho at cell level.

    Normalization options:
      - "tic": frac = feature / TIC
      - "log2": frac = log2(feature + 1)
      - "none": frac = feature

    For each feature, the normalized value is computed per cell, then
    delta = frac - mean(frac in lumen bin).  Spearman rho is computed
    between delta and r_norm over all cells in the rosette, giving
    continuous rho values (unlike the bin-level discretised version).
    """
    # Compute TIC (even if not using it, for compatibility)
    tic = df_proc[mz_cols_for_tic].sum(axis=1).replace(0, np.nan)
    rows: list[dict] = []
    for feat in targets:
        if feat not in df_proc.columns:
            continue

        # Apply normalization
        if normalization == "tic":
            frac = df_proc[feat] / tic
        elif normalization == "log2":
            frac = np.log2(df_proc[feat] + 1)
        elif normalization == "none":
            frac = df_proc[feat]
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        for rid, group in df_proc.groupby("rosette_id"):
            g_frac = frac.loc[group.index]
            lumen_mask = group[bin_col] == group[bin_col].min()
            lumen_mean = g_frac.loc[lumen_mask].mean()
            delta = g_frac - lumen_mean
            r_norm = group["r_norm"]
            if len(r_norm.unique()) < 3 or delta.isna().all():
                continue
            rho = r_norm.corr(delta, method="spearman")
            if pd.isna(rho):
                continue
            rows.append(
                {"rosette_id": rid, "feature": feat, "spearman_rho": float(rho)}
            )
    return rows


def per_feature_direction_stats_cell_level(
    df_proc: pd.DataFrame,
    targets: list[str],
    bin_col: str,
    mz_cols_for_tic: list[str],
    normalization: str = "tic",
) -> pd.DataFrame:
    """Aggregate Spearman rho per feature across rosettes (cell-level)."""
    rows = _cell_level_rho_rows(df_proc, targets, bin_col, mz_cols_for_tic, normalization=normalization)
    rosette_rhos = pd.DataFrame(rows)
    if rosette_rhos.empty:
        return pd.DataFrame(
            columns=["feature", "mean_rho", "median_rho", "std_rho", "frac_pos", "n_rosettes"]
        )
    feat_stats = (
        rosette_rhos.groupby("feature")["spearman_rho"]
        .agg(mean_rho="mean", median_rho="median", std_rho="std", n_rosettes="count")
        .reset_index()
    )
    feat_stats["frac_pos"] = (
        rosette_rhos.groupby("feature")["spearman_rho"]
        .apply(lambda s: float((s > 0).mean()))
        .values
    )
    return feat_stats


def per_rosette_direction_stats_cell_level(
    df_proc: pd.DataFrame,
    targets: list[str],
    bin_col: str,
    mz_cols_for_tic: list[str],
    normalization: str = "tic",
) -> pd.DataFrame:
    """Per-rosette, per-feature Spearman rho (cell-level)."""
    rows = _cell_level_rho_rows(df_proc, targets, bin_col, mz_cols_for_tic, normalization=normalization)
    return pd.DataFrame(rows)


# =========================================================================
# Permutation test (full pool, memory-aware)
# =========================================================================
def permutation_test_full_pool(
    df_proc: pd.DataFrame,
    good: list[str],
    pool: list[str],
    n_perm: int,
    seed: int,
    center_col: str,
    bin_col: str,
    mz_cols_for_tic: list[str],
    normalization: str = "tic",
) -> tuple[pd.DataFrame, float, float]:
    """
    Compare GOOD set vs random sets of equal size drawn from pool.
    All m/z columns are already attached to df_proc (merged once at load),
    so each permutation draw is a fast column-index operation.

    Returns (random_scores_df, good_score, p_value).
    """
    rng = np.random.default_rng(seed)

    # --- GOOD score ---
    delta_good = build_delta_tic_long(
        df_proc,
        good,
        center_col=center_col,
        bin_col=bin_col,
        mz_cols_for_tic=mz_cols_for_tic,
        normalization=normalization,
    )
    stats_good = per_feature_direction_stats(delta_good, center_col=center_col)
    good_score = (
        float(stats_good["mean_rho"].mean()) if len(stats_good) else float("nan")
    )

    if len(pool) < len(good):
        raise ValueError(
            f"Pool size ({len(pool)}) < good set size ({len(good)})."
        )

    scores: list[float] = []
    for i in range(n_perm):
        sample = rng.choice(pool, size=len(good), replace=False).tolist()
        available = [s for s in sample if s in df_proc.columns]
        if len(available) < 2:
            continue
        try:
            delta = build_delta_tic_long(
                df_proc,
                available,
                center_col=center_col,
                bin_col=bin_col,
                mz_cols_for_tic=mz_cols_for_tic,
                normalization=normalization,
            )
            st = per_feature_direction_stats(delta, center_col=center_col)
            score = float(st["mean_rho"].mean()) if len(st) else np.nan
            scores.append(score)
        except Exception:
            continue

        if (i + 1) % 50 == 0:
            print(f"    Permutation {i + 1}/{n_perm} ...")

    rand_df = pd.DataFrame({"score": scores}).dropna()
    p = (
        float((rand_df["score"] >= good_score).mean())
        if len(rand_df)
        else float("nan")
    )
    return rand_df, good_score, p


def permutation_test_full_pool_cell_level(
    df_proc: pd.DataFrame,
    good: list[str],
    pool: list[str],
    n_perm: int,
    seed: int,
    bin_col: str,
    mz_cols_for_tic: list[str],
    normalization: str = "tic",
) -> tuple[pd.DataFrame, float, float]:
    """Cell-level variant of permutation_test_full_pool.

    Uses per-cell Spearman (delta_frac vs r_norm) instead of bin-averaged.
    """
    rng = np.random.default_rng(seed)

    stats_good = per_feature_direction_stats_cell_level(
        df_proc, good, bin_col=bin_col, mz_cols_for_tic=mz_cols_for_tic,
        normalization=normalization,
    )
    good_score = (
        float(stats_good["mean_rho"].mean()) if len(stats_good) else float("nan")
    )

    if len(pool) < len(good):
        raise ValueError(
            f"Pool size ({len(pool)}) < good set size ({len(good)})."
        )

    scores: list[float] = []
    for i in range(n_perm):
        sample = rng.choice(pool, size=len(good), replace=False).tolist()
        available = [s for s in sample if s in df_proc.columns]
        if len(available) < 2:
            continue
        try:
            st = per_feature_direction_stats_cell_level(
                df_proc, available, bin_col=bin_col,
                mz_cols_for_tic=mz_cols_for_tic,
                normalization=normalization,
            )
            score = float(st["mean_rho"].mean()) if len(st) else np.nan
            scores.append(score)
        except Exception:
            continue

        if (i + 1) % 50 == 0:
            print(f"    Permutation {i + 1}/{n_perm} ...")

    rand_df = pd.DataFrame({"score": scores}).dropna()
    p = (
        float((rand_df["score"] >= good_score).mean())
        if len(rand_df)
        else float("nan")
    )
    return rand_df, good_score, p


# =========================================================================
# Plotting
# =========================================================================
def plot_aspect_ratio(stats_df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(6, 4))
    sns.histplot(stats_df["aspect_ratio"], kde=True)
    plt.axvline(1.0, linestyle="--")
    plt.title("Rosette aspect ratio (lumen elongation)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


def plot_shell_bins_spatial(
    df: pd.DataFrame, bin_col: str, outpath: Path, title: str | None = None
):
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(df["x"], df["y"], c=df[bin_col], s=5, cmap="viridis")
    plt.gca().set_aspect("equal")
    plt.colorbar(sc, label=bin_col)
    plt.title(title or f"Spatial: {bin_col}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


# ---------- Binning comparison heatmaps ----------
def plot_binning_main(df: pd.DataFrame, outpath: Path):
    """Main figure panel: Fixed 8-bin quantile spatial heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df["x"], df["y"], c=df["shell_bin_fixed8"], s=8, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Fixed 8-Bin Quantile")
    fig.colorbar(sc, ax=ax, label="Radial bin")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


def plot_binning_supplementary(df: pd.DataFrame, outpath: Path):
    """Supplementary figure: Dynamic (Rice Rule) binning spatial heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df["x"], df["y"], c=df["shell_bin_dynamic"], s=8, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Dynamic (Rice Rule) Bins")
    fig.colorbar(sc, ax=ax, label="Radial bin")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


def plot_binning_comparison(df: pd.DataFrame, outpath: Path):
    """Legacy: All rosettes, side-by-side Fixed 5-bin vs Dynamic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, bin_col, label in [
        (axes[0], "shell_bin_fixed5", "Fixed 5-Bin Quantile"),
        (axes[1], "shell_bin_dynamic", "Dynamic (Rice Rule) Bins"),
    ]:
        sc = ax.scatter(df["x"], df["y"], c=df[bin_col], s=5, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, label="Bin")
    plt.suptitle("Binning Comparison: Fixed vs Dynamic", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


def plot_binning_comparison_per_rosette(
    df: pd.DataFrame, outdir: Path, stem: str
):
    """One side-by-side figure per rosette."""
    for rid in sorted(df["rosette_id"].dropna().unique()):
        sub = df[df["rosette_id"] == rid]
        n_dyn = (
            int(sub["n_dynamic_bins"].iloc[0])
            if "n_dynamic_bins" in sub.columns
            else "?"
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, bin_col, label in [
            (axes[0], "shell_bin_fixed5", "Fixed 5-Bin Quantile"),
            (axes[1], "shell_bin_dynamic", f"Dynamic ({n_dyn} bins, Rice Rule)"),
        ]:
            sc = ax.scatter(
                sub["x"], sub["y"], c=sub[bin_col], s=8, cmap="viridis"
            )
            ax.set_aspect("equal")
            ax.set_title(label)
            fig.colorbar(sc, ax=ax, label="Bin")
        plt.suptitle(
            f"Rosette {rid}: Fixed vs Dynamic Binning (n={len(sub)} cells)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            outdir / f"{stem}_binning_compare_rosette_{rid}.png", dpi=PUBLICATION_DPI
        )
        plt.close()


# ---------- Slope-by-mz ----------
COLOR_GOOD = "tab:green"
COLOR_BAD = "tab:red"
COLOR_UNKNOWN = "steelblue"


def get_delta_label(normalization: str) -> str:
    """Get appropriate delta label based on normalization method."""
    if normalization == "tic":
        return "Δ(fractional abundance)"
    elif normalization == "log2":
        return "Δ(log2 intensity)"
    elif normalization == "none":
        return "Δ(raw intensity)"
    else:
        return "Δ(value)"


def get_normalization_suffix(normalization: str) -> str:
    """Get human-readable suffix for plot titles."""
    if normalization == "tic":
        return "TIC-normalized"
    elif normalization == "log2":
        return "log2-transformed"
    elif normalization == "none":
        return "raw intensities"
    else:
        return normalization


def plot_slope_by_mz(
    feat_stats: pd.DataFrame,
    outpath: Path,
    title: str,
    good: list[str] | None = None,
    bad: list[str] | None = None,
    normalization: str = "tic",
):
    """Aggregate bar plot: one bar per feature, height = mean Spearman rho,
    colored by GOOD/BAD membership."""
    if feat_stats.empty:
        return
    good_set = set(good or [])
    bad_set = set(bad or [])
    fs = feat_stats.sort_values("mean_rho")
    colors = [
        COLOR_GOOD if f in good_set else COLOR_BAD if f in bad_set else COLOR_UNKNOWN
        for f in fs["feature"]
    ]
    plt.figure(figsize=(max(8, 0.25 * len(fs)), 4))
    plt.bar(fs["feature"], fs["mean_rho"], color=colors)
    plt.axhline(0, linewidth=1.0)
    plt.xticks(rotation=90, fontsize=7)
    delta_label = get_delta_label(normalization)
    plt.ylabel(f"Mean Spearman rho ({delta_label} vs radius)")
    plt.title(title)
    # legend for GOOD / BAD
    from matplotlib.patches import Patch
    handles = []
    if good_set:
        handles.append(Patch(facecolor=COLOR_GOOD, label="GOOD subset"))
    if bad_set:
        handles.append(Patch(facecolor=COLOR_BAD, label="BAD subset"))
    if handles:
        plt.legend(handles=handles, loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


def plot_slope_by_mz_per_rosette(
    rosette_rhos: pd.DataFrame,
    outdir: Path,
    stem: str,
    n_bins: int = 8,
    feature_order: list[str] | None = None,
    good: list[str] | None = None,
    bad: list[str] | None = None,
    normalization: str = "tic",
):
    """One bar plot per rosette showing per-feature Spearman rho.

    Parameters
    ----------
    feature_order : list[str] | None
        If provided, all rosette plots use this fixed x-axis ordering
        (e.g. sorted by aggregate mean_rho) for cross-rosette comparison.
    good, bad : list[str] | None
        Feature lists for GOOD/BAD color labeling.
    """
    if rosette_rhos.empty:
        return
    good_set = set(good or [])
    bad_set = set(bad or [])
    from matplotlib.patches import Patch

    delta_label = get_delta_label(normalization)
    norm_suffix = get_normalization_suffix(normalization)

    for rid in sorted(rosette_rhos["rosette_id"].unique()):
        sub = rosette_rhos[rosette_rhos["rosette_id"] == rid]
        if sub.empty:
            continue

        # Use global feature order if provided, otherwise sort by rho
        if feature_order is not None:
            ordered_feats = [f for f in feature_order if f in sub["feature"].values]
            sub = sub.set_index("feature").loc[ordered_feats].reset_index()
        else:
            sub = sub.sort_values("spearman_rho")

        colors = [
            COLOR_GOOD if f in good_set else COLOR_BAD if f in bad_set else COLOR_UNKNOWN
            for f in sub["feature"]
        ]
        plt.figure(figsize=(max(8, 0.25 * len(sub)), 4))
        plt.bar(sub["feature"], sub["spearman_rho"], color=colors)
        plt.axhline(0, linewidth=1.0)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel(f"Spearman rho ({delta_label} vs radius)")
        plt.title(f"Rosette {rid}: Slope-by-m/z ({n_bins} bins, {norm_suffix})")
        handles = []
        if good_set:
            handles.append(Patch(facecolor=COLOR_GOOD, label="GOOD subset"))
        if bad_set:
            handles.append(Patch(facecolor=COLOR_BAD, label="BAD subset"))
        if handles:
            plt.legend(handles=handles, loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(
            outdir / f"{stem}_slope_by_mz_rosette_{rid}.png",
            dpi=PUBLICATION_DPI,
        )
        plt.close()


# ---------- Decay / trend ----------
def plot_decay_mean_sd(
    df_proc: pd.DataFrame,
    target: str,
    center_col: str,
    outpath: Path,
    label: str,
):
    """Per-target: mean +/- SD band across bins."""
    if target not in df_proc.columns:
        return
    g = (
        df_proc.groupby(center_col)[target]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
        .sort_values(center_col)
    )
    x, y, sd = g[center_col].values, g["mean"].values, g["sd"].fillna(0).values
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o", label=label)
    plt.fill_between(x, y - sd, y + sd, alpha=0.25)
    plt.xlabel("Normalised radial position (0=lumen -> 1=outer)")
    plt.ylabel(target)
    plt.title(f"Radial trend: {target} -- {label} (mean +/- SD)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()



def plot_per_rosette_variability(
    delta_long: pd.DataFrame,
    features: list[str],
    center_col: str,
    outdir: Path,
    stem: str,
    group_label: str,
    color: str = "tab:blue",
    normalization: str = "tic",
):
    """
    One plot PER ROSETTE.
    Thin translucent lines: one per m/z feature in the subset.
    Bold line: mean across all m/z features in the subset.
    """
    sub = delta_long[delta_long["feature"].isin(features)]
    if sub.empty:
        return

    for rid in sorted(sub["rosette_id"].unique()):
        rsub = sub[sub["rosette_id"] == rid]
        if rsub.empty:
            continue

        plt.figure(figsize=(9, 5))

        # thin line per m/z feature
        for feat in features:
            fsub = rsub[rsub["feature"] == feat].sort_values(center_col)
            if fsub.empty:
                continue
            plt.plot(
                fsub[center_col],
                fsub["fold_change"],
                alpha=0.3,
                linewidth=1.0,
                color=color,
            )

        # bold mean across all m/z features
        mean_curve = (
            rsub.groupby(center_col)["fold_change"]
            .agg(mean="mean", sd="std")
            .reset_index()
            .sort_values(center_col)
        )
        x = mean_curve[center_col].values
        y = mean_curve["mean"].values
        sd = mean_curve["sd"].fillna(0).values

        plt.plot(
            x, y, linewidth=3.0, color=color, marker="o",
            label=f"{group_label} mean (n={len(features)} m/z)",
        )
        plt.fill_between(x, y - sd, y + sd, alpha=0.2, color=color)
        plt.axhline(0, linewidth=0.8, color="gray", linestyle="--")
        plt.xlabel("Normalised radial position (0=lumen -> 1=outer)")
        delta_label = get_delta_label(normalization)
        plt.ylabel(f"{delta_label} relative to lumen")
        plt.title(f"Rosette {rid}: {group_label} m/z variability")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(
            outdir / f"{stem}_variability_{group_label}_rosette_{rid}.png",
            dpi=PUBLICATION_DPI,
        )
        plt.close()


def plot_per_rosette_good_vs_bad(
    delta_long: pd.DataFrame,
    good: list[str],
    bad: list[str],
    center_col: str,
    outdir: Path,
    stem: str,
    normalization: str = "tic",
    highlight_features: list[str] | None = None,
):
    """One combined GOOD vs BAD variability plot per rosette.

    Args:
        highlight_features: Optional list of specific m/z features to highlight and label
    """
    sub = delta_long[delta_long["feature"].isin(good + bad)]
    if sub.empty:
        return

    highlight_set = set(highlight_features or [])

    for rid in sorted(sub["rosette_id"].unique()):
        rsub = sub[sub["rosette_id"] == rid]
        if rsub.empty:
            continue

        plt.figure(figsize=(10, 5.5))

        highlight_colors = ["gold", "darkcyan"]

        for feats, color, ls, glabel in [
            (good, "tab:green", "-", "GOOD"),
            (bad, "tab:red", "--", "BAD"),
        ]:
            fsub = rsub[rsub["feature"].isin(feats)]
            if fsub.empty:
                continue

            # Only plot highlighted features (skip all non-highlighted thin lines)
            for feat in feats:
                if feat not in highlight_set:
                    continue
                ff = fsub[fsub["feature"] == feat].sort_values(center_col)
                if ff.empty:
                    continue
                feat_idx = sorted(highlight_set).index(feat)
                plt.plot(
                    ff[center_col], ff["fold_change"],
                    alpha=0.9, linewidth=2.5,
                    color=highlight_colors[feat_idx % len(highlight_colors)],
                    linestyle="-",
                    label=f"{feat}",
                )

            # bold mean across m/z features
            gm = (
                fsub.groupby(center_col)["fold_change"]
                .agg(mean="mean", sd="std")
                .reset_index()
                .sort_values(center_col)
            )
            plt.plot(
                gm[center_col], gm["mean"],
                linewidth=3.0, color=color, marker="o", linestyle=ls,
                label=f"{glabel} mean",
            )
            plt.fill_between(
                gm[center_col],
                gm["mean"] - gm["sd"].fillna(0),
                gm["mean"] + gm["sd"].fillna(0),
                alpha=0.15, color=color,
            )

        plt.axhline(0, linewidth=0.8, color="gray", linestyle="--")
        plt.xlabel("Normalised radial position (0=lumen -> 1=outer)")
        delta_label = get_delta_label(normalization)
        plt.ylabel(f"{delta_label} relative to lumen")
        plt.title(f"Rosette {rid}: GOOD vs BAD m/z variability")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(
            outdir / f"{stem}_variability_GOOD_vs_BAD_rosette_{rid}.png",
            dpi=PUBLICATION_DPI,
        )
        plt.close()


# ---------- Good vs Bad overlay (v3 style) ----------
def plot_overlay_good_bad_delta(
    delta_long: pd.DataFrame,
    good: list[str],
    bad: list[str],
    center_col: str,
    outpath: Path,
    title: str,
    normalization: str = "tic",
    highlight_features: list[str] | None = None,
):
    """v3-style overlay: thin per-mz lines + thick group mean with error bars.

    Args:
        highlight_features: Optional list of specific m/z features to highlight and label
    """
    if delta_long.empty:
        return

    feat_curve = (
        delta_long.groupby(["feature", center_col], as_index=False)["fold_change"]
        .mean()
        .sort_values(["feature", center_col])
    )

    plt.figure(figsize=(9, 5))

    highlight_set = set(highlight_features or [])

    # Only plot highlighted features (each gets a distinct color)
    highlight_colors = ["gold", "darkcyan"]

    for feat in good:
        if feat not in highlight_set:
            continue
        sub = feat_curve[feat_curve["feature"] == feat]
        if len(sub):
            feat_idx = sorted(highlight_set).index(feat)
            plt.plot(
                sub[center_col],
                sub["fold_change"],
                linewidth=2.5,
                alpha=0.9,
                linestyle="-",
                color=highlight_colors[feat_idx % len(highlight_colors)],
                label=f"{feat}",
            )

    for feat in bad:
        if feat not in highlight_set:
            continue
        sub = feat_curve[feat_curve["feature"] == feat]
        if len(sub):
            feat_idx = sorted(highlight_set).index(feat)
            plt.plot(
                sub[center_col],
                sub["fold_change"],
                linewidth=2.5,
                alpha=0.9,
                linestyle="-",
                color=highlight_colors[feat_idx % len(highlight_colors)],
                label=f"{feat}",
            )

    def _group_mean(feats, ls, label, color):
        sub = feat_curve[feat_curve["feature"].isin(feats)]
        if sub.empty:
            return
        gm = (
            sub.groupby(center_col)["fold_change"]
            .agg(mean="mean", sd="std")
            .reset_index()
            .sort_values(center_col)
        )
        plt.errorbar(
            gm[center_col],
            gm["mean"],
            yerr=gm["sd"].fillna(0),
            fmt="o",
            capsize=3,
            elinewidth=1.5,
            linestyle=ls,
            linewidth=3.0,
            color=color,
            label=label,
        )

    _group_mean(good, "-", "GOOD mean +/- SD", color="tab:green")
    _group_mean(bad, "--", "BAD mean +/- SD", color="tab:red")

    plt.axhline(0, linewidth=1.0)
    plt.xlabel("Normalised radial position (0=lumen -> 1=outer)")
    delta_label = get_delta_label(normalization)
    plt.ylabel(f"{delta_label} relative to lumen")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


# ---------- Null distribution ----------
def plot_null_distribution(
    rand_df: pd.DataFrame,
    good_score: float,
    bad_score: float | None,
    outpath: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram with KDE
    sns.histplot(rand_df["score"], kde=True, ax=ax, color="steelblue", alpha=0.6)

    # Extend x-limits to show all scores clearly
    null_min = rand_df["score"].min()
    null_max = rand_df["score"].max()
    all_scores = [good_score]
    if bad_score is not None:
        all_scores.append(bad_score)

    x_min = min(null_min, *all_scores) - 0.1
    x_max = max(null_max, *all_scores) + 0.1
    ax.set_xlim(x_min, x_max)

    # Add vertical lines for observed scores
    ax.axvline(
        good_score,
        linestyle="-",
        linewidth=3,
        color="tab:green",
        label=f"GOOD = {good_score:.3f}",
        alpha=0.8,
    )
    if bad_score is not None:
        ax.axvline(
            bad_score,
            linestyle="--",
            linewidth=3,
            color="tab:red",
            label=f"BAD = {bad_score:.3f}",
            alpha=0.8,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Spearman rho (averaged over features)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


# ---------- Representative spatial maps ----------
def plot_spatial_value(
    df_one_rosette: pd.DataFrame,
    value_col: str,
    outpath: Path,
    title: str,
):
    plt.figure(figsize=(6, 6))
    vals = pd.to_numeric(df_one_rosette[value_col], errors="coerce")
    vmax = np.nanmax(np.abs(vals))
    vmin = -vmax
    sc = plt.scatter(
        df_one_rosette["x"],
        df_one_rosette["y"],
        c=vals,
        s=28,
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )
    plt.gca().set_aspect("equal")
    plt.colorbar(sc, label=value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=PUBLICATION_DPI)
    plt.close()


# =========================================================================
# Utility helpers (kept from v3)
# =========================================================================
def choose_representative_features(
    feat_stats: pd.DataFrame, good: list[str], bad: list[str]
) -> tuple[str | None, str | None]:
    """GOOD: highest mean_rho.  BAD: closest to 0 mean_rho."""
    good_feat = bad_feat = None
    if good:
        gs = feat_stats[feat_stats["feature"].isin(good)].dropna()
        if not gs.empty:
            good_feat = gs.sort_values("mean_rho", ascending=False).iloc[0][
                "feature"
            ]
    if bad:
        bs = feat_stats[feat_stats["feature"].isin(bad)].dropna()
        if not bs.empty:
            bs = bs.assign(abs_rho=bs["mean_rho"].abs())
            bad_feat = bs.sort_values("abs_rho", ascending=True).iloc[0][
                "feature"
            ]
    return good_feat, bad_feat


def rosette_rho_for_feature(
    delta_long: pd.DataFrame, feature: str, center_col: str
) -> pd.DataFrame:
    """Per-rosette Spearman rho of delta vs radius for one feature."""
    rows = []
    sub_all = delta_long[delta_long["feature"] == feature]
    for rid, sub in sub_all.groupby("rosette_id"):
        sub = sub.sort_values(center_col)
        x, y = sub[center_col].values, sub["delta"].values
        if len(np.unique(x)) < 3:
            continue
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        if pd.isna(rho):
            continue
        rows.append({"rosette_id": rid, "rho": float(rho)})
    return pd.DataFrame(rows)


def add_delta_tic_per_cell(
    df_proc: pd.DataFrame,
    feature: str,
    bin_col: str,
    mz_cols_for_tic: list[str],
) -> pd.DataFrame:
    """Add per-cell TIC, frac, and delta_frac columns for one feature."""
    df = df_proc.copy()
    df["TIC"] = df[mz_cols_for_tic].sum(axis=1).replace(0, np.nan)
    df["frac"] = df[feature] / df["TIC"]
    lumen_bin = df.groupby("rosette_id")[bin_col].transform("min")
    df["_is_lumen_bin"] = df[bin_col] == lumen_bin
    lumen_mean = (
        df[df["_is_lumen_bin"]].groupby("rosette_id")["frac"].mean()
    )
    df["lumen_frac_mean"] = df["rosette_id"].map(lumen_mean)
    df["delta_frac"] = df["frac"] - df["lumen_frac_mean"]
    df = df.drop(columns=["_is_lumen_bin"])
    return df


# =========================================================================
# CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="v4 Multi-rosette radial gradient analysis",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to merged_multi_rosette_data CSV (from preprocessing)",
    )
    ap.add_argument(
        "--master",
        default=None,
        help="Path to master metrics CSV (full m/z pool for permutation)",
    )
    ap.add_argument("--outdir", default="Analysis_Results_v4")
    ap.add_argument("--max_thickness", type=float, default=None)
    ap.add_argument(
        "--good_mz",
        default="",
        help='Comma-separated GOOD m/z list, e.g. "mz 914.741, mz 885.724"',
    )
    ap.add_argument(
        "--bad_mz",
        default="",
        help='Comma-separated BAD m/z list (optional; auto-sampled if omitted)',
    )
    ap.add_argument(
        "--n_perm",
        type=int,
        default=200,
        help="Number of permutation draws (0 to skip)",
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    ap.add_argument(
        "--exclude_rosettes",
        default="",
        help='Comma-separated rosette IDs to exclude, e.g. "4,5"',
    )
    ap.add_argument(
        "--spearman_level",
        choices=["bin", "cell"],
        default="bin",
        help=(
            "Level at which Spearman rho is computed. "
            "'bin' = correlate bin-averaged values (discrete rho); "
            "'cell' = correlate per-cell delta_frac vs r_norm (continuous rho). "
            "Default: bin"
        ),
    )
    ap.add_argument(
        "--n_bins",
        type=int,
        default=5,
        help=(
            "Number of quantile bins to use for binning (only applies when spearman_level='bin'). "
            "More bins = higher resolution but still aggregated to reduce noise. "
            "Examples: 5 (default), 10, 15, 20. Ignored when spearman_level='cell'."
        ),
    )
    ap.add_argument(
        "--normalization",
        choices=["tic", "log2", "none"],
        default="tic",
        help=(
            "Normalization method for m/z intensities. "
            "'tic' = divide by total ion current (compositional, creates artifacts); "
            "'log2' = log2(intensity+1) transformation (recommended, no artifacts); "
            "'none' = raw intensities (no normalization). "
            "Default: tic"
        ),
    )

    args = ap.parse_args()

    set_publication_style()

    inpath = Path(args.input).resolve()
    master_path = Path(args.master).resolve() if args.master else None
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    good = parse_list_flags(args.good_mz)
    bad = parse_list_flags(args.bad_mz)
    needed = good + bad

    # ============================================================
    # 1. Load & merge
    # ============================================================
    print("Loading data ...")
    df_raw, all_master_mz = load_and_merge_master(
        inpath, master_path, needed_cols=needed
    )
    print(
        f"  Cells: {len(df_raw)}  |  "
        f"Rosettes: {df_raw['rosette_id'].nunique()}"
    )
    print(f"  m/z columns in data: {len(get_mz_columns(df_raw))}")
    print(f"  m/z columns in master pool: {len(all_master_mz)}")

    # Exclude rosettes if requested
    if args.exclude_rosettes:
        exclude_ids = [x.strip() for x in args.exclude_rosettes.split(",") if x.strip()]
        # Try numeric conversion to match rosette_id dtype
        try:
            exclude_ids_num = [int(x) for x in exclude_ids]
        except ValueError:
            exclude_ids_num = exclude_ids
        before_cells = len(df_raw)
        before_rosettes = df_raw['rosette_id'].nunique()
        df_raw = df_raw[~df_raw["rosette_id"].isin(exclude_ids_num)].copy()
        after_rosettes = df_raw['rosette_id'].nunique()
        print(f"  Excluded rosettes {exclude_ids_num}:")
        print(f"    Rosettes: {before_rosettes} -> {after_rosettes}")
        print(f"    Cells: {before_cells} -> {len(df_raw)}")
        print(f"    Remaining rosettes for max_cells calculation: {sorted(df_raw['rosette_id'].unique())}")

    # ============================================================
    # 2. Core pipeline (per-rosette normalisation + binning)
    # ============================================================
    print("\nRunning per-rosette pipeline ...")
    df_proc, stats_df = full_rosette_pipeline(
        df_raw, max_thickness=args.max_thickness
    )

    # Add flexible N-bin quantile binning (user-specified via --n_bins)
    n_bins = args.n_bins
    print(f"\nCreating flexible {n_bins}-bin quantile binning ...")
    df_proc[f"shell_bin_flex{n_bins}"] = (
        df_proc.groupby("rosette_id")["r_norm"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"), q=n_bins, labels=False, duplicates="drop"
            )
        )
        .astype(int)
    )
    df_proc[f"shell_center_flex{n_bins}"] = df_proc.groupby("rosette_id")[
        f"shell_bin_flex{n_bins}"
    ].transform(
        lambda b: (b.astype(float) + 0.5) / float(b.max() + 1)
        if b.max() >= 0
        else 0.0
    )

    stem = inpath.stem
    df_proc.to_csv(outdir / f"{stem}_processed.csv", index=False)
    stats_df.to_csv(outdir / f"{stem}_rosette_stats.csv", index=False)

    print(f"  Rosettes processed: {len(stats_df)}")
    print(
        stats_df[
            ["rosette_id", "n_cells", "n_dynamic_bins", "aspect_ratio"]
        ].to_string(index=False)
    )

    # ============================================================
    # 3. Basic plots
    # ============================================================
    print("\nGenerating basic plots ...")
    plot_aspect_ratio(stats_df, outdir / f"{stem}_aspect_ratio.png")
    plot_shell_bins_spatial(
        df_proc,
        "shell_bin_fixed5",
        outdir / f"{stem}_bins_fixed5.png",
        "All rosettes: Fixed 5-Bin Quantile",
    )
    plot_shell_bins_spatial(
        df_proc,
        "shell_bin_dynamic",
        outdir / f"{stem}_bins_dynamic.png",
        "All rosettes: Dynamic (Rice Rule) Bins",
    )

    # ============================================================
    # 4. Binning comparison heatmaps
    # ============================================================
    print("Generating binning plots ...")
    plot_binning_main(df_proc, outdir / f"{stem}_binning_fixed8.png")
    plot_binning_supplementary(df_proc, outdir / f"{stem}_binning_dynamic_supplementary.png")
    plot_binning_comparison_per_rosette(df_proc, outdir, stem)

    # Precompute TIC using MASTER FILE m/z columns only (for consistent TIC across runs)
    mz_for_tic = [c for c in all_master_mz if c in df_proc.columns]
    if mz_for_tic:
        df_proc["TIC"] = df_proc[mz_for_tic].sum(axis=1).replace(0, np.nan)
        print(f"  TIC computed from {len(mz_for_tic)} master m/z columns.")

    # ============================================================
    # 5. GOOD / BAD analysis
    # ============================================================
    if not good:
        print("\nNo --good_mz provided. Skipping Good/Bad analysis.")
        print(f"\nDone. Outputs in {outdir}")
        return

    # verify columns exist
    good_missing = [g for g in good if g not in df_proc.columns]
    bad_missing = [b for b in bad if b not in df_proc.columns]
    if good_missing:
        print(f"  WARNING: GOOD columns not found (skipped): {good_missing}")
    if bad_missing:
        print(f"  WARNING: BAD columns not found (skipped): {bad_missing}")
    good = [g for g in good if g in df_proc.columns]
    bad = [b for b in bad if b in df_proc.columns]

    if not good:
        print("  No valid GOOD columns remain after merge. Stopping.")
        print(f"\nDone. Outputs in {outdir}")
        return

    # pool = master mz columns present in data, excluding GOOD and explicit BAD
    exclude = set(good + bad)
    pool_all = [
        c for c in all_master_mz if c in df_proc.columns and c not in exclude
    ]

    # auto-sample BAD if not provided
    rng = np.random.default_rng(args.seed)
    if not bad and len(pool_all) >= len(good):
        bad = rng.choice(pool_all, size=len(good), replace=False).tolist()
        print(f"\n  BAD set auto-sampled ({len(bad)} features) from pool.")

    compare_feats = good + bad

    # column aliases (flexible N-bin quantile binning, N specified by --n_bins)
    n_bins = args.n_bins
    center_col_main = f"shell_center_flex{n_bins}"
    bin_col_main = f"shell_bin_flex{n_bins}"

    # ------ 5a. Delta-TIC (flexible {n_bins} bins) ------
    print(f"\nBuilding delta-TIC tables (normalization: {args.normalization}) ...")
    delta_long_main = build_delta_tic_long(
        df_proc,
        compare_feats,
        center_col=center_col_main,
        bin_col=bin_col_main,
        mz_cols_for_tic=mz_for_tic,
        normalization=args.normalization,
    )

    # ------ 5b. Aggregate direction stats ------
    use_cell_level = args.spearman_level == "cell"
    rho_label = "cell-level" if use_cell_level else "bin-level"
    print(f"  Spearman mode: {rho_label}")

    if use_cell_level:
        feat_stats = per_feature_direction_stats_cell_level(
            df_proc, compare_feats,
            bin_col=bin_col_main, mz_cols_for_tic=mz_for_tic,
            normalization=args.normalization,
        )
    else:
        feat_stats = per_feature_direction_stats(
            delta_long_main, center_col=center_col_main
        )
    feat_stats.to_csv(
        outdir / f"{stem}_feature_direction_stats.csv", index=False
    )
    norm_suffix = get_normalization_suffix(args.normalization)
    plot_slope_by_mz(
        feat_stats,
        outdir / f"{stem}_slope_by_mz_aggregate.png",
        f"Aggregate slope-by-m/z ({rho_label} Spearman, {n_bins} bins, {norm_suffix})",
        good=good,
        bad=bad,
        normalization=args.normalization,
    )

    # ------ 5c. Per-rosette direction stats ------
    print("Computing per-rosette slope-by-mz ...")
    if use_cell_level:
        rosette_rhos = per_rosette_direction_stats_cell_level(
            df_proc, compare_feats,
            bin_col=bin_col_main, mz_cols_for_tic=mz_for_tic,
        )
    else:
        rosette_rhos = per_rosette_direction_stats(
            delta_long_main, center_col=center_col_main
        )
    rosette_rhos.to_csv(
        outdir / f"{stem}_per_rosette_direction_stats.csv", index=False
    )
    # Build a global feature order grouped by GOOD then BAD, each sorted by aggregate mean_rho
    good_order = feat_stats[feat_stats["feature"].isin(good)].sort_values("mean_rho")["feature"].tolist()
    bad_order = feat_stats[feat_stats["feature"].isin(bad)].sort_values("mean_rho")["feature"].tolist()
    global_feature_order = good_order + bad_order
    plot_slope_by_mz_per_rosette(
        rosette_rhos, outdir, stem, n_bins=n_bins,
        feature_order=global_feature_order,
        good=good,
        bad=bad,
        normalization=args.normalization,
    )

    # ------ 5d. Good vs Bad overlay (v3-style) ------
    if bad:
        print("Generating Good vs Bad overlay ...")
        plot_overlay_good_bad_delta(
            delta_long_main,
            good=good,
            bad=bad,
            center_col=center_col_main,
            outpath=outdir / f"{stem}_good_vs_bad_overlay.png",
            title=f"GOOD vs BAD: gradient from lumen ({n_bins} bins, {norm_suffix})",
            normalization=args.normalization,
            highlight_features=["mz 911.724", "mz 885.724"],
        )

    # ------ 5e. Per-Rosette Variability plots (reuse flexible N-bin delta table) ------
    print("Generating per-rosette variability plots ...")

    plot_per_rosette_variability(
        delta_long_main,
        good,
        center_col=center_col_main,
        outdir=outdir,
        stem=stem,
        group_label="GOOD",
        color="tab:green",
        normalization=args.normalization,
    )
    if bad:
        plot_per_rosette_variability(
            delta_long_main,
            bad,
            center_col=center_col_main,
            outdir=outdir,
            stem=stem,
            group_label="BAD",
            color="tab:red",
            normalization=args.normalization,
        )
        plot_per_rosette_good_vs_bad(
            delta_long_main,
            good=good,
            bad=bad,
            center_col=center_col_main,
            outdir=outdir,
            stem=stem,
            normalization=args.normalization,
            highlight_features=["mz 911.724", "mz 885.724"],
        )

    # ------ 5f. Permutation test ------
    if args.n_perm > 0 and len(pool_all) >= len(good):
        print(
            f"\nRunning permutation test "
            f"({args.n_perm} draws, pool={len(pool_all)}, {rho_label}) ..."
        )
        if use_cell_level:
            rand_df, good_score, p = permutation_test_full_pool_cell_level(
                df_proc=df_proc,
                good=good,
                pool=pool_all,
                n_perm=args.n_perm,
                seed=args.seed,
                bin_col=bin_col_main,
                mz_cols_for_tic=mz_for_tic,
                normalization=args.normalization,
            )
        else:
            rand_df, good_score, p = permutation_test_full_pool(
                df_proc=df_proc,
                good=good,
                pool=pool_all,
                n_perm=args.n_perm,
                seed=args.seed,
                center_col=center_col_main,
                bin_col=bin_col_main,
                mz_cols_for_tic=mz_for_tic,
                normalization=args.normalization,
            )
        rand_df.to_csv(
            outdir / f"{stem}_perm_random_scores.csv", index=False
        )

        bad_score = None
        if bad:
            if use_cell_level:
                st_bad = per_feature_direction_stats_cell_level(
                    df_proc, bad,
                    bin_col=bin_col_main, mz_cols_for_tic=mz_for_tic,
                    normalization=args.normalization,
                )
            else:
                delta_bad = build_delta_tic_long(
                    df_proc,
                    bad,
                    center_col=center_col_main,
                    bin_col=bin_col_main,
                    mz_cols_for_tic=mz_for_tic,
                    normalization=args.normalization,
                )
                st_bad = per_feature_direction_stats(
                    delta_bad, center_col=center_col_main
                )
            if not st_bad.empty:
                bad_score = float(st_bad["mean_rho"].mean())

        plot_null_distribution(
            rand_df,
            good_score,
            bad_score,
            outpath=outdir / f"{stem}_perm_null_distribution.png",
            title=(
                f"Permutation null (n={len(good)}, "
                f"draws={args.n_perm}): p={p:.4f}"
            ),
        )

        print("\n  Permutation Test Summary")
        print(f"    GOOD set size:  {len(good)}")
        print(f"    Pool size:      {len(pool_all)}")
        print(f"    Random draws:   {args.n_perm}")
        print(f"    GOOD mean rho:  {good_score:.4f}")
        if bad_score is not None:
            print(f"    BAD mean rho:   {bad_score:.4f}")
        print(f"    p-value (random >= good): {p:.4f}")
    else:
        if args.n_perm > 0:
            print(
                f"\n  Skipping permutation: pool ({len(pool_all)}) "
                f"< good set ({len(good)})."
            )
        else:
            print("\n  Permutation test skipped (n_perm=0).")

    # ------ 5g. Representative spatial maps ------
    print("\nGenerating representative spatial maps ...")
    good_feat, bad_feat = choose_representative_features(
        feat_stats, good, bad
    )

    rep: list[tuple[str, str, object]] = []
    if good_feat is not None:
        rr = rosette_rho_for_feature(
            delta_long_main, good_feat, center_col=center_col_main
        )
        if not rr.empty:
            rep_rid = rr.sort_values("rho", ascending=False).iloc[0][
                "rosette_id"
            ]
            rep.append(("good", good_feat, rep_rid))
    if bad_feat is not None:
        rr = rosette_rho_for_feature(
            delta_long_main, bad_feat, center_col=center_col_main
        )
        if not rr.empty:
            rr = rr.assign(abs_rho=rr["rho"].abs())
            rep_rid = rr.sort_values("abs_rho", ascending=True).iloc[0][
                "rosette_id"
            ]
            rep.append(("bad", bad_feat, rep_rid))

    for tag, feat, rid in rep:
        df_r = df_proc[df_proc["rosette_id"] == rid].copy()
        if df_r.empty:
            continue

        # RAW intensity
        plot_spatial_value(
            df_r,
            feat,
            outpath=outdir / f"{stem}_REP_{tag}_raw_{safe_name(feat)}.png",
            title=f"REP {tag.upper()}: RAW {feat} (rosette {rid})",
        )

        # delta(mz/TIC)
        df_delta = add_delta_tic_per_cell(
            df_r, feat, bin_col=bin_col_main, mz_cols_for_tic=mz_for_tic
        )
        plot_spatial_value(
            df_delta,
            "delta_frac",
            outpath=outdir
            / f"{stem}_REP_{tag}_deltaTIC_{safe_name(feat)}.png",
            title=f"REP {tag.upper()}: delta({feat}/TIC) (rosette {rid})",
        )

    # ------ 5h. Per-target decay plots ------
    print("Generating per-target decay plots ...")
    for t in compare_feats:
        if t not in df_proc.columns:
            continue
        sn = safe_name(t)
        plot_decay_mean_sd(
            df_proc,
            t,
            center_col=center_col_main,
            outpath=outdir / f"{stem}_decay_{sn}.png",
            label=f"Quantile {n_bins} bins",
        )

    print(f"\nDone. All outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
