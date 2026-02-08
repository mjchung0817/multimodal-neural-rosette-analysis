# Rosette Radial Gradient Analysis Pipeline

Radial gradient analysis of metabolite distributions within neural rosettes from MALDI mass spectrometry imaging data.

## Overview

This pipeline quantifies radial metabolite gradients in neural rosette structures by:
1. Preprocessing cell-level MALDI data with rosette annotations
2. Computing radial binning (lumen → periphery)
3. Calculating fold-change profiles relative to the lumen
4. Statistical testing via Spearman correlation and permutation analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (with Example Data)

Example data is included in `example_data/`:

```bash
python radial_gradient_analysis.py \
  --input "example_data/merged_rosette_data_v1.csv" \
  --master "example_data/no_isotopes_400to1100_daTol0p01.csv" \
  --outdir "my_results/" \
  --good_mz "mz 917.739, mz 885.724, mz 888.74, mz 911.724, mz 914.741, mz 887.946, mz 890.733, mz 912.712" \
  --bad_mz "" \
  --n_perm 200 \
  --normalization log2 \
  --n_bins 8
```

## Input Data Requirements

### What Data Works With This Pipeline?

This pipeline is designed for **cell-level spatial omics data** where:
- Cells are grouped into discrete multicellular structures (rosettes, organoids, spheroids, colonies)
- Each structure has an identifiable center or lumen region
- Molecular features (metabolites, lipids, proteins) are quantified per cell

**Compatible data types:**
- MALDI mass spectrometry imaging (MSI) with single-cell resolution
- DESI-MSI or other ambient ionization MSI
- Spatial metabolomics/lipidomics from any platform
- Immunofluorescence intensity data (with appropriate column naming)
- Any spatial -omics data with cell coordinates and feature intensities

### File 1: Merged Rosette Data (`--input`)

Cell-level data with rosette assignments. See `example_data/merged_rosette_data_v1.csv`.

| Column | Required | Description |
|--------|----------|-------------|
| `x` | ✓ | X coordinate (pixels or µm) |
| `y` | ✓ | Y coordinate (pixels or µm) |
| `rosette_id` | ✓ | Unique identifier for each rosette/structure |
| `cell_type` | ✓ | Must include "Lumen" label for center cells |
| `distance` | Optional | Distance to structure centroid (used if cell_type missing) |
| `mz *` | Optional | Any columns starting with "mz " are treated as features |

**How lumen cells are identified:**
- If `cell_type` column exists: cells labeled "Lumen" are used
- Otherwise: cells within 20th percentile of `distance` to centroid

### File 2: Master Metrics (`--master`)

Full feature panel for permutation testing. See `example_data/no_isotopes_400to1100_daTol0p01.csv`.

| Column | Required | Description |
|--------|----------|-------------|
| `X` or `x` | ✓ | X coordinate (must match input file) |
| `Y` or `y` | ✓ | Y coordinate (must match input file) |
| `mz *` | ✓ | Feature columns (e.g., "mz 885.724", "mz 911.724") |

**Tips for the master file:**
- Should contain ALL detected features (not just your hypothesis set)
- Can be filtered (e.g., remove isotopes, specific m/z range)
- Coordinates are matched to input file via rounded (x, y) pairs
- Features not in master file won't be included in permutation pool

### Preparing Your Own Data

1. **From MALDI imaging software:** Export cell-level data with coordinates and peak intensities
2. **Rosette annotation:** Manually annotate rosette IDs (e.g., in ImageJ, QuPath, or custom script)
3. **Lumen labeling:** Either:
   - Add `cell_type` column with "Lumen" for center cells
   - Use `radial_gradient_preprocessing.py` to auto-label based on distance percentile

## Usage

### Option A: If you already have merged data with lumen labels

```bash
python radial_gradient_analysis.py \
  --input "your_data/merged_cells.csv" \
  --master "your_data/all_features.csv" \
  --outdir "results/" \
  --good_mz "mz 885.724, mz 911.724" \
  --n_perm 200 \
  --normalization log2 \
  --n_bins 8
```

### Option B: Preprocessing first (to merge and label lumen)

```bash
# Step 1: Merge per-rosette files with master metrics
python radial_gradient_preprocessing.py \
  --folder "data/" \
  --master "master_metrics.csv" \
  --target_mz "mz 885.724, mz 911.724"

# Step 2: Run analysis
python radial_gradient_analysis.py \
  --input "data/merged_multi_rosette_data_v2.csv" \
  --master "data/master_metrics.csv" \
  --outdir "results/" \
  --good_mz "mz 885.724, mz 911.724" \
  --n_perm 200
```

## Analysis Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | Required | Path to merged rosette CSV |
| `--master` | Required | Path to master metrics CSV (feature pool) |
| `--outdir` | `Analysis_Results_v4` | Output directory |
| `--good_mz` | `""` | Hypothesis features (comma-separated, e.g., "mz 885.724, mz 911.724") |
| `--bad_mz` | `""` | Comparison features (leave empty for random sampling) |
| `--n_perm` | `200` | Number of permutation draws |
| `--normalization` | `tic` | `log2` (recommended), `tic`, or `none` |
| `--n_bins` | `5` | Number of radial bins |
| `--spearman_level` | `bin` | `bin` (aggregated) or `cell` (per-cell correlation) |
| `--exclude_rosettes` | `""` | Rosette IDs to exclude (comma-separated) |

## Output Files

### CSV Results
- `*_processed.csv`: Full processed dataset with radial bins
- `*_rosette_stats.csv`: Per-rosette summary (n_cells, aspect_ratio, n_bins)
- `*_feature_direction_stats.csv`: Aggregate Spearman ρ per feature
- `*_per_rosette_direction_stats.csv`: Per-rosette Spearman ρ
- `*_perm_random_scores.csv`: Permutation null distribution scores

### Figures
- `*_binning_fixed8.png`: Spatial heatmap of radial binning
- `*_good_vs_bad_overlay.png`: GOOD vs BAD subset intensity profiles
- `*_perm_null_distribution.png`: Permutation test histogram
- `*_slope_by_mz_aggregate.png`: Bar plot of per-feature Spearman ρ
- `*_decay_mz_*.png`: Individual feature radial trend plots

See `example_output/` for sample figures.

## Methods

### Radial Binning
Cells are assigned to radial bins based on normalized distance to the nearest lumen cell:
1. **Lumen identification**: Cells labeled "Lumen" or within 20th percentile of distance to centroid
2. **Distance calculation**: KDTree query to nearest lumen cell
3. **Binning**: Fixed quantile binning (default 8 bins) on normalized radial distance

### Gradient Quantification
- **Delta**: `log2(intensity + 1)` or TIC-normalized fold-change relative to lumen bin mean
- **Spearman ρ**: Correlation between delta and radial position
  - Negative ρ = lumen-enriched (higher in center)
  - Positive ρ = periphery-enriched (higher at edge)

### Permutation Testing
Random draws of features from the full pool are compared against the hypothesis-driven "GOOD" subset to assess statistical significance of the observed gradient pattern.

## Citation

If you use this pipeline, please cite:

> [Citation TBD - Kemp Lab, Georgia Tech]

