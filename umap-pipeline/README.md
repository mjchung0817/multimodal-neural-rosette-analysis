# MALDI Single-Cell UMAP Analysis Pipeline

**Author:** Tomisin N. Adebayo

This pipeline processes single-cell MALDI-MS intensity matrices, removes isotopic peak duplicates, identifies cellular subpopulations via unsupervised clustering (PCA → kNN → Leiden → UMAP), and nominates m/z features that discriminate clusters. Where available, antibody-derived markers (e.g., SSEA-1 and NCAM) are merged into the AnnData object for visualization and statistical testing.

**For code access, contact:** oadebayo31@gatech.edu, aturaga6@gatech.edu, mchung98@gatech.edu

## Input

A metrics table (CSV) containing:
- **m/z intensity features**: columns starting with `mz` (e.g., `mz 1046.859`)
- **Cell-level metadata**: all remaining columns (e.g., spatial coordinates `X`, `Y`, morphology metrics, region IDs)

## Output

1. AnnData object containing the processed intensity matrix and metadata
2. Filtered peak list after isotope removal
3. Cluster labels and UMAP embeddings
4. Ranked m/z markers per cluster
5. Optional antibody-marker overlays and statistics

## Workflow

### 1. Load Data and Structure for Single-Cell Analysis

From the exported metrics table, construct:
- `X`: numeric matrix of shape (cells x m/z features)
- `obs`: per-cell metadata table
- `var`: per-feature table containing the numeric m/z value for each column (stored in `var["mz"]`)

This is the required structure for a `scanpy.AnnData` object.

### 2. Create AnnData Object and Preprocessing

Convert the matrix into an AnnData container and apply preprocessing:

1. **Total-intensity normalization** (`normalize_total`): Scale each cell to the same total ion intensity (`target_sum = 1e4`) to reduce differences driven by overall signal magnitude
2. **Log transform** (`log1p`): Apply `log(1 + x)` to stabilize variance and compress extreme values
3. **Preserve log-transformed values**: Store as `adata.layers["log1p"]` for later reuse (e.g., feature ranking)
4. **Z-score scaling** (`scale`): Z-score each m/z feature across cells (clip extreme values with `max_value=10`) for PCA/clustering

### 3. Isotopic Peak Removal (Monoisotopic Filtering)

Rule-based isotopic filtering to remove M+1 and M+2 duplicate peaks:

**Assumptions:**
- Charge state z = 1 (typical for MALDI)
- Isotope spacing ~1.0033548378 Da for 13C substitution (M+1)

**Algorithm:**
1. Sort features by numeric m/z
2. For each potential parent peak, search for peaks within +/- `da_tol` of `mz + 1.003...` (M+1) and `mz + 2.006...` (M+2)
3. Mark matched peaks as isotopes to remove

**Parameters:**
- `da_tol = 0.01 Da`
- `max_iso = 2` (remove M+1 and M+2)

**Output:**
- `no_isotopes_400to1100_daTol0p01.csv`: filtered dataset retaining only monoisotopic candidates
- `removed_isotope_cols_daTol0p01.txt`: list of removed isotope feature column names

A quality check verifies that removed peaks have valid parent peaks at `m - 1.003...` (M+1) or `m - 2.006...` (M+2) within tolerance.

### 4. Unsupervised Clustering and UMAP Embedding

Scanpy-style workflow to identify cellular subpopulations:

1. **PCA** (`sc.tl.pca`): Reduce dimensionality and denoise
2. **kNN graph** (`sc.pp.neighbors`): Build neighborhood graph in PCA space (`n_neighbors=15`, `n_pcs=20`)
3. **Leiden clustering** (`sc.tl.leiden`): Partition the kNN graph into communities (`resolution` controls granularity)
4. **UMAP embedding** (`sc.tl.umap`): 2D visualization

**Discriminative feature identification:** Wilcoxon rank-sum test (`rank_genes_groups`) ranks m/z features that best separate each cluster from the rest. m/z values enriched in a specific cluster suggest metabolite/lipid features characteristic of that subpopulation.

### 5. Visualization in UMAP Space and Tissue Space

1. Collect top marker m/z values per cluster (filtered by adjusted p-value and effect size)
2. Visualize cluster labels on UMAP
3. Project cluster identity back onto spatial coordinates (`X`, `Y`) to evaluate whether clusters correspond to spatially organized structures (e.g., rosette-like regions)

Individual clusters can be highlighted in spatial coordinates to assess whether they localize to rosette cores, enrich at boundaries, or appear spatially diffuse.

### 6. Cluster Composition by Experimental Condition (Optional)

If `adata.obs` contains a `condition` column, a normalized contingency table of cluster label x condition estimates whether specific subpopulations are enriched/depleted across conditions (e.g., treatment vs control).
