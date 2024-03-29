"""
File: leiden_clustering.py
Author: Lucas Brunschwig
Email: lucas.brunschwig@hotmail.fr
GitHub: @LucasBrunschwig

Description: This file implements a naive approach to match cells with single cell RNA seq.
             The file uses k-means clustering with a number that match the number of cell types
             and match the centroid to the closest cell types.
"""

# Std
import os
from pathlib import Path

# Third party
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import seaborn as sns

# Relative
from src.utils import load_rna_seq_data, load_xenium_data, preprocess_transcriptomics, get_name_from_path
from src.visualization import visualize

N_NEIGHBORS = 50
N_COMPONENTS = 50

# Make directory
RESULTS_DIR = Path(f"../../scratch/lbrunsch/results/leiden_clustering/n_neighbors{N_NEIGHBORS}_n_pca{N_COMPONENTS}")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_DIR_XENIUM = RESULTS_DIR / "xenium"
os.makedirs(RESULTS_DIR_XENIUM, exist_ok=True)
RESULTS_DIR_REF = RESULTS_DIR / "ref"
os.makedirs(RESULTS_DIR_REF, exist_ok=True)


def compute_ref_labels(adata, n_comp: int = 50, n_neighbors: int = 13, variance_thrs=0.9):

    adata_ = adata.copy()
    sc.pp.pca(adata_, n_comps=n_comp)

    variance_explained = 0
    n_pcs = 0
    for ratio in adata_.uns["pca"]["variance_ratio"]:
        variance_explained += ratio
        n_pcs += 1
        if variance_explained > variance_thrs:
            break
    print(f"Number of PCs for more than 0.9 variance explained: {n_pcs} - {variance_explained}")

    # Compute leiden with neighbors
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors, n_pcs=n_pcs)  # necessary for UMAP (k-neighbors with weights)
    sc.tl.leiden(adata_)

    # Project data to umap and visualize label
    sc.tl.umap(adata_)
    return adata_


def main():
    # Load Data
    data_path = Path("../../scratch/lbrunsch/data")

    path_ref = data_path / "Brain_Atlas_RNA_seq/l5_all.loom"
    adata_ref = load_rna_seq_data(path_ref)

    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    adata_xenium = load_xenium_data(path_replicate_1)

    # Perform Data Preprocessing (intersect data)
    adata_ref = preprocess_transcriptomics(adata_ref)
    intersect = np.intersect1d(adata_ref.var_names, adata_xenium.var_names)
    adata_ref = adata_ref[:, intersect]

    adata_xenium.var.index = adata_xenium.var["SYMBOL"]
    adata_ref.var.index = adata_ref.var["SYMBOL"]

    n_comps = [N_COMPONENTS]
    for comp in n_comps:
        print(f"Adata Ref: {comp}")
        sc.pp.pca(adata_ref, n_comps=comp)
        sc.pp.neighbors(adata_ref, n_neighbors=N_NEIGHBORS) # necessary for UMAP (k-neighbors with weights)
        sc.tl.umap(adata_ref)
        sc.tl.leiden(adata_ref)
        sc.tl.rank_genes_groups(adata_ref, groupby="leiden")

        # Store the new clusters for C2L
        adata_ref_copy = sc.read_loom(path_ref)
        adata_ref_copy.obs["leiden"] = adata_ref.obs["leiden"].values.tolist()
        adata_ref_copy.write_loom(RESULTS_DIR / "l5_all_leiden.loom")

        fig1, ax1 = plt.subplots()
        sc.pl.umap(
            adata_ref,
            ax=ax1,
            color=["leiden"],
            wspace=0.4,
            show=False
        )
        fig1.savefig(RESULTS_DIR_REF / f"total_brain_mouse_atlas_leiden_PCA{comp}.png", bbox_inches="tight")
        plt.close(fig1)

        sc.pl.rank_genes_groups(adata_ref, groupby="leiden", show=False)
        plt.savefig(RESULTS_DIR_REF / f"total_rank_gene_groups_ref_PCA{comp}.png", bbox_inches="tight")
        plt.close()

    # Perform data processing
    adata_xenium = preprocess_transcriptomics(adata_xenium)

    # Use various principal components to evaluate
    n_comps = [40, 50, None]

    for n_comp in n_comps:
        print(f"Performing Clustering on Xenium {n_comp}")
        if n_comps is not None:
            sc.pp.pca(adata_xenium, n_comps=n_comp)  # even though not many genes -> noise reduction / batch effect
        sc.pp.neighbors(adata_xenium, n_neighbors=N_NEIGHBORS)  # necessary for UMAP (k-neighbors with weights)
        sc.tl.umap(adata_xenium)
        sc.tl.leiden(adata_xenium)

        sc.tl.rank_genes_groups(adata_xenium, groupby="leiden")
        sc.pl.rank_genes_groups(adata_xenium, show=False)
        plt.savefig(
            RESULTS_DIR_XENIUM / f"rank_genes_group_{get_name_from_path(path_replicate_1)}_PCA{n_comp}_Neighbors{N_NEIGHBORS}.png",
            bbox_inches="tight")
        plt.close()

        n_categories = adata_xenium.obs['leiden'].nunique()
        if n_categories > 26:
            palette = sns.color_palette("hls", n_categories)
            adata_xenium.uns['leiden_colors'] = palette

        visualize(adata_xenium, label_key="leiden")
        plt.savefig(
            RESULTS_DIR_XENIUM / f"leiden_{get_name_from_path(path_replicate_1)}_PCA{n_comp}_Neighbors{N_NEIGHBORS}.png", )
        plt.close()

    return 0


if "__main__" == __name__:
    main()
