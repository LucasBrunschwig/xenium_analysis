import os

# Third party
import scanpy as sc
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt

# Relative import
from utils import load_xenium_data

if __name__ == "__main__":

    # Path compatibility with any OS
    data_path = os.path.join("..", "..", "scratch", "lbrunsch", "data", "Xenium_V1_FF_Mouse_Brain_MultiSection_1")
    save_path = os.path.join("..", "..", "scratch", "lbrunsch", "results")


    # Load Data
    adata = load_xenium_data(data_path)

    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)

    c_probes = (
        adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100
    )
    c_words = (
        adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100
    )
    print(f"Negative DNA probe count % : {c_probes}")
    print(f"Negative decoding count % : {c_words}")

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    axs[0].set_title("Total transcripts per cell")
    sns.histplot(
        adata.obs["total_counts"],
        kde=False,
        ax=axs[0],
    )

    axs[1].set_title("Unique transcripts per cell")
    sns.histplot(
        adata.obs["n_genes_by_counts"],
        kde=False,
        ax=axs[1],
    )

    axs[2].set_title("Area of segmented cells")
    sns.histplot(
        adata.obs["cell_area"],
        kde=False,
        ax=axs[2],
    )

    axs[3].set_title("Nucleus ratio")
    sns.histplot(
        adata.obs["nucleus_area"] / adata.obs["cell_area"],
        kde=False,
        ax=axs[3],
    )

    plt.savefig(os.path.join(save_path, "xenium_qc_analysis_mouse_replicate_1.png"))

    # Filter adata by number of counts per cell and number of gene abundance across cells
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)

    # Copy data in layers to to keep original data
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)  # necessary for UMAP (k-neighbors with weights)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    sc.pl.umap(
        adata,
        color=[
            "total_counts",
            "n_genes_by_counts",
            "leiden",
        ],
        wspace=0.4,
    )

    plt.savefig(os.path.join(save_path, "umap_mouse_replicate_1.png"))

    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        shape=None,
        color=[
            "leiden",
        ],
        wspace=0.4,
    )

    plt.savefig(os.path.join(save_path, "scatter_leiden_mouse_replicate_1.png"))

    # ---------------------------------------------------------------------------------------------------------------- #

    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
