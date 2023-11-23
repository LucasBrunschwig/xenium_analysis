# std
import os
from pathlib import Path

# third party
import scanpy as sc
import pandas as pd
import gzip
import squidpy as sq


def load_xenium_data(path):

    # Load h5 file for transcriptomics matrix data
    adata = sc.read_10x_h5(os.path.join(path, "cell_feature_matrix.h5"))

    # Load Observation for each spots
    with gzip.open(os.path.join(path, "cells.csv.gz"), "rt") as file:
        df = pd.read_csv(file)

    # Combine both information
    df.set_index(adata.obs_names, inplace=True)
    adata.obs = df

    # Format Spatial information for plotting
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

    # Ensure unique index for gene
    adata.var["SYMBOL"] = adata.var.index
    adata.var.set_index("gene_ids", drop=True, inplace=True)

    return adata


def load_rna_seq_data(path):
    adata = sc.read_loom(path)

    # Ensure unique var names
    adata.var["SYMBOL"] = adata.var.index
    adata.var.set_index("Accession", drop=True, inplace=True)

    adata.obs["SYMBOL"] = adata.obs.index
    adata.obs_names_make_unique()

    return adata


def preprocess_transcriptomics(adata):
    """Perform normalization on transcriptomics data obtained through xenium

        (1) Normalize total (2) log(X+1)
    """

    # Filter adata by number of counts per cell and number of gene abundance across cells
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)

    # Normalize
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    return adata


def plot_xenium_labels(adata, label_key):
    # Spatial Distribution of counts
    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        shape=None,
        color=[
            label_key,
        ],
        wspace=0.4,
    )


def get_name_from_path(path: Path) -> str:
    return str(path).split(os.sep)[-1]