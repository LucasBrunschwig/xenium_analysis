# std
import os

# third party
import scanpy as sc
import pandas as pd
import gzip


def load_xenium_data(path):
    adata = sc.read_10x_h5(os.path.join(path, "cell_feature_matrix.h5"))
    with gzip.open(os.path.join(path, "cells.csv.gz"), "rt") as file:
        df = pd.read_csv(file)
    df.set_index(adata.obs_names, inplace=True)
    adata.obs = df
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

    return adata


def load_rna_seq_data(path):
    adata = sc.read_loom(path)

    # Ensure unique var names
    adata.var["SYMBOL"] = adata.var.index
    adata.var.set_index("Accession", drop=True, inplace=True)

    adata.obs["SYMBOL"] = adata.obs.index
    adata.obs_names_make_unique()

    return adata
