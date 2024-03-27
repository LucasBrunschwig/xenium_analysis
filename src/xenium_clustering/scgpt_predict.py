import os
from pathlib import Path
import numpy as np
from scipy.stats import mode
import scanpy as sc
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import sys

from src.utils import get_data_path, get_results_path
from src.nucleus_classification.prepare_atlas import load_files, find_gene_intersection

import scgpt as scg

# extra dependency for similarity search
try:
    import faiss
    faiss_imported = True
except ImportError:
    faiss_imported = False
    print(
        "faiss not installed! We highly recommend installing it for fast similarity search."
    )
    print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")

warnings.filterwarnings("ignore", category=ResourceWarning)


def build_dir():
    dir_ = get_results_path() / "scgpt"
    os.makedirs(dir_, exist_ok=True)
    return dir_


def main():

    results_dir = build_dir()

    adata_path = get_data_path() / "Xenium_FFPE_Human_Breast_Cancer_Rep1.h5ad"
    adata = sc.read_h5ad(adata_path)
    genes_list_xenium = adata.var["SYMBOL"].to_list()
    atlas_path = get_data_path() / "Breast_Cancer_Atlas"

    count_matrix, metadata = load_files(atlas_path)

    genes_intersection = find_gene_intersection(genes_list_xenium, count_matrix.var_names)
    count_matrix = count_matrix[:, genes_intersection]

    model_dir = get_data_path() / "scGPT_human"

    cell_type_key = "celltype"
    gene_col = 0

    ref_embed_adata = scg.tasks.embed_data(
        count_matrix,
        model_dir,
        gene_col=gene_col,
        batch_size=64,
    )

    test_embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        batch_size=64,
        device="mps"
    )

    # concatenate the two datasets
    adata_concat = test_embed_adata.concatenate(ref_embed_adata, batch_key="dataset")
    # mark the reference vs. query dataset
    adata_concat.obs["is_ref"] = ["Query"] * len(test_embed_adata) + ["Reference"] * len(
        ref_embed_adata
    )
    adata_concat.obs["is_ref"] = adata_concat.obs["is_ref"].astype("category")
    # mask the query dataset cell types
    adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].astype("category")
    adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].cat.add_categories(["To be predicted"])
    adata_concat.obs[cell_type_key][: len(test_embed_adata)] = "To be predicted"

    sc.pp.neighbors(adata_concat, use_rep="X_scGPT")
    sc.tl.umap(adata_concat)
    sc.pl.umap(
        adata_concat, color=["is_ref", cell_type_key], wspace=0.4, frameon=False, ncols=1
    )


if __name__ == "__main__":
    main()
