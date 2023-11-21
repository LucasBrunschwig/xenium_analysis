"""
File: k_means_clustering.py
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

import pandas as pd

# Relative
from utils import load_xenium_data, load_rna_seq_data

# Third Party
import numpy as np

RESULTS_DIR = Path("../../scratch/lbrunsch/results/k_means_clustering")
os.makedirs(RESULTS_DIR, exist_ok=True)


def k_mean_clustering():
    pass


def mean_cell_type_expression(adata_ref, label_key):

    print("Computing Mean ")

    label_names = adata_ref.obs[label_key].tolist()

    cluster_means = {}
    for label in label_names:
        cluster_means[label] = np.mean(adata_ref.X[np.where(label_names[0] == adata_ref.obs["ClusterName"])[0], :], axis=0).squeeze(axis=0).A1

    return pd.DataFrame.from_dict(cluster_means, orient="index")


if "__main__" == __name__:

    #
    compute_cluster_means = False

    # Path to data
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    path_replicate_2 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_2"
    path_replicate_3 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_3"
    paths = [path_replicate_1]
    path_ref = data_path / "Brain_Atlas_RNA_seq/l5_all.loom"

    replicates = []
    for path in paths:
        replicates.append(load_xenium_data(path))
    adata_ref = load_rna_seq_data(path_ref)

    intersect = np.intersect1d(adata_ref.var_names, replicates[0].var_names)

    adata_ref = adata_ref[:, intersect]
    for i, replicate in enumerate(replicates):
        replicates[i] = replicate[:, intersect]

    if compute_cluster_means:
        cluster_means = mean_cell_type_expression(adata_ref, label_key="ClusterName")
    else:
        with open(RESULTS_DIR / "cluster_means.pkl", "rb") as file:
            cluster_means = pd.DataFrame(file)

