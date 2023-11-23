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

import matplotlib.pyplot as plt
import pandas as pd

# Relative
from utils import load_xenium_data, load_rna_seq_data
from visualization import visualize

# Third Party

from sklearn.cluster import KMeans
import numpy as np

RESULTS_DIR = Path("../../scratch/lbrunsch/results/k_means_clustering")
os.makedirs(RESULTS_DIR, exist_ok=True)


def k_mean_clustering(adata, n_clusters, init_centroids):
    return KMeans(n_clusters=n_clusters, init=init_centroids,
                  random_state=0, n_init="auto").fit(adata.X.toarray(), n_clusters)


def mean_cell_type_expression(adata_ref, label_names, label_key):

    print("Computing Mean ")

    cluster_means = {}
    for label in label_names:
        cluster_means[label] = np.mean(adata_ref.X[np.where(label == adata_ref.obs[label_key])[0], :], axis=0).A1

    return pd.DataFrame.from_dict(cluster_means, orient="index")


def main(label_key_):

    # Path to data
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    path_replicate_2 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_2"
    path_replicate_3 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_3"
    paths = [path_replicate_1]
    path_ref = data_path / "Brain_Atlas_RNA_seq/l5_all.agg.loom"

    # Load Data
    replicates = []
    for path in paths:
        replicates.append(load_xenium_data(path))
    adata_ref = load_rna_seq_data(path_ref)

    # Match Transcriptomics RNA
    intersect = np.intersect1d(adata_ref.var_names, replicates[0].var_names)
    adata_ref = adata_ref[:, intersect]
    for i, replicate in enumerate(replicates):
        replicates[i] = replicate[:, intersect]

    label_names = np.unique(adata_ref.obs[label_key_].tolist())
    cluster_means = mean_cell_type_expression(adata_ref, label_names, label_key=label_key_).to_numpy()

    for i in range(len(replicates)):
        k_ = k_mean_clustering(replicates[i], n_clusters=len(adata_ref.obs[label_key_].unique()),
                               init_centroids=cluster_means)

        # Trying to automatically impute the cell type
        # labels = [label_names[label_] for label_ in k_.labels_]

        replicates[i].obs["k_means_labels"] = k_.labels_
        replicates[i].obs["k_means_labels"] = replicates[i].obs["k_means_labels"].astype("category")
        visualize(replicates[i], "k_means_labels")
        plt.savefig(RESULTS_DIR / f"{label_key_}_{str(paths[i]).split(os.sep)[-1]}.png")
        plt.close()


if "__main__" == __name__:

    label_key = ["TaxonomyRank1", "TaxonomyRank2", "TaxonomyRank3", "TaxonomyRank4"]
    for key in label_key:
        main(key)

