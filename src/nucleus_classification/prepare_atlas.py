"""
Description: This file prepare the atlas from the paper - "A single-cell and spatially resolved atlas of human breast
             cancers" to perform NMF on one of the datasets we get


"""
import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csc_matrix
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.optimize import nnls


from src.utils import get_data_path, get_results_path


def compare_clusters(path: Path, genes_list: list, strategy, save_dir: Path):

    # Load Matrix
    count_matrix, metadata = load_files(path)

    # Gene Intersection
    if genes_list:
        genes_intersection = find_gene_intersection(genes_list, count_matrix.var_names)
        count_matrix = count_matrix[:, genes_intersection]
    count_matrix_norm = preprocess_matrix(count_matrix)

    sc.pp.pca(count_matrix_norm, n_comps=500)
    variance_explained = 0
    for ratio in count_matrix_norm.uns["pca"]["variance_ratio"]:
        variance_explained += ratio
    print(variance_explained)

    clusters = ["major_cluster", "minor_cluster"]
    for cluster_type in clusters:

        if cluster_type == "major_cluster":
            n_comp = len(count_matrix.obs["group.1"].unique())
            clusters_name = count_matrix.obs["group.1"].unique()
            group = "group.1"
        else:
            group = "group.2"
            clusters_name = count_matrix.obs["group.2"].unique()
            n_comp = len(count_matrix.obs["group.2"].unique())
            # TODO: create subclusters from the paper -> too much granularity with 27

        # Mean - Signature
        mean_signatures = []
        median_signatures = []
        for label in clusters_name:
            mean_signatures.append(np.mean(count_matrix_norm[count_matrix_norm.obs[group] == label, :].X.toarray(), axis=0))
            median_signatures.append(np.median(count_matrix_norm[count_matrix_norm.obs[group] == label, :].X.toarray(), axis=0))
        mean_signatures = np.array(mean_signatures)
        median_signatures = np.array(median_signatures)

        cut_off = 90
        if strategy == f"keep_high_genes":
            cut_off_value = np.percentile(mean_signatures, cut_off)
            mean_signatures[mean_signatures < cut_off_value] = 0.0
            cut_off_value = np.percentile(median_signatures, cut_off)
            median_signatures[median_signatures < cut_off_value] = 0.0

        # NMF - signature
        data_matrix = np.array(count_matrix_norm.X.todense())
        nmf_model = NMF(n_components=n_comp, init='random', random_state=0)
        _ = nmf_model.fit_transform(data_matrix)  # Basis matrix (cell memberships to components)
        nmf_signatures = nmf_model.components_  # Coefficient matrix (gene contributions to components)

        # # CORRELATIONS
        # correlations_mean = pairwise_row_correlations(nmf_signatures, mean_signatures)
        # plt.figure(figsize=(14, 10))  # Adjust the figure size as necessary
        # sns.heatmap(correlations_mean, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, cbar=True)
        # plt.xticks(np.array(range(0, len(clusters_name))) + 0.5, labels=clusters_name, rotation=45, ha="right")
        # plt.xlabel(f"{cluster_type} from Papers (mean aggregate)")
        # plt.ylabel("NMF signature")
        # plt.title("Correlation between mean expression and NMF signature")
        # plt.tight_layout()
        # if strategy == "keep_high_genes":
        #     plt.savefig(save_dir / f"{cluster_type}_mean_aggregate_high_genes-{cut_off}.png")
        # else:
        #     plt.savefig(save_dir / f"{cluster_type}_mean_aggregate.png")
        # plt.close()
        #
        # correlations_median = pairwise_row_correlations(nmf_signatures, median_signatures)
        # plt.figure(figsize=(14, 10))  # Adjust the figure size as necessary
        # sns.heatmap(correlations_median, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, cbar=True)
        # plt.xticks(np.array(range(0, len(clusters_name))) + 0.5, labels=clusters_name, rotation=43, ha="right")
        # plt.xlabel(f"{cluster_type} from Papers (median aggregate)")
        # plt.ylabel("NMF signature")
        # plt.title("Correlation between median expression and NMF signature")
        # plt.tight_layout()
        # if strategy == "keep_high_genes":
        #     plt.savefig(save_dir / f"{cluster_type}_median_aggregate_high_genes-{cut_off}.png")
        # else:
        #     plt.savefig(save_dir / f"{cluster_type}_median_aggregate.png")

        # NNLS:
        nnls_mean = get_nnls_score(mean_signatures, nmf_signatures)
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
        for i, ax in enumerate(axs.ravel()):
            ax.bar(range(0, 9), nnls_mean[i])
            ax.set_xticks(range(0, 9), clusters_name, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_dir / "nnls_projections.png")


def get_nnls_score(aggregates, nmf):
    projections = []

    #aggregates_norm = (aggregates - np.mean(aggregates, axis=1, keepdims=True)) / np.std(aggregates, axis=1, keepdims=True)
    #nmf_norm = (nmf - np.mean(nmf, axis=1, keepdims=True)) / np.std(nmf, axis=1, keepdims=True)
    aggregates_norm = aggregates
    nmf_norm = nmf

    for nmf_ in nmf_norm:
        projection, _ = nnls(aggregates_norm.T, nmf_)
        projections.append(projection)

    return projections


def pairwise_row_correlations(A, B):

    A_normalized = A
    A_normalized = (A - np.mean(A, axis=1, keepdims=True)) / np.std(A, axis=1, keepdims=True)
    B_normalized = B
    B_normalized = (B - np.mean(B, axis=1, keepdims=True)) / np.std(B, axis=1, keepdims=True)
    correlation_matrix = np.zeros((A_normalized.shape[0], B_normalized.shape[0]))
    for i in range(A_normalized.shape[0]):
        for j in range(B_normalized.shape[0]):
            correlation_matrix[i, j] = np.dot(A_normalized[i], B_normalized[j].T) / (A.shape[1] - 1)
            correlation_matrix[i, j] = np.corrcoef(A_normalized[i], B_normalized[j])[0, 1]

    return correlation_matrix


def find_gene_intersection(genes_1, genes_2):

    intersection = list(set(genes_1).intersection(genes_2))
    difference = list(set(genes_1).difference(genes_2))
    # TCIM -> C8orf4, CAVIN2 -> SDPR, TENT5C -> FAM46C, OPRPN -> PROL1, PCLAF -> KIAA0101
    if len(difference) > 0:
        print(f"The selected atlas do not overlap all genes: {difference}")
    intersection.extend(["C8orf4", "SDPR", "FAM46C", "PROL1", "KIAA0101"])
    return intersection


def load_files(path: Path):
    """

    :param path:
    :return: count_matrix, metadata
    """

    # Load Metadata
    metadata = pd.read_csv(path / "Whole_miniatlas_meta.csv", skiprows=1)
    metadata.index = metadata.TYPE

    # Load Features
    with ZipFile(path / "AllCells_raw_count_out.zip", 'r') as zObject:
        zObject.extractall(path=path)
    mtx_path = path / "BrCa_Atlas_Count_out"
    features = pd.read_csv(mtx_path / "features.tsv.gz", sep="\t", header=None)
    matrix = mmread(mtx_path / "matrix.mtx.gz")

    # Read Matrix and Replace obs and var by additional files
    atlas_matrix = sc.read_10x_mtx(mtx_path, var_names="gene_symbols", cache=True)
    atlas_matrix.var = features
    atlas_matrix.var_names = atlas_matrix.var[0]
    atlas_matrix.obs = metadata
    atlas_matrix.X = csc_matrix(matrix.T)

    return atlas_matrix, metadata


def preprocess_matrix(count_matrix, min_genes=2, min_cells=5):

    sc.pp.filter_cells(count_matrix, min_genes=min_genes)
    sc.pp.filter_genes(count_matrix, min_cells=min_cells)

    sc.pp.normalize_total(count_matrix, target_sum=1e4)
    sc.pp.log1p(count_matrix)

    return count_matrix


def prepare_atlas(path: Path, method="mean", genes_list=None, cluster_type="major", n_components: int = None):
    """ Prepare an atlas that will return gene signatures for a certain number of cell type

    :param path: path to the atlas data
    :param method: which method to use for gene signature extraction
    :param genes_list: subset of genes from Xenium experiment
    :param cluster_type: which cluster to use
    :param n_components: if the cluster type is not define use a number of components
    :return:
    """

    assert method in ["mean", "median", "NMF"]

    count_matrix, metadata = load_files(path)

    genes_intersection = None
    if genes_list:
        genes_intersection = find_gene_intersection(genes_list, count_matrix.var_names)
        count_matrix = count_matrix[:, genes_intersection]

    count_matrix_norm = preprocess_matrix(count_matrix)

    data_matrix = np.array(count_matrix_norm.X.todense())
    nmf_model = NMF(n_components=n_components, init='random', random_state=0)

    # Fit the model to the data matrix.
    W = nmf_model.fit_transform(data_matrix)  # Basis matrix (cell memberships to components)
    H = nmf_model.components_  # Coefficient matrix (gene contributions to components)

    cell_type_signature = H

    if genes_list:
        return cell_type_signature, genes_intersection
    return cell_type_signature,


def build_dir():
    dir_ = get_results_path() / "cell_type_signature"
    os.makedirs(dir_, exist_ok=True)
    return dir_


if __name__ == "__main__":

    run_main = False

    results_dir = build_dir()

    adata_path = get_data_path() / "Xenium_FFPE_Human_Breast_Cancer_Rep1.h5ad"
    adata = sc.read_h5ad(adata_path)
    genes_list_xenium = adata.var["SYMBOL"].to_list()
    atlas_path = get_data_path() / "Breast_Cancer_Atlas"

    if run_main:
        atlas = prepare_atlas(atlas_path, method="mean", genes_list=None)
    else:
        strategies = [None, "keep_high_genes"]
        for strategy in strategies:
            compare_clusters(atlas_path, None, strategy, results_dir)
