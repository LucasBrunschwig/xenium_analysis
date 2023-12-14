# std
import os
from pathlib import Path
import shutil
import zipfile
import re

import anndata
# third party
import scanpy as sc
import pandas as pd
import gzip
import squidpy as sq
import numpy as np
import tifffile


def decompress(path_file, extension='gz'):
    if not os.path.isfile(f'{path_file}'):
        if extension == 'gz':
            with gzip.open(f'{path_file}.gz', 'rb') as f_in:
                with open(f'{path_file}', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif extension == 'zip':
            with zipfile.ZipFile(f'{path_file}.zip', 'r') as f_in:
                with open(f'{path_file}', 'w') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def format_xenium_adata(path: Path, output_path: Path,
                      segmentation_info: bool = True,
                      cluster_info: bool = True):
    """

    Parameters
    ----------
    path (Path): Path to the folder containing Xenium output
    output_path (str): output_path where to save the formatted adata file
    segmentation_info (bool): loads the nuclear and cellular boundaries
    cluster_info (bool): loads the predefined clustering (graph, kmeans (1-10))

    Returns
    -------
    adata object

    """
    if not os.path.isfile(Path(str(output_path) + '.h5ad')):
        # decompress various files
        decompress(path / 'transcripts.csv')
        decompress(path / 'cell_boundaries.csv')
        decompress(path / 'nucleus_boundaries.csv')

        # read data that contains cell feature matrix and some additional information
        adata = sc.read_10x_h5(path / "cell_feature_matrix.h5")

        cell = pd.read_csv(path / "cells.csv")

        # Integrate cells.csv in adata object
        cell.set_index(adata.obs_names, inplace=True)
        adata.obs = cell
        adata.obs["og_index"] = adata.obs.index
        adata.obs_names_make_unique()

        # Format Spatial information for plotting
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

        # Ensure unique index for gene by focusing on gene_ids
        adata.var["SYMBOL"] = adata.var.index
        adata.var.set_index("gene_ids", drop=True, inplace=True)

        # Specific information about transcript
        transcripts = pd.read_csv(path / 'transcripts.csv', index_col=0)
        transcripts = transcripts[~transcripts["feature_name"].str.contains("blank", case=False)]
        transcripts = transcripts[~transcripts["feature_name"].str.contains("negcontrol", case=False)]

        adata.uns['spots'] = transcripts

        # Retrieve UMAP, TSNE, PCA information
        additional_cat = ["X_umap", "X_tsne", "X_pca"]
        additional_paths = ['/analysis/umap/gene_expression_2_components/projection.csv',
                            '/analysis/tsne/gene_expression_2_components/projection.csv',
                            '/analysis/pca/gene_expression_10_components/projection.csv']
        for cat, sub_path in zip(additional_cat, additional_paths):
            try:
                X = pd.read_csv(Path(str(path) + sub_path))
                adata.obsm[cat] = np.array(X)
            except Exception as e:
                print(f"Failed retrieving: {additional_cat}")

        if segmentation_info:
            nuclear_boundaries = pd.read_csv(path / "nucleus_boundaries.csv")
            adata.uns['nucleus_boundaries'] = nuclear_boundaries
            cell_boundaries = pd.read_csv(path / "cell_boundaries.csv")
            adata.uns['cell_boundaries'] = cell_boundaries

        if cluster_info:
            # Retrieve Clustering and differential expression values
            additional_cat = (['graph_clusters'] +
                              [f'kmeans{k}_clusters' for k in range(2, 11)])
            additional_paths = (['/gene_expression_graphclust/'] +
                                [f'/gene_expression_kmeans_{k}_clusters/' for k in range(2, 11)])
            for main_folder, filename in zip(['clustering', 'diffexp'], ['clusters.csv', 'differential_expression.csv']):
                for cat, sub_path in zip(additional_cat, additional_paths):
                    try:
                        path_ = Path(str(path) + f'/analysis/' + main_folder + sub_path + filename)
                        X = pd.read_csv(path_, index_col=0)
                        if main_folder == "clustering":
                            adata.obs[cat] = list(X["Cluster"].astype(str))
                        elif main_folder == "diffexp":
                            if cat == 'graph_clusters':
                                k_mean_n = adata.obs[cat].to_numpy().astype(int).max()
                            else:
                                k_mean_n = int(re.search("\d+", sub_path).group())
                            for j in range(1, k_mean_n + 1):
                                adata.var[cat + f"_cluster{j}_log2_fold"] = X[f"Cluster {j} Log2 fold change"].tolist()
                                adata.var[cat + f"_cluster{j}_ajusted_pvalue"] = X[f"Cluster {j} Adjusted p value"].tolist()
                    except Exception as e:
                        print(f"Error: {e}")

        # Save h5ad file to specified location
        adata.write(Path(str(output_path) + '.h5ad'))
    else:
        adata = anndata.read_h5ad(Path(str(output_path) + '.h5ad'))

    return adata


def load_xenium_images(path: Path, type: str = None):
    """

    Parameters
    ----------
    path
    type

    Returns
    -------

    """
    results = []
    if type is None or type == "morphology":
        results.append(tifffile.imread(str(path / 'morphology.ome.tif')))
    if type is None or type == "focus":
        results.append(tifffile.imread(str(path / 'morphology_focus.ome.tif')))
    if type is None or type == "mip":
        results.append(tifffile.imread(str(path / 'morphology_mip.ome.tif')))
    return results


def load_xenium_data(path: Path, formatted: bool = True):
    """

    Parameters
    ----------
    path: if formatted expect a .h5ad, else the main sample folder
    formatted: if loading the formatted dataset

    Returns
    -------

    """

    if formatted and str(path).endswith(".h5ad"):
        adata = anndata.read_h5ad(path)
    elif not formatted:
        adata = format_xenium_adata(path, output_path=path)
    else:
        raise ValueError("Formatted Expect a '.h5ad' file path")

    # ##############################################################
    # Deprecated
    # # Load h5 file for transcriptomics matrix data
    # adata = sc.read_10x_h5(os.path.join(path, "cell_feature_matrix.h5"))
    #
    # # Load Observation for each spots
    # with gzip.open(os.path.join(path, "cells.csv.gz"), "rt") as file:
    #     df = pd.read_csv(file)
    #
    # # Combine both information
    # df.set_index(adata.obs_names, inplace=True)
    # adata.obs = df
    # adata.obs["og_index"] = adata.obs.index
    # adata.obs_names_make_unique()
    #
    # # Format Spatial information for plotting
    # adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
    #
    # # Ensure unique index for gene
    # adata.var["SYMBOL"] = adata.var.index
    # adata.var.set_index("gene_ids", drop=True, inplace=True)
    #
    # # Mark the 'mt' gene
    # adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    # ##############################################################

    return adata


def load_rna_seq_data(path):
    adata = sc.read_loom(path)

    # Ensure unique var names
    adata.var["SYMBOL"] = adata.var.index
    adata.var.set_index("Accession", drop=True, inplace=True)

    adata.obs["og_index"] = adata.obs.index
    adata.obs_names_make_unique()

    # Mark the 'mt' gene
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]

    return adata


def preprocess_transcriptomics(adata, filter_: bool = True):
    """Perform normalization on transcriptomics data obtained through xenium

        (1) Normalize total (2) log(X+1)
    """

    # Filter adata by number of counts per cell and number of gene abundance across cells
    if filter_:
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


if __name__ == "__main__":
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    path_output_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1_Formatted"
    format_xenium_adata(path=path_replicate_1, output_path=data_path / 'Xenium_FF_MB_1')
    load_xenium_images(path=path_replicate_1)
