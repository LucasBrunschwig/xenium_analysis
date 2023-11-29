# std
import os
from pathlib import Path
import shutil

# third party
import scanpy as sc
import pandas as pd
import gzip
import squidpy as sq
import numpy as np


def decompress(path_file):
    if not os.path.isfile(f'{path_file}'):
        with gzip.open(f'{path_file}.gz', 'rb') as f_in:
            with open(f'{path_file}', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def format_xenium_adata(path: Path, tag: str, output_path: str):
    """ Largely Inspired by https://github.com/Moldia/Xenium_benchmarking/blob/main/xb/formatting.py

    Parameters
    ----------
    path (Path): Path to the folder containing Xenium output
    tag (str): ?
    output_path (str): output_path where to save the formatted adata file

    Returns
    -------
    adata object

    """

    # decompress various files
    decompress(path / 'transcripts.csv')
    decompress(path / 'cell_feature_matrix' / 'barcodes.tsv')
    decompress(path / 'cell_feature_matrix' / 'features.tsv')
    decompress(path / 'cell_feature_matrix' / 'matrix.mtx')
    decompress(path / 'cells.csv')

    a = mmread(path / 'cell_feature_matrix' / 'matrix.mtx')
    ad = a.todense()

    cell_info = pd.read_csv(path / "cells.csv")
    features = pd.read_csv(path / 'cell_feature_matrix' / 'features.tsv', sep='\t', header=None, index_col=0)
    barcodes = pd.read_csv(path / 'cell_feature_matrix' / 'barcodes.tsv', header=None, index_col=0)
    adata = sc.AnnData(ad.transpose(), obs=cell_info, var=features)
    adata.var.index.name = 'index'
    adata.var.columns = ['gene_id', 'reason_of_inclusion']
    panel_info = pd.read_csv(path / 'panel.tsv', sep='\t')

    try:
        panel_info['Gene']
    except Exception as e:
        panel_info['Gene'] = panel_info['Name']

    dict_annotation = dict(zip(panel_info['Gene'],panel_info['Annotation']))
    dict_ENSEMBL = dict(zip(panel_info['Gene'],panel_info['Ensembl ID']))
    adata.var['Annotation'] = adata.var.index.map(dict_annotation)
    adata.var['Ensembl ID'] = adata.var.index.map(dict_ENSEMBL)
    adata.var['in_panel'] = adata.var.index.isin(panel_info['Gene'])
    transcripts = pd.read_csv(path / 'transcripts.csv', index_col=0)
    adata.uns['spots'] = transcripts
    try:
        UMAP=pd.read_csv(path+'/analysis/umap/gene_expression_2_components/projection.csv',index_col=0)
        adata.obsm['X_umap'] = np.array(UMAP)
        TSNE=pd.read_csv(path+'/analysis/tsne/gene_expression_2_components/projection.csv',index_col=0)
        adata.obsm['X_tsne'] = np.array(TSNE)
        PCA=pd.read_csv(path+'/analysis/PCA/gene_expression_10_components/projection.csv',index_col=0)
        adata.obsm['X_pca'] = np.array(PCA)
        clusters=pd.read_csv(path+'/analysis/clustering/gene_expression_graphclust/clusters.csv',index_col=0)
        adata.obs['graph_clusters']=list(clusters['Cluster'].astype(str))
        kmeans2=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans2_clusters']=list(kmeans2['Cluster'].astype(str))
        kmeans3=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans3_clusters']=list(kmeans3['Cluster'].astype(str))
        kmeans4=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans4_clusters']=list(kmeans4['Cluster'].astype(str))
        kmeans5=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans5_clusters']=list(kmeans5['Cluster'].astype(str))
        kmeans6=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans6_clusters']=list(kmeans6['Cluster'].astype(str))
        kmeans7=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans7_clusters']=list(kmeans7['Cluster'].astype(str))
        kmeans8=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans8_clusters']=list(kmeans8['Cluster'].astype(str))
        kmeans9=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans9_clusters']=list(kmeans9['Cluster'].astype(str))
        kmeans10=pd.read_csv(path+'/analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv',index_col=0)
        adata.obs['kmeans10_clusters']=list(kmeans10['Cluster'].astype(str))
    except:
        print('UMAP and clusters_could not be recovered')
    adata.write(output_path+tag+'.h5ad')
    return adata


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