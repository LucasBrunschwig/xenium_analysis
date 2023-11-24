# Std
import os
from pathlib import Path

# Third party
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import seaborn as sns
import cell2location
from cell2location.utils.filtering import filter_genes
import scanpy as sc
import numpy as np
import scvi
import squidpy as sq
from scipy.sparse import csr_matrix
import logging
import torch

# relative
from utils import load_xenium_data, load_rna_seq_data
from leiden_clustering import compute_ref_labels

scvi.settings.seed = 0

RESULTS_DIR = Path("../../scratch/lbrunsch/results/cell2location")
RESULTS_DIR_SIGNATURE = RESULTS_DIR / "adata_ref"
RESULTS_DIR_C2L = RESULTS_DIR / "cell2location"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR_C2L, exist_ok=True)
os.makedirs(RESULTS_DIR_SIGNATURE, exist_ok=True)

logging.basicConfig(filename='../../scratch/lbrunsch/results/cell2location/log.txt', level=logging.INFO)


HOUSE_KEEPING_GENES_ENSEMBLE_ID = [
    "ENSMUSG00000005610",
    "ENSMUSG00000005779",
    "ENSMUSG00000010376",
    "ENSMUSG00000014294",
    "ENSMUSG00000014769",
    "ENSMUSG00000015671",
    "ENSMUSG00000018286",
    "ENSMUSG00000018567",
    "ENSMUSG00000019362",
    "ENSMUSG00000024248",
    "ENSMUSG00000024870",
    "ENSMUSG00000026750",
    "ENSMUSG00000028452",
    "ENSMUSG00000028837",
    "ENSMUSG00000029649",
    "ENSMUSG00000031532",
    "ENSMUSG00000032301",
    "ENSMUSG00000035242",
    "ENSMUSG00000035530",
    "ENSMUSG00000041881",
    "ENSMUSG00000048076",
    "ENSMUSG00000060073",
    "ENSMUSG00000069744",
    "ENSMUSG00000072772",
    "ENSMUSG00000078812",
    "ENSMUSG00000084786"
]


def load_replicates(paths: list):

    adata_list = []
    for path in paths:
        adata = load_xenium_data(path)
        adata.obs['sample'] = str(path).split(os.sep)[-1]
        adata.X = csr_matrix(adata.X)
        adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
        adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze() / adata.obs['total_counts']

        # add sample name to obs names
        adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
        adata.obs_names = adata.obs["sample"] + '_' + adata.obs_names
        adata.obs.index.name = 'spot_id'

        adata_list.append(adata)

    return adata_list


def qc_metrics(adata):
    r""" This function calculates QC metrics

    :param adata: receive AnnData object
    """
    adata_vis = adata.copy()

    for sample in adata.obs["sample"].unique():
        adata_sample = adata_vis[adata_vis.obs['sample'].isin([sample]), :].copy()

        # Calculate QC metrics
        adata_sample.X = adata_sample.X.toarray()
        sc.pp.calculate_qc_metrics(adata_sample, inplace=True, percent_top=[50, 100, 200])

        fig, axs = plt.subplots(1, 4, figsize=(24, 5))

        # Total Counts per cell
        sns.histplot(
            adata.obs["total_counts"],
            kde=False,
            ax=axs[0],
        )

        formatter = FuncFormatter(lambda x, _: f'{int(x / 1000)}K')
        original_formatter = plt.gca().xaxis.get_major_formatter()
        plt.gca().xaxis.set_major_formatter(formatter)

        # For each gene compute the number of cells that express it
        sns.histplot(
            adata_sample.var["n_cells_by_counts"],
            kde=False,
            ax=axs[1],
        )

        plt.gca().xaxis.set_major_formatter(original_formatter)

        # Unique transcripts per cell
        sns.histplot(
            adata_sample.obs["n_genes_by_counts"],
            kde=False,
            ax=axs[2],
        )

        # Area of segmented cells
        axs[2].set_title("Area of segmented cells")
        sns.histplot(
            adata_sample.obs["cell_area"],
            kde=False,
            ax=axs[3],
        )

        fig.savefig(RESULTS_DIR / f"qc_metrics_histogram_{sample}.png")
        plt.close(fig)

        # Spatial Distribution of counts
        sq.pl.spatial_scatter(
            adata_sample,
            library_id="spatial",
            shape=None,
            color=[
                "total_counts", "n_genes_by_counts",
            ],
            wspace=0.4,
        )

        plt.savefig(RESULTS_DIR / f"spatial_observation_{sample}.png")
        plt.close()


def plot_umap_samples(adata_vis):

    adata_vis_plt = adata_vis.copy()

    # log(p + 1)
    sc.pp.log1p(adata_vis_plt)

    # Scale the data ( (data - mean) / sd )
    sc.pp.scale(adata_vis_plt, max_value=10)

    # PCA, KNN construction, UMAP
    sc.tl.pca(adata_vis_plt, svd_solver='arpack', n_comps=40)
    sc.pp.neighbors(adata_vis_plt, n_neighbors=20, n_pcs=40, metric='cosine')
    sc.tl.umap(adata_vis_plt, min_dist=0.3, spread=1)

    # Plot
    with mpl.rc_context({'figure.figsize': [8, 8],
                         'axes.facecolor': 'white'}):
        sc.pl.umap(adata_vis_plt, color=['sample'], size=2,
                   color_map='RdPu', ncols=1,
                   legend_fontsize=10)
    plt.savefig(RESULTS_DIR / "umap_samples.png")
    plt.close()


def plot_umap_ref(adata, cell_taxonomy: list):
    adata_vis_plt = adata.copy()
    # log(p + 1)
    sc.pp.log1p(adata_vis_plt)

    adata_vis_plt.var['highly_variable'] = False
    sc.pp.highly_variable_genes(adata_vis_plt, min_mean=0.0125, max_mean=5, min_disp=0.5, n_top_genes=1000)

    hvg_list = list(adata_vis_plt.var_names[adata_vis_plt.var['highly_variable']])
    adata_vis_plt.var.loc[hvg_list, 'highly_variable'] = True
    # Scale the data ( (data - mean) / sd )
    sc.pp.scale(adata_vis_plt, max_value=10)
    # PCA, KNN construction, UMAP
    sc.tl.pca(adata_vis_plt, svd_solver='arpack', n_comps=40, use_highly_variable=True)
    sc.pp.neighbors(adata_vis_plt, n_neighbors=20, n_pcs=40, metric='cosine')
    sc.tl.umap(adata_vis_plt, min_dist=0.3, spread=1)

    with mpl.rc_context({'axes.facecolor': 'white', 'savefig.bbox': 'tight'}):
        sc.pl.umap(adata_vis_plt, color=cell_taxonomy, size=2,
                   color_map='RdPu', ncols=1,
                   legend_fontsize=10)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "umap_ref.png")
        plt.close()


def filter_gene_index(adata, cell_cutoff=5, cell_cutoff2=0.03, nonz_mean_cutoff=1.12):
    return filter_genes(adata,
                        cell_count_cutoff=cell_cutoff,
                        cell_percentage_cutoff2=cell_cutoff2,
                        nonz_mean_cutoff=nonz_mean_cutoff)


def signature_ref(annotated_ref_seq, label, save_path):

    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=annotated_ref_seq,
                                                       # 10X reaction / sample / batch
                                                       # batch_key='SampleID',
                                                       # cell type, co-variate used for constructing signatures
                                                       labels_key=label,
                                                       # multiplicative technical effects (platform, 3' - 5', donor)
                                                       # categorical_covariate_keys=["Age"]  # "ChipID"]
                                                       )

    from cell2location.models import RegressionModel

    mod = RegressionModel(annotated_ref_seq)

    # view anndata_setup as a sanity check
    mod.view_anndata_setup()

    # train the probabilistic model
    mod.train(max_epochs=5000, use_gpu=True)

    # Plot History
    mod.plot_history(10)
    plt.savefig(save_path / "adata_signature_training.png")
    plt.close()

    # Save model
    mod.save(f"{save_path}", overwrite=True)

    return mod


def run_cell2location(adata_vis, inf_aver, save_path):

    sc.settings.set_figure_params(dpi=100, color_map='viridis', dpi_save=100,
                                  vector_friendly=True,
                                  facecolor='white')

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

    # create and train the model
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=1,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()

    mod.train(max_epochs=5000,
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
              use_gpu=True,
              )

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    mod.plot_history(50)
    plt.legend(labels=['full data training'])
    plt.savefig(save_path / "plot_history_c2l.png")
    plt.close()

    mod.save(f"{save_path}", overwrite=True)

    return mod


def cell2location_xenium(extract_signature: bool = True, run_c2l_training: bool = True,
                         use_gene_intersection: bool = False, label_key: str = "ClusterName"):

    # Path to data
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    path_replicate_2 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_2"
    paths = [path_replicate_1, path_replicate_2]
    path_ref = data_path / "Brain_Atlas_RNA_seq/l5_all.loom"

    # Load Xenium mouse brain replicates
    slides = load_replicates(paths)

    # Combine replicates in a unique adata
    annotated_data = slides[0].concatenate(slides[1:],
                                           batch_key="sample",
                                           uns_merge="unique",
                                           index_unique=None,
                                           )

    # Load scRNA-seq
    annotated_ref_seq = load_rna_seq_data(path_ref)

    if label_key == "leiden":
        annotated_ref_seq.obs["leiden"] = compute_ref_labels(annotated_ref_seq)

    # # Select Genes that are present in both ref and xenium data from the start
    # if use_gene_intersection:
    #     gene_intersection = set(annotated_data.var.index).intersection(annotated_ref_seq.var.index)
    #     is_in = [True if ensemble_id in gene_intersection else False
    #              for ensemble_id in annotated_ref_seq.var.index.tolist()]
    #     annotated_ref_seq = annotated_ref_seq[:, is_in]
    #     is_in = [True if ensemble_id in gene_intersection else False
    #              for ensemble_id in annotated_data.var.index.tolist()]
    #     annotated_data = annotated_data[:, is_in]

    # Examine QC metrics of Xenium data
    print("QC Metrics evaluation for replicates")
    # qc_metrics(annotated_data)

    # mitochondria-encoded (MT) genes should be removed for spatial mapping
    annotated_data.obsm['mt'] = annotated_data[:, annotated_data.var['mt'].values].X.toarray()
    annotated_data = annotated_data[:, ~annotated_data.var['mt'].values]

    # plot umap as Control for replicate
    print("UMAP for replicates")
    # plot_umap_samples(annotated_data)

    print(len(annotated_ref_seq.obs[label_key].unique()), annotated_ref_seq.obs[label_key].unique())
    # plot_umap_ref(annotated_ref_seq, cell_taxonomy=[label_key])

    # filter genes
    selected = filter_gene_index(annotated_ref_seq)
    annotated_ref_seq = annotated_ref_seq[:, selected].copy()

    if extract_signature:
        print("Running Cell Signature Extraction")
        mod = signature_ref(annotated_ref_seq, label=label_key, save_path=RESULTS_DIR_SIGNATURE)
    else:
        mod = cell2location.models.RegressionModel.load(str(RESULTS_DIR_SIGNATURE), annotated_ref_seq)

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    annotated_ref_seq = mod.export_posterior(
        annotated_ref_seq,
        sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': False},
    )

    # Save anndata object with results
    adata_file = f"{RESULTS_DIR_SIGNATURE}/sc.h5ad"
    annotated_ref_seq.write(adata_file)

    # Check issue with inference and noisy diagonal -> because corrected batch effect
    mod.plot_QC()
    plt.savefig(RESULTS_DIR_SIGNATURE / "QC_adata_ref.png")
    plt.close()

    # Reload the data with the define parameters
    adata_file = f"{RESULTS_DIR_SIGNATURE}/sc.h5ad"
    adata_ref = sc.read_h5ad(adata_file)

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                              for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                  for i in adata_ref.uns['mod']['factor_names']]].copy()

    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(annotated_data.var_names, inf_aver.index)
    annotated_data = annotated_data[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    if run_c2l_training:
        print("Running Cell2Location with determined Cell Signature")
        mod = run_cell2location(annotated_data, inf_aver, save_path=RESULTS_DIR_C2L)
    else:
        mod = cell2location.models.RegressionModel.load(str(RESULTS_DIR_C2L), annotated_data)

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    annotated_data = mod.export_posterior(
        annotated_data, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': False}
    )

    # Save anndata object with results
    adata_file = f"{RESULTS_DIR_C2L}/sp.h5ad"
    annotated_data.write(adata_file)

    mod.plot_QC()
    plt.savefig(RESULTS_DIR_C2L / "QC_spatial_mapping.png")
    plt.close()

    return 0


if "__main__" == __name__:

    os.makedirs(RESULTS_DIR, exist_ok=True)

    extract_signature_cell = True
    run_cell2location_training = True

    # Perform C2L on xenium data
    cell2location_xenium(extract_signature_cell, run_cell2location_training, label_key="leiden")
