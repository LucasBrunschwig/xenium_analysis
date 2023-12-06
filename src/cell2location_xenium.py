# Std
import os
from pathlib import Path

# Third party
import matplotlib.pyplot as plt
import pandas as pd
import scanpy
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import seaborn as sns
import cell2location
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
import scanpy as sc
import numpy as np
import squidpy as sq
from scipy.sparse import csr_matrix
import logging

# relative
from utils import load_xenium_data, load_rna_seq_data, preprocess_transcriptomics
from leiden_clustering import compute_ref_labels
from visualization import visualize

RESULTS_DIR = Path()
RESULTS_DIR_SIGNATURE = Path()
RESULTS_DIR_C2L = Path()


def load_replicates(paths: list):
    adata_list = []
    for path in paths:
        adata = load_xenium_data(path)
        adata.obs['sample'] = str(path).split(os.sep)[-1]
        adata.X = csr_matrix(adata.X)

        # add sample name to obs names
        adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
        adata.obs_names = adata.obs["sample"] + '_' + adata.obs_names
        adata.obs.index.name = 'spot_id'

        adata_list.append(adata)

    return adata_list


def qc_metrics(adata: scanpy.AnnData, save_path: Path = RESULTS_DIR):
    """ This function calculates QC metrics, plot results and save figures.

    This function expect the obs dataframe to have a column "sample" affecting each data point to a sample.

    :param adata: AnnData object containing transcriptomics data
    :param save_path: expect a path to save figures
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

        plt.savefig(save_path / f"spatial_observation_{sample}.png")
        plt.close()


def plot_umap_samples(adata_vis: scanpy.AnnData, save_path: Path = RESULTS_DIR):
    """
    Compute umap and plot/save a superposition of replicates for visual comparison:
    - The obs dataframe of adata_vis should contain a column called "sample" containing sample label.
    - This function expect raw transcriptomics data.

    Parameters
    ----------
    :param adata_vis: AnnData containing transcriptomics replicates
    :param save_path: expect a path to save figures


    """
    adata_vis_plt = adata_vis.copy()

    # log(p + 1)
    sc.pp.log1p(adata_vis_plt)

    # Scale the data ( (data - mean) / sd )
    sc.pp.scale(adata_vis_plt, max_value=10)

    # PCA, KNN construction, UMAP
    sc.tl.pca(adata_vis_plt, svd_solver='arpack', n_comps=40)
    sc.pp.neighbors(adata_vis_plt, n_neighbors=20, n_pcs=40, metric='cosine')
    sc.tl.umap(adata_vis_plt, min_dist=0.3, spread=1)

    # Plot and save figure
    with mpl.rc_context({'figure.figsize': [8, 8],
                         'axes.facecolor': 'white'}):
        sc.pl.umap(adata_vis_plt, color=['sample'], size=2,
                   color_map='RdPu', ncols=1,
                   legend_fontsize=10)
        plt.savefig(save_path / "umap_samples.png")
        plt.close()


def plot_umap_ref(adata: scanpy.AnnData, cell_taxonomy: list):
    """

    Parameters
    ----------
    :param adata:
    :param cell_taxonomy:

    Returns
    -------

    """
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


def filter_gene_index(adata, cell_cutoff: int = 5, cell_cutoff2: float = 0.03, nonz_mean_cutoff: float = 1.12):
    return filter_genes(adata,
                        cell_count_cutoff=cell_cutoff,
                        cell_percentage_cutoff2=cell_cutoff2,
                        nonz_mean_cutoff=nonz_mean_cutoff)


def signature_ref(annotated_ref_seq, label: str, save_path: Path):
    """

    Parameters
    ----------
    annotated_ref_seq: AnnData object containing transcriptomics data
    label: the column name representing all dimensions
    save_path: the path to save the model

    Returns
    -------

    """

    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=annotated_ref_seq,
                                                       # 10X reaction / sample / batch
                                                       batch_key='SampleID',  # DateCaptured / SampleID is similar
                                                       # cell type, co-variate used for constructing signatures
                                                       labels_key=label,
                                                       # multiplicative technical effects (platform, 3' - 5', donor)
                                                       # categorical_covariate_keys=["Age",
                                                       #                             "AnalysisPool",
                                                       #                             "Q30 Bases in Barcode"
                                                       #                             "Q30 Bases in RNA Read"
                                                       #                             "ChipID",
                                                       #                             "Flowcell
                                                       #                            ]
                                                       )

    mod = RegressionModel(annotated_ref_seq)

    # view anndata_setup as a sanity check
    mod.view_anndata_setup()

    # train the probabilistic model
    mod.train(max_epochs=5000, use_gpu=True)

    # Plot History and save figure
    mod.plot_history(10)
    plt.savefig(save_path / "adata_signature_training.png")
    plt.close()

    # Save model and anndata
    mod.save(f"{save_path}", overwrite=True)

    return mod


def run_cell2location(adata_vis, inf_aver, save_path: Path, n_training_: int):
    """

    Parameters
    ----------
    adata_vis
    inf_aver
    save_path
    n_training_

    Returns
    -------

    """

    sc.settings.set_figure_params(dpi=100, color_map='viridis', dpi_save=100,
                                  vector_friendly=True,
                                  facecolor='white')

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

    # create and train the model
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        N_cells_per_location=1,
        detection_alpha=20
    )
    mod.view_anndata_setup()

    mod.train(max_epochs=n_training_,
              batch_size=None,
              train_size=1,
              use_gpu=True,
              )

    # plot loss history
    mod.plot_history(50)
    plt.legend(labels=['full data training'])
    plt.savefig(save_path / "plot_history_c2l.png")
    plt.close()

    # save model and transcriptomic data
    mod.save(f"{save_path}", overwrite=True)

    return mod


def run_cell2location_xenium(run_qc_plots_: bool = True, run_extract_signature_: bool = True,
                             run_c2l_training_: bool = True, n_training_: int = 10000,
                             label_key_: str = "ClusterName", n_comp_: int = 50,
                             n_neighbors_: int = 13, subset_: bool = False):
    # Path to data
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    path_replicate_2 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_2"
    path_replicate_3 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_3"
    paths = [path_replicate_1]  # path_replicate_2, path_replicate_3]
    path_ref = data_path / "Brain_Atlas_RNA_seq/l5_all.loom"
    path_ref_agg = data_path / "Brain_Atlas_RNA_seq/l5_all.agg.loom"

    # Load Xenium mouse brain replicates
    slides = load_replicates(paths)

    # Combine replicates in a unique adata
    annotated_data = slides[0].concatenate(slides[1:],
                                           batch_key="sample",
                                           uns_merge="unique",
                                           index_unique=None,
                                           )

    if label_key_ != "AtlasAggregate":
        # Load scRNA-seq
        annotated_ref_seq = load_rna_seq_data(path_ref)

        if subset_:
            intersect = np.intersect1d(annotated_data.var_names, annotated_ref_seq.var_names)
            annotated_ref_seq = annotated_ref_seq[:, intersect]

    # Examine QC metrics of Xenium data
    if run_qc_plots_:
        print("QC Metrics evaluation for replicates")
        qc_metrics(annotated_data)

        # plot umap as Control for replicate
        print("UMAP for replicates")
        plot_umap_samples(annotated_data)

        if label_key_ not in ["leiden", "AtlasAggregate"]:
            print(len(annotated_ref_seq.obs[label_key_].unique()), annotated_ref_seq.obs[label_key_].unique())
            plot_umap_ref(annotated_ref_seq, cell_taxonomy=[label_key_])

    # mitochondria-encoded (MT) genes should be removed for spatial mapping
    annotated_data.obsm['mt'] = annotated_data[:, annotated_data.var['mt'].values].X.toarray()
    annotated_data = annotated_data[:, ~annotated_data.var['mt'].values]

    if label_key_ != "AtlasAggregate":

        annotated_ref_seq.obsm['mt'] = annotated_ref_seq[:, annotated_ref_seq.var['mt'].values].X.toarray()
        annotated_ref_seq = annotated_ref_seq[:, ~annotated_ref_seq.var['mt'].values]

        # filter genes
        if not subset_:
            selected = filter_gene_index(annotated_ref_seq)
            annotated_ref_seq = annotated_ref_seq[:, selected].copy()

        if label_key_ == "leiden":
            annotated_ref_seq_copy = annotated_ref_seq.copy()
            annotated_ref_seq_copy = preprocess_transcriptomics(annotated_ref_seq_copy, filter_=False)
            annotated_ref_seq.obs["leiden"] = compute_ref_labels(annotated_ref_seq_copy, n_neighbors_, n_comp_)

            print(len(annotated_ref_seq.obs[label_key_].unique()), annotated_ref_seq.obs[label_key_].unique())
            plot_umap_ref(annotated_ref_seq, cell_taxonomy=[label_key_])

    if run_extract_signature_ and label_key_ != "AtlasAggregate":
        print("Running Cell Signature Extraction")
        mod = signature_ref(annotated_ref_seq, label=label_key_, save_path=RESULTS_DIR_SIGNATURE)
    elif label_key_ == "AtlasAggregate":
        annotated_ref_agg = sc.read_loom(path_ref_agg)
        inf_aver = pd.DataFrame(annotated_ref_agg.X.toarray().T)
        inf_aver.columns = annotated_ref_agg.obs["ClusterName"]
        inf_aver.index = annotated_ref_agg.var["Accession"]
    else:
        mod = cell2location.models.RegressionModel.load(str(RESULTS_DIR_SIGNATURE), annotated_ref_seq)

    if label_key_ != "AtlasAggregate":
        # export the estimated cell abundance (summary of the posterior distribution).
        annotated_ref_seq = mod.export_posterior(
            annotated_ref_seq,
            sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True},
        )

        # Save anndata object with results
        adata_file = f"{RESULTS_DIR_SIGNATURE}/sc.h5ad"
        annotated_ref_seq.write(adata_file)

        # Check issue with inference and noisy diagonal -> because corrected batch effect
        mod.plot_QC()
        plt.savefig(RESULTS_DIR_SIGNATURE / "QC_adata_ref.png")
        plt.close()

        # Format signature expression for each cluster
        if 'means_per_cluster_mu_fg' in annotated_ref_seq.varm.keys():
            inf_aver = annotated_ref_seq.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                                          for i in annotated_ref_seq.uns['mod'][
                                                                              'factor_names']]].copy()
        else:
            inf_aver = annotated_ref_seq.var[[f'means_per_cluster_mu_fg_{i}'
                                              for i in annotated_ref_seq.uns['mod']['factor_names']]].copy()

        inf_aver.columns = annotated_ref_seq.uns['mod']['factor_names']

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(annotated_data.var_names, inf_aver.index)
    annotated_data = annotated_data[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    if run_c2l_training_:
        print("Running Cell2Location with determined Cell Signature")
        mod = run_cell2location(annotated_data, inf_aver, save_path=RESULTS_DIR_C2L, n_training_=n_training_)
    else:
        mod = cell2location.models.RegressionModel.load(str(RESULTS_DIR_C2L), annotated_data)

    # export the estimated cell abundance (summary of the posterior distribution).
    # for sample in annotated_data.obs["sample"].unique():
    #     adata_sample = annotated_data[annotated_data.obs['sample'].isin([sample]), :].copy()

    annotated_data = mod.export_posterior(
        annotated_data, sample_kwargs={'num_samples': 500, 'batch_size': len(annotated_data), 'use_gpu': True},
    )
    mod.plot_QC()
    plt.savefig(RESULTS_DIR_C2L / f"QC_spatial_mapping.png")
    plt.close()

    for sample in annotated_data.obs["sample"].unique():
        adata_sample = annotated_data[annotated_data.obs['sample'].isin([sample]), :].copy()

        adata_sample.obs["c2l_label"] = [cat.split("_")[-1] for cat in
                                         adata_sample.obsm["means_cell_abundance_w_sf"].idxmax(axis=1).tolist()]
        visualize(adata_sample, "c2l_label", savefig_path=RESULTS_DIR_C2L / f"{sample}_cluster_visualization.png")

        # Save anndata object with results
        adata_file = f"{RESULTS_DIR_C2L}/sp_{sample}.h5ad"
        adata_sample.write(adata_file)

    return 0


def build_results_dir(label_, n_neighbors_, n_comp_, subset_):
    subset_key = {True: "_subset", False: ""}
    # Declare Global Path
    global RESULTS_DIR
    RESULTS_DIR = Path(f"../../scratch/lbrunsch/results/cell2location{subset_key[subset_]}")

    if label_ == "leiden":
        RESULTS_DIR = RESULTS_DIR / f"_leiden_neigh{n_neighbors_}_pca{n_comp_}"
    else:
        RESULTS_DIR = RESULTS_DIR / f"_{label_}"

    global RESULTS_DIR_SIGNATURE
    RESULTS_DIR_SIGNATURE = RESULTS_DIR / "adata_ref"
    global RESULTS_DIR_C2L
    RESULTS_DIR_C2L = RESULTS_DIR / "cell2location"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR_C2L, exist_ok=True)
    os.makedirs(RESULTS_DIR_SIGNATURE, exist_ok=True)

    return RESULTS_DIR


if "__main__" == __name__:
    # Run different steps
    run_qc_plots = False
    extract_signature_cell = True
    run_cell2location_training = True

    # Training
    n_training = 1000

    # Select Labels ("leiden", "ClusterName", "AtlasAggregate)
    label_key = "ClusterName"

    # Specific to Leiden Clustering
    n_comp = 50
    n_neighbors = 13
    subset = False

    # Build directory results
    main_dir = build_results_dir(label_key, n_neighbors, n_comp, subset)

    logging.basicConfig(
        filename=str(main_dir / 'log_python.txt'), level=logging.INFO)

    # Perform C2L on xenium data
    run_cell2location_xenium(run_qc_plots, extract_signature_cell, run_cell2location_training, n_training_=n_training,
                             label_key_=label_key, n_comp_=n_comp, n_neighbors_=n_neighbors, subset_=subset)
