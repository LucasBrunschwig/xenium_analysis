# Std
import os

# Third party
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cell2location
import scanpy as sc
import numpy as np

# relative
from utils import load_xenium_data, load_rna_seq_data


def transcripts_distribution(adata):

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.show()

    axs[0].set_title("Total Transcripts per cell")
    sns.histplot(adata.obs["total_counts"],
                 kde=False,
                 ax=axs[0])

    axs[1].set_title("Area of segmented cells")
    sns.histplot(
        adata.obs["cell_area"],
        kde=False,
        ax=axs[1],
    )

    axs[2].set_title("Nucleus ratio")
    sns.histplot(
        adata.obs["nucleus_area"] / adata.obs["cell_area"],
        kde=False,
        ax=axs[2],
    )

    fig.savefig(os.path.join("../scratch/lbrunsch/results", "basic_analysis.png"))


def cell2location_xenium(adata, aref):
    pass

# -------------------------------------------------------------------------------------------------------------------- #


if "__main__" == __name__:

    os.makedirs("../scratch/lbrunsch/results", exist_ok=True)

    # Spatial Transcriptomics from Xenium
    annotated_data = load_xenium_data(r"..\scratch\lbrunsch\data\Xenium_V1_FF_Mouse_Brain_MultiSection_1")

    # By cluster
    # annotated_ref_cluster = load_rna_seq_data(r"..\scratch\lbrunsch\data\Brain_Atlas_RNA_seq\l5_all.agg.loom")

    # scRNA-seq
    annotated_ref_seq = load_rna_seq_data(r"..\scratch\lbrunsch\data\Brain_Atlas_RNA_seq\l5_all.loom")

    print(len(annotated_ref_seq.obs["TaxonomyRank1"].unique()), annotated_ref_seq.obs["TaxonomyRank1"].unique())
    print(len(annotated_ref_seq.obs["TaxonomyRank3"].unique()), annotated_ref_seq.obs["TaxonomyRank3"].unique())

    from cell2location.utils.filtering import filter_genes

    # Filtering requires unique index
    annotated_ref_seq.var["SYMBOL"] = annotated_ref_seq.var.index
    annotated_ref_seq.var.set_index("Accession", drop=True, inplace=True)

    selected = filter_genes(annotated_ref_seq, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)

    # filter the object
    annotated_ref_seq = annotated_ref_seq[:, selected].copy()

    # ---------------------------------------------------------------------------------------------------------------- #
    # Signature Reference Estimation

    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=annotated_ref_seq,
                                                       # 10X reaction / sample / batch
                                                       batch_key='SampleID',
                                                       # cell type, covariate used for constructing signatures
                                                       labels_key='TaxonomyRank3',
                                                       # multiplicative technical effects (platform, 3' vs 5', donor effect)
                                                       # categorical_covariate_keys=["DonorID"]
                                                       )

    from cell2location.models import RegressionModel

    mod = RegressionModel(annotated_ref_seq)

    # view anndata_setup as a sanity check
    mod.view_anndata_setup()

    mod.train(max_epochs=10, use_gpu=True)

    mod.plot_history(5)

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    annotated_ref_seq = mod.export_posterior(
        annotated_ref_seq, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': False}
    )

    ref_run_name = "results"
    os.makedirs(ref_run_name, exist_ok=True)

    # Save model
    mod.save(f"{ref_run_name}", overwrite=True)

    # Save anndata object with results
    adata_file = f"{ref_run_name}/sc.h5ad"
    annotated_ref_seq.write(adata_file)

    annotated_ref_seq = mod.export_posterior(
        annotated_ref_seq, use_quantiles=True,
        # choose quantiles
        add_to_varm=["q05", "q50", "q95", "q0001"],
        sample_kwargs={'batch_size': 2500}
    )

    # Check issue with inference and noisy diagonal -> because corrected batch effect
    mod.plot_QC()

    adata_file = f"{ref_run_name}/sc.h5ad"
    adata_ref = sc.read_h5ad(adata_file)
    mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", adata_ref)

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                              for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                  for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    # ---------------------------------------------------------------------------------------------------------------- #
    # Spatial Mapping

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(annotated_data.var_names, inf_aver.index)
    adata_vis = annotated_data[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

    # create and train the model
    mod = cell2location.models.Cell2location(
        adata_vis, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=1,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()

    mod.train(max_epochs=30000,
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
              use_gpu=True,
              )

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    mod.plot_history(1000)
    plt.legend(labels=['full data training'])

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_vis = mod.export_posterior(
        adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
    )

    # Save model
    run_name = "results_c2l"
    os.makedirs(run_name, exist_ok=True)
    mod.save(f"{run_name}", overwrite=True)

    # mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

    # Save anndata object with results
    adata_file = f"{run_name}/sp.h5ad"
    adata_vis.write(adata_file)

    mod.plot_QC()

    # add 5% quantile, representing confident cell abundance, 'at least this amount is present',
    # to adata.obs with nice names for plotting
    adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

    # select one slide
    from cell2location.utils import select_slide

    slide = select_slide(adata_vis, 'V1_Human_Lymph_Node')

    # plot in spatial coordinates
    with mpl.rc_context({'axes.facecolor': 'black',
                         'figure.figsize': [4.5, 5]}):

        sc.pl.spatial(slide, cmap='magma',
                      # show first 8 cell types
                      color=['B_Cycling', 'B_GC_LZ', 'T_CD4+_TfH_GC', 'FDC',
                             'B_naive', 'T_CD4+_naive', 'B_plasma', 'Endo'],
                      ncols=4, size=1.3,
                      img_key='hires',
                      # limit color scale at 99.2% quantile of cell abundance
                      vmin=0, vmax='p99.2'
                      )

    # Plot Annotated Data
    # transcripts_distribution(annotated_data)


