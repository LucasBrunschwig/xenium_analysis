import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import cell2location

if __name__ == "__main__":

    results_folder = './results/lymph_nodes_analysis/'

    # create paths and names to results folders for reference regression and cell2location models
    ref_run_name = f'{results_folder}/reference_signatures'
    run_name = f'{results_folder}/cell2location_map'

    adata_vis = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
    adata_vis.obs['sample'] = list(adata_vis.uns['spatial'].keys())[0]

    adata_vis.var['SYMBOL'] = adata_vis.var_names
    adata_vis.var.set_index('gene_ids', drop=True, inplace=True)

    sc.pl.spatial(adata_vis, color='PTPRC', gene_symbols='SYMBOL')

    # find mitochondria-encoded (MT) genes
    adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]

    # remove MT genes for spatial mapping (keeping their counts in the object)
    adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
    adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]

    # Read data
    adata_ref = sc.read(
        f'../scratch/lbrunsch/sc.h5ad',
        backup_url='https://cell2location.cog.sanger.ac.uk/paper/integrated_lymphoid_organ_scrna/RegressionNBV4Torch_57covariates_73260cells_10237genes/sc.h5ad'
    )

    adata_ref.var['SYMBOL'] = adata_ref.var.index
    # rename 'GeneID-2' as necessary for your data
    adata_ref.var.set_index('GeneID-2', drop=True, inplace=True)

    # delete unnecessary raw slot (to be removed in a future version of the tutorial)
    del adata_ref.raw

    # Filtering of the genes with various parameters
    # cell_count_cutoff < 5, cell_percentage_cutoff < 0.03
    from cell2location.utils.filtering import filter_genes

    selected = filter_genes(adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)

    # filter the object
    adata_ref = adata_ref[:, selected].copy()

    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                                                       # 10X reaction / sample / batch
                                                       batch_key='Sample',
                                                       # cell type, covariate used for constructing signatures
                                                       labels_key='Subset',
                                                       # multiplicative technical effects (platform, 3' vs 5', donor effect)
                                                       categorical_covariate_keys=['Method']
                                                       )

    # create the regression model
    from cell2location.models import RegressionModel

    mod = RegressionModel(adata_ref)

    # view anndata_setup as a sanity check
    mod.view_anndata_setup()

    mod.train(max_epochs=250, use_gpu=True)

    mod.plot_history(20)

