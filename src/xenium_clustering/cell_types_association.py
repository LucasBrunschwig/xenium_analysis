"""
File: cell_types_association.py
Author: Lucas Brunschwig
Email: lucas.brunschwig@epfl.ch
GitHub: @LucasBrunschwig

Description: This file implements cell type association compared using leiden clustering, Cell2Location, others
"""

# Std
import os
from pathlib import Path

# Third Party
import scanpy as sc
import GraphST

# Relative
from utils import load_xenium_data

CELL_TYPE_ASSOCIATION = Path("../../scratch/lbrunsch/results/cell_type_association")
os.makedirs(CELL_TYPE_ASSOCIATION, exist_ok=True)


ATLAS = "mousebrain_atlas"
LEIDEN = "leiden_clustering"
LOUVAIN = "louvain_clustering"
MC_LUST = "mclust_clustering"


def cell_type_association(method_: str):
    """

    Returns
    -------

    """

    data_path = Path("../../../scratch/lbrunsch/data")

    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    replicate = load_xenium_data(path_replicate_1)

    GraphST.preprocess(replicate)
    GraphST.construct_interaction(replicate)
    GraphST.add_contrastive_label(replicate)

    if method_ == ATLAS:
        # Load the Reference cell atlas from mousebrain.org
        path_rna_ref_aggregate = data_path / "Brain_Atlas_RNA_seq/l5_all.agg.loom"
        adata_ref_clusters = sc.read_loom(path_rna_ref_aggregate)
        adata_ref_clusters.var_names_make_unique()

    elif method_ == LEIDEN:
        raise ValueError("Not Implemented")

    return 0


if __name__ == "__main__":

    method = "mousebrain_atlas"

    cell_type_association(method)