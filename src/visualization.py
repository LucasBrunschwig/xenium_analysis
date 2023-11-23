"""
File: visualization.py
Author: Lucas Brunschwig
Email: lucas.brunschwig@hotmail.fr
GitHub: @LucasBrunschwig

Description: This file implements visualization methods which represent the spatial representation of a xenium slice
             stored in a scanpy objects with the associated label names in the
"""

import squidpy as sq


def visualize_cortex_layers():
    raise ValueError("Not Implemented")


def visualize_hippocampus():
    raise ValueError("Not Implemented")


def visualize(adata, label_key):

    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        color=[
            label_key,
        ],
        shape=None,
        size=2,
        img=False,
        figsize=(15, 15)
    )
