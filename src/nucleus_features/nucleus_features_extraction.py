"""
Description: This python file has the purpose to extract nuclei features based on nuclear seg. The goal is to use the
             segmentation information provided by different tools such as CellPose, Stardist, CellProfiler and
             compute different features to see how one  can classify nucleus based on nuclear morphology features.

Author: Lucas Brunschwig (lucas.brunschwig@epfl.ch)

Methods:
    stardist_qupath: These masks were extracted from the QuPath Stardist extension.
        - pretrained: weights = he_heavy_model.pb
        - image: aligned image_he (obtained through image registration directory)
        - normalize(1, 99), pixelSize(0.4), Prob(0.5)
    stardist:
        - TBD:

Remarks:
    - Stardist in QuPath performs normalization per tiles


Development:
[ x ](1)Choose an algorithm and a type of image to extract the nucleus information

[ x ](2) Extract Masks from QuPath to a usable version in Python

[ x ](3) Extract nucleus individually for analysis

[ x ](4) Create adata objects from masks and transcripts

[ x ](5) Visualization and Comparison between Masks

[ x ](6) Transcriptomics Clustering

"""

# Std
from pathlib import Path
import os
import heapq
import json
import random

# Third party
import cv2
import numpy as np
import pandas as pd
import shapely.geometry
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy import stats
from shapely import Point
import geopandas as gpd
from collections import OrderedDict
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D

# Relative import
from src.utils import (get_results_path, get_human_breast_dapi_aligned_path, load_xenium_he_ome_tiff,
                       load_xenium_masks, load_xenium_transcriptomics, get_human_breast_he_path, load_geojson_masks,
                       preprocess_transcriptomics, get_human_breast_he_aligned_path, get_geojson_masks)

# -------------------------------------------------------------------------------------------------------------------- #
# HELPER METHODS


def compute_fos(flattened_signal) -> tuple:

    features = [np.mean(flattened_signal),
                np.median(flattened_signal),
                np.std(flattened_signal),
                stats.skew(flattened_signal),
                stats.kurtosis(flattened_signal),
                min(flattened_signal),
                max(flattened_signal),
                np.percentile(flattened_signal, 10),
                np.percentile(flattened_signal, 25),
                np.percentile(flattened_signal, 75),
                np.percentile(flattened_signal, 90)]

    features_name = ["mean", "median", "std", "skewness", "kurtosis", "min", "max", "q10", "q25", "q75", "q90"]

    return features, features_name


def get_cell_area(dapi_signal):

    cell_per_micron = 0.2125
    return len(dapi_signal) * (cell_per_micron ** 2)


def visualize_umap_pca(nucleus_features, label, save_dir_, img_type_):
    cmap = plt.get_cmap("tab20")

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=5, label=label)
                       for i, label in enumerate(np.sort(np.unique(label)))]

    # Visualize U-map with different neighboring values
    nucleus_features_normalized = (nucleus_features - np.mean(nucleus_features, axis=0)) / np.std(nucleus_features, axis=0)
    for neigh in [5, 15, 25, 35, 50]:
        embedding = umap.UMAP(n_neighbors=neigh, n_components=2).fit_transform(nucleus_features_normalized)
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(label), s=1)
        plt.xlabel("umap-1")
        plt.ylabel("umap-2")
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
                   title_fontsize="large")
        plt.title(f"U-map (n={neigh}, {img_type_})")
        plt.tight_layout()
        plt.savefig(save_dir_ / f"umap_fos_{img_type_}_{neigh}.png")
        plt.close()
        min_x = min(embedding[:, 0])
        max_x = max(embedding[:, 0])
        min_y = min(embedding[:, 1])
        max_y = max(embedding[:, 1])
        # Visualize decomposition cluster by cluster
        for sub_label in np.unique(label):
            sub_label_list = [ix for ix, label_ in enumerate(label) if label_ == sub_label]
            plt.figure()
            plt.scatter(embedding[sub_label_list, 0], embedding[sub_label_list, 1], c=cmap([sub_label] * len(sub_label_list)), s=1)
            plt.xlim(min_x - 1, max_x + 1)
            plt.ylim(min_y - 1, max_y + 1)
            plt.xlabel("umap-1")
            plt.ylabel("umap-2")
            plt.title(f"U-map (n={neigh}, {img_type_}) - sub_label - {sub_label}")
            plt.tight_layout()
            plt.savefig(save_dir_ / f"umap_{sub_label}_fos_{img_type_}_{neigh}.png")
            plt.close()

    embedding = PCA(n_components=2).fit_transform(nucleus_features_normalized)
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(label), s=1)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
               title_fontsize="large")
    plt.title(f"PCA ({img_type_})")
    plt.tight_layout()
    plt.savefig(save_dir_ / f"pca_fos_{img_type_}_{neigh}.png")
    plt.close()
    min_x = min(embedding[:, 0])
    max_x = max(embedding[:, 0])
    min_y = min(embedding[:, 1])
    max_y = max(embedding[:, 1])
    # Visualize decomposition cluster by cluster
    for sub_label in np.unique(label):
        sub_label_list = [ix for ix, label_ in enumerate(label) if label_ == sub_label]
        plt.figure()
        plt.scatter(embedding[sub_label_list, 0], embedding[sub_label_list, 1], c=cmap([sub_label]*len(sub_label_list)), s=1)
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(min_y - 1, max_y + 1)
        plt.xlabel("umap-1")
        plt.ylabel("umap-2")
        plt.title(f"PCA ({img_type_}) - sub_label - {sub_label}")
        plt.tight_layout()
        plt.savefig(save_dir_ / f"pca_{sub_label}_fos_{img_type_}.png")
        plt.close()


def compute_ref_labels(adata, n_comp: int = 50, n_neighbors: int = 13):
    adata_ = adata.copy()
    sc.pp.pca(adata_, n_comps=n_comp)
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors)  # necessary for UMAP (k-neighbors with weights)
    sc.tl.umap(adata_)
    sc.tl.leiden(adata_)

    return adata_


def convert_masks_to_df(masks_, cell_id_):
    """

    :param masks_: a collection of polygon
    :param cell_id_: the list of cell ids
    :return: a dataframe containing vertex x vertex y and cell_id
    """

    df_dict = {"cell_id": [], "vertex_x": [], "vertex_y": []}
    for i, mask in enumerate(masks_.polygon):
        id_ = masks_.iloc[i].cell_id
        if id_ in cell_id_:
            pts = np.array(mask.exterior.coords, np.int32)
            for pt in pts:
                df_dict["cell_id"].append(id_)
                df_dict["vertex_x"].append(pt[0])
                df_dict["vertex_y"].append(pt[1])

    return pd.DataFrame.from_dict(df_dict)


def create_masks_image(masks_outline, width, height):
    """ Takes a collection of polygon, and draws corresponding masks with
        width and height.

        Remark:
            - the number of polygons at the end is 99'000 whereas the total number of mask 128'000
            - this would mean a lot of masks are contained with one another which makes no sense

    :param masks_outline:
    :param width:
    :param height:
    :return:
    """

    mask = np.zeros((width, height), dtype=np.int32)
    for i, polygon in enumerate(masks_outline["polygon"]):
        mask = draw_polygon_on_mask_cv2(polygon, mask, i+1)

    return mask


def draw_polygon_on_mask_cv2(polygon, mask, label):
    """ Draw a mask with corresponding label on an array based on a Polygon from shapely
    """
    pts = np.array(polygon.exterior.coords, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Use cv2.fillPoly to fill the polygon area with white (1)
    cv2.fillPoly(mask, [pts], color=label)
    return mask


def get_polygon_indices(masks_outline):
    indices = []
    for polygon in masks_outline["polygon"]:
        minx, miny, maxx, maxy = polygon.bounds

        x = np.arange(int(minx), int(maxx) + 1)
        y = np.arange(int(miny), int(maxy) + 1)
        xx, yy = np.meshgrid(x, y)
        points_inside_polygon = np.vstack((yy.flatten(), xx.flatten())).T
        indices.append(points_inside_polygon)
    return indices


def load_geojson_background(path: Path):
    """ Create a MultiPolygon shapely to remove background mask"""
    with open(path, "r") as file:
        dict_ = json.load(file)
    multipolygon = shapely.geometry.shape(dict_["features"][0]["geometry"])
    return multipolygon


def visualize_masks(ix, save_file, dapi_img, he_img, spatial_index_, spatial_index_target,
                    gdf_polygons_, gdf_polygons_target):
    try:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

        polygon = gdf_polygons_[gdf_polygons_.cell_id == ix].polygon.tolist()[0]
        area = gdf_polygons_[gdf_polygons_.cell_id == ix].nucleus_area.tolist()[0]

        bounds = np.array([bound - 10 if i < 2 else bound + 10 for i, bound in enumerate(polygon.bounds)])

        [ax.plot(np.array(polygon.exterior.coords)[:, 0], np.array(polygon.exterior.coords)[:, 1], 'b',
                 zorder=2, label="custom") for ax in axs]

        nearby_index = list(spatial_index_.intersection(bounds))
        masks_near = gdf_polygons_.iloc[nearby_index]

        for polygon_ in masks_near.polygon:
            [ax.plot(np.array(polygon_.exterior.coords)[:, 0], np.array(polygon_.exterior.coords)[:, 1], 'g',
                     zorder=2, label="xenium") for ax in axs]
            np.vstack((bounds, polygon_.bounds))

        nearby_index = list(spatial_index_target.intersection(bounds))
        masks_near_target = gdf_polygons_target.iloc[nearby_index]

        for polygon_ in masks_near_target.polygon:
            [ax.plot(np.array(polygon_.exterior.coords)[:, 0], np.array(polygon_.exterior.coords)[:, 1], 'g',
                     zorder=2, label="xenium") for ax in axs]
            np.vstack((bounds, polygon_.bounds))

        if len(bounds) == 1:
            bounds_min = bounds[[0, 1]].astype(int)
            bounds_max = bounds[[2, 3]].astype(int)
        else:
            bounds_min = np.min(bounds[[0, 1], :]).astype(int)
            bounds_max = np.max(bounds[[2, 3], :]).astype(int)

        axs[0].imshow(dapi_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")
        axs[1].imshow(he_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))  #
        plt.tight_layout()
        plt.savefig(str(save_file)[:-3] + f"{area}.png")
        plt.close()
    except:
        pass


# -------------------------------------------------------------------------------------------------------------------- #
# TEST METHODS


def test_masks_area_distribution(masks_xenium_: pd.DataFrame, masks_custom_: pd.DataFrame, results_dir_: Path):

    results_dir_ = results_dir_ / "test_masks_area_distribution"
    os.makedirs(results_dir_, exist_ok=True)

    xenium_areas = []
    for polygon in masks_xenium_["polygon"]:
        xenium_areas.append(polygon.area * 0.2125**2)
    print("Max Area (xenium):", np.sort(xenium_areas)[-10:])

    custom_areas = []
    for polygon in masks_custom_["polygon"]:
        if not polygon.area * 0.2125 > 2600:
            custom_areas.append(polygon.area * 0.2125**2)
        else:
            print("Giant Polygons:", polygon)

    print("Max Area (stardist):", np.sort(custom_areas)[-10:])

    plt.figure()
    plt.hist(xenium_areas, bins=100, label="Xenium DAPI", density=True, fill=False, ec="b")
    plt.hist(custom_areas, bins=100, label="Stardist H&E", density=True, fill=False, ec="r")
    plt.xlabel("nucleus area [um^2]")
    plt.ylabel("density [a.u.]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir_ / "nucleus_areas_histogram.png")


def test_masks_transcripts_difference(adata_xenium_, adata_custom_, results_dir_, segmentation_method_):

    # Specific subdir
    results_dir_ = results_dir_ / "masks_transcripts"
    os.makedirs(results_dir_, exist_ok=True)

    # Filter
    filter_index = np.argsort(adata_custom_.obs["nucleus_area"])[-1]
    adata_custom_ = adata_custom_[~adata_custom_.obs.index.astype(int).isin([filter_index])]

    results_dir_ = results_dir_ / segmentation_method_
    os.makedirs(results_dir_, exist_ok=True)

    transcripts_nucleus_counts_xenium = np.sum(adata_xenium_.X, axis=1)
    transcripts_nucleus_counts_custom = np.sum(adata_custom_.X, axis=1)

    transcripts_nucleus_counts_xenium = np.array(transcripts_nucleus_counts_xenium)
    print("Total Count For Xenium:", np.sum(transcripts_nucleus_counts_xenium))
    print("Mean Count For Xenium:", np.mean(transcripts_nucleus_counts_xenium))
    print("Median Count For Xenium:", np.median(transcripts_nucleus_counts_xenium))
    print("Std Count For Xenium:", np.std(transcripts_nucleus_counts_xenium))
    print("Number of 0 counts For Xenium:", len(np.where(transcripts_nucleus_counts_xenium == 0)[0]))
    print()
    transcripts_nucleus_counts_custom = np.array(transcripts_nucleus_counts_custom)
    print("Total Count For Custom:", np.sum(transcripts_nucleus_counts_custom))
    print("Mean Count For Custom:", np.mean(transcripts_nucleus_counts_custom))
    print("Median Count For Custom:", np.median(transcripts_nucleus_counts_custom))
    print("Std Count For Custom:", np.std(transcripts_nucleus_counts_custom))
    print("Number of 0 counts For Custom:", len(np.where(transcripts_nucleus_counts_custom == 0)[0]))

    plt.figure()
    bins = plt.hist(transcripts_nucleus_counts_custom, bins=100, label="Custom H&E", density=True, fill=False, ec="b")
    plt.hist(transcripts_nucleus_counts_xenium, bins=bins[1], label="Xenium DAPI", density=True, fill=False, ec="r")
    plt.xlabel("Number of Transcripts per nucleus [a.u.]")
    plt.ylabel("density [a.u.]")
    plt.legend()
    plt.tight_layout()
    plt.title("Transcripts per nucleus comparison")
    plt.savefig(results_dir_ / f"transcripts_nucleus.png")
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    max_y = np.max((np.max(transcripts_nucleus_counts_custom), np.max(transcripts_nucleus_counts_xenium))).astype(int)
    max_x = np.max((np.max(adata_custom_.obs["nucleus_area"]), np.max(adata_xenium_.obs["nucleus_area"]))).astype(int)

    axs[0].scatter(adata_xenium_.obs["nucleus_area"], transcripts_nucleus_counts_xenium, s=1)
    axs[0].set_xlabel("nucleus area [um^2]")
    axs[0].set_ylabel("number of transcripts")
    axs[0].set_title("10X Segmentation")
    axs[0].set_xlim(0, max_x)
    axs[0].set_ylim(0, max_y)

    axs[1].scatter(adata_custom_.obs["nucleus_area"], transcripts_nucleus_counts_custom, s=1)
    axs[1].set_ylim(0, max_y)
    axs[1].set_xlim(0, max_x)
    axs[1].set_xlabel("nucleus area [um^2]")
    axs[1].set_ylabel("number of transcripts")
    axs[1].set_title("Custom Segmentation")

    plt.tight_layout()
    plt.savefig(results_dir_ / f"transcripts_vs_area.png")
    plt.close()

    # Visualize the cell area with less than 10 transcripts
    ix_10_xenium = np.where(transcripts_nucleus_counts_xenium <= 10)
    ix_10_custom = np.where(transcripts_nucleus_counts_custom <= 10)
    area_10_xenium = adata_xenium_.obs["nucleus_area"].iloc[ix_10_xenium]
    area_10_custom = adata_custom_.obs["nucleus_area"].iloc[ix_10_custom]

    # Combine the data into a single DataFrame for plotting
    data = pd.DataFrame({
        'Nucleus Area': np.concatenate([area_10_xenium, area_10_custom]),
        'Dataset': ['Xenium'] * len(area_10_xenium) + ['Custom'] * len(area_10_custom)
    })

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Dataset', y='Nucleus Area', data=data)
    plt.title('Nucleus Area Distribution for Nuclei with <= 10 Transcripts')
    plt.xlabel('Custom and Xenium segmentation (<10 transcripts)')
    plt.ylabel('nucleus area [um^2]')
    plt.tight_layout()
    plt.savefig(results_dir_ / f"transcripts_10_vs_area.png")
    plt.close()

    ix_g10_xenium = np.where(transcripts_nucleus_counts_xenium > 10)
    ix_g10_custom = np.where(transcripts_nucleus_counts_custom > 10)
    area_g10_xenium = adata_xenium_.obs["nucleus_area"].iloc[ix_g10_xenium]
    area_g10_custom = adata_custom_.obs["nucleus_area"].iloc[ix_g10_custom]
    # Gene Distribution / Number of time a gene is detected inside a nucleus

    # Combine the data into a single DataFrame for plotting
    data = pd.DataFrame({
        'Nucleus Area': np.concatenate([area_g10_xenium, area_g10_custom]),
        'Dataset': ['Xenium'] * len(area_g10_xenium) + ['Custom'] * len(area_g10_custom)
    })

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Dataset', y='Nucleus Area', data=data)
    plt.title('Nucleus Area Distribution for Nuclei with > 10 Transcripts')
    plt.xlabel('Custom and Xenium segmentation (>10 transcripts)')
    plt.ylabel('nucleus area [um^2]')
    plt.tight_layout()
    plt.savefig(results_dir_ / f"transcripts_g10_vs_area.png")
    plt.close()

    # Bin the nucleus area in 10 bins and show transcripts distribution
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 9))
    axs[0].xaxis.set_tick_params(labelsize=8)
    axs[1].xaxis.set_tick_params(labelsize=8)
    series, bins = pd.cut(adata_custom_.obs["nucleus_area"], 10, retbins=True)
    adata_custom_.obs["area_bins"] = series
    sns.violinplot(ax=axs[0], x='area_bins', y='transcripts_count', data=adata_custom_.obs)
    adata_xenium_.obs["area_bins"] = pd.cut(adata_xenium_.obs["nucleus_area"], bins=bins, include_lowest=True,
                                            labels=series.cat.categories)
    axs[0].set_title("Custom")

    sns.violinplot(ax=axs[1], x='area_bins', y='transcripts_count', data=adata_xenium_.obs)
    axs[1].set_title("10X Genomics")
    fig.savefig(results_dir_ / f"area_bins_vs_transcripts.png")

    # Plot the number of unique genes
    plt.figure(figsize=(10, 5))
    matrix = adata_custom_.X
    matrix[np.where(matrix > 1.0)] = 1.0
    number_of_genes_custom = np.sum(matrix, axis=1)
    matrix = adata_xenium_.X
    matrix[np.where(matrix > 1.0)] = 1.0
    number_of_genes_xenium = np.sum(matrix, axis=1)

    adata_xenium_.obs["unique_genes"] = number_of_genes_xenium
    adata_custom_.obs["unique_genes"] = number_of_genes_custom

    plt.hist(number_of_genes_custom, bins=np.max(number_of_genes_custom).astype(int), density=True, fill=False, ec='r',
             label="custom")
    plt.hist(number_of_genes_xenium, bins=np.max(number_of_genes_custom).astype(int), density=True, fill=False, ec='b',
             label="xenium")
    plt.xlabel("number of unique genes")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir_ / f"number_of_genes.png")

    # Bin the nucleus area in 10 bins and show transcripts distribution
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 9))
    axs[0].xaxis.set_tick_params(labelsize=8)
    axs[1].xaxis.set_tick_params(labelsize=8)
    sns.violinplot(ax=axs[0], x='area_bins', y='unique_genes', data=adata_custom_.obs)
    adata_xenium_.obs["area_bins"] = pd.cut(adata_xenium_.obs["nucleus_area"], bins=bins, include_lowest=True,
                                            labels=series.cat.categories)
    axs[0].set_title("Custom")

    sns.violinplot(ax=axs[1], x='area_bins', y='unique_genes', data=adata_xenium_.obs)
    axs[1].set_title("10X Genomics")
    fig.savefig(results_dir_ / f"unique_genes_vs_area_bins.png")

    return transcripts_nucleus_counts_xenium, transcripts_nucleus_counts_custom


def test_masks_overlap(masks_xenium_, masks_custom_, dapi_img, he_img, results_dir_, segmentation_method_):
    """ Goal analyze the difference in overlap

    :param masks_xenium_: masks from default Xenium
    :param masks_custom_: masks from custom method
    :return: 0
    """

    # Step 1: check the overlap between nuclei:
    #   - how many nuclei are segmented compared to one another
    #       - how many nuclei are contained within another nuclei
    #       - is there many cases where custom has two masks in one
    #       - check over-segmentation by testing number of nuclei contained in one nucleus
    #       - for both if nuclei are neither contained in another nor matched with another -> not segmented
    #   - are there regions of nuclei that are segmented differently ?
    #       - plot not shared nuclei centroid on an image

    visualization_dir = results_dir_ / segmentation_method_ / "visualize"
    os.makedirs(visualization_dir, exist_ok=True)

    gdf_polygons_custom = gpd.GeoDataFrame(masks_custom_, geometry=masks_custom_.polygon)
    gdf_polygons_xenium = gpd.GeoDataFrame(masks_xenium_, geometry=masks_xenium_.polygon)
    spatial_index_custom = gdf_polygons_custom.sindex
    spatial_index_xenium = gdf_polygons_xenium.sindex

    # Initiate list
    contained_in_xenium = {i: 0 for i in range(len(masks_xenium_))}
    contained_in_custom = {i: 0 for i in range(len(masks_custom_))}
    matched = 0
    overlap = 0
    matched_xenium = {i: 0 for i in range(len(masks_xenium_))}
    matched_custom = {i: 0 for i in range(len(masks_custom_))}
    matched_pairs = []
    overlap_pairs = []
    iou_distribution = []
    intersect_xenium_index = []
    intersect_custom_index = []

    big_cell_id = masks_custom_[masks_custom_.nucleus_area > 256 / (0.2125**2)].cell_id.tolist()
    big_cell = visualization_dir / "big_custom_nucleus"
    os.makedirs(big_cell, exist_ok=True)
    for id_ in big_cell_id:
        visualize_masks(id_, big_cell / f"example{id_}_.png",  masks_custom_, dapi_img, he_img, spatial_index_xenium, gdf_polygons_xenium)

    # Iterate over all the masks
    progress_bar = tqdm(total=int(len(masks_xenium_)), desc="Processing...")
    viz_count = 0
    viz_max = 100

    for xenium_id, mask_xenium in enumerate(gdf_polygons_xenium.polygon):

        xenium_bounds = np.array([bound - 10 if i < 2 else bound + 10 for i, bound in enumerate(mask_xenium.bounds)])
        masks_custom_near_index = list(spatial_index_custom.intersection(xenium_bounds))
        masks_custom_near = gdf_polygons_custom.iloc[masks_custom_near_index]

        # Randomly Visualize
        check_viz = True if random.uniform(0, 1) > 0.9 else False
        if check_viz:
            viz_count += 1

        # Visualize Xenium and Nearby Xenium
        bounds = None
        axs = None
        if viz_count < viz_max and check_viz:
            bounds = np.array(mask_xenium.bounds)
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
            masks_xenium_near_index = list(spatial_index_xenium.intersection(xenium_bounds))
            masks_xenium_near = masks_xenium_.iloc[masks_xenium_near_index]
            for j, mask_xenium_n in enumerate(masks_xenium_near.polygon):
                if mask_xenium_n != mask_xenium:
                    bounds = np.vstack((bounds, np.array(mask_xenium_n.bounds)))
                    bounds = np.vstack((bounds, np.array(mask_xenium_n.bounds)))
                    [ax.plot(np.array(mask_xenium_n.exterior.coords)[:, 0], np.array(mask_xenium_n.exterior.coords)[:, 1], 'g', zorder=2, label="xenium nearby") for ax in axs]

            [ax.plot(np.array(mask_xenium.exterior.coords)[:, 0], np.array(mask_xenium.exterior.coords)[:, 1], 'r', zorder=2, label="xenium") for ax in axs]

        for mask_custom, custom_id in zip(masks_custom_near.polygon, masks_custom_near.cell_id):

            # Visualize nearby Custom masks
            if viz_count < viz_max and check_viz:
                bounds = np.vstack((bounds, np.array(mask_custom.bounds)))
                [ax.plot(np.array(mask_custom.exterior.coords)[:, 0], np.array(mask_custom.exterior.coords)[:, 1], 'b', zorder=2, label="custom") for ax in axs]

            try:
                if mask_xenium.intersects(mask_custom):

                    intersect_custom_index.append(custom_id)
                    intersect_xenium_index.append(xenium_id)

                    # Masks that are almost the same in both segmentation
                    iou = mask_xenium.intersection(mask_custom).area / mask_xenium.union(mask_custom).area
                    iou_distribution.append(iou)
                    if iou > 0.6:
                        matched_pairs.append((xenium_id, custom_id))
                        matched += 1
                        matched_xenium[xenium_id] += 1
                        matched_custom[custom_id] += 1
                    elif mask_xenium.intersection(mask_custom).area / mask_xenium.area > 0.6 > iou:
                        contained_in_custom[custom_id] += 1
                    elif mask_xenium.intersection(mask_custom).area / mask_custom.area > 0.6 > iou:
                        contained_in_xenium[xenium_id] += 1
                    elif iou > 0.1:
                        overlap_pairs.append((xenium_id, custom_id))
                        overlap += 1

            except:
                pass

        # Visualization with Images and saving image
        if check_viz and viz_count < viz_max:

            if len(bounds.shape) > 1:
                bounds_min = np.min(bounds[:, [0, 1]], axis=0).astype(int)
                bounds_max = np.max(bounds[:, [2, 3]], axis=0).astype(int)

            else:
                bounds_min = bounds[[0, 1]].astype(int)
                bounds_max = bounds[[2, 3]].astype(int)

            extend = 10
            bounds_min -= extend if np.min(bounds_min) - extend > 0 else np.min(bounds_min)
            bounds_max += extend if np.max(bounds_max) + extend < np.max(dapi_img.shape) else np.max(
                dapi_img.shape) - 1 - np.max(bounds_max)

            axs[0].imshow(dapi_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                          extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")
            axs[1].imshow(he_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                          extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(visualization_dir / f"example_unique_{viz_count}.png")
            plt.close()

        progress_bar.update(1)

    progress_bar.close()

    print(f"match xenium: {matched/len(masks_xenium_):.2f}")
    print(f"match custom: {matched/len(masks_custom_):.2f}")

    print(f"overlap xenium: {overlap/len(masks_xenium_):.2f}")
    print(f"overlap custom: {overlap/len(masks_custom_):.2f}")

    highly_contained = heapq.nlargest(5, contained_in_custom, key=contained_in_custom.get)
    contained_dir = visualization_dir / "contained_in_custom"
    os.makedirs(contained_dir, exist_ok=True)
    for ix in highly_contained:
        visualize_masks(ix, contained_dir / f"example_{ix}_.png", masks_custom_, dapi_img, he_img, spatial_index_xenium, gdf_polygons_xenium)

    # Plot IOU Distribution
    plt.figure()
    plt.hist(iou_distribution, bins=100, fill=False, ec="r")
    plt.xlabel("IOU distribution")
    plt.ylabel("counts")
    plt.title("Jaccard Index Distribution Xenium-Custom")
    plt.savefig(results_dir_ / segmentation_method_ / f"iou_distribution.png")
    plt.close()

    # Visualize the number of Xenium mask that contains other Xenium
    plt.figure()
    contained_values = np.array(list(contained_in_xenium.values()))
    contained_values = contained_values[contained_values > 0]
    plt.yscale("log")
    bins = np.array(range(1, max(np.unique(contained_values)) + 2)) - 0.5
    plt.hist(contained_values, bins=bins, fill=False, ec="r")
    plt.xticks(range(1, max(np.unique(contained_values))+1))
    plt.xlim(0.3, max(np.unique(contained_values)) + 0.7)
    plt.savefig(results_dir_ / segmentation_method_ / f"xenium_contained_distribution.png")

    # Visualize the number of Custom mask that contains other Xenium
    plt.figure()
    contained_values = np.array(list(contained_in_custom.values()))
    contained_values = contained_values[contained_values > 0]
    plt.yscale("log")
    bins = np.array(range(1, max(np.unique(contained_values))+2)) - 0.5
    plt.hist(contained_values, bins=bins, fill=False, ec="r")
    plt.xticks(range(1, max(np.unique(contained_values))+1))
    plt.xlim(0.3,  max(np.unique(contained_values))+0.7)
    plt.savefig(results_dir_ / segmentation_method_ / f"custom_contained_distribution.png")

    # Visualize example of masks that overlap partially

    # Visualize example of masks that do not overlap with Xenium
    non_overlapping = visualization_dir / "non_overlapping"
    os.makedirs(non_overlapping, exist_ok=True)
    non_overlapping_xenium = non_overlapping / "xenium"
    os.makedirs(non_overlapping_xenium, exist_ok=True)
    non_overlapping_custom = non_overlapping / "custom"
    os.makedirs(non_overlapping_custom, exist_ok=True)
    non_overlapping_index_xenium = set(masks_xenium_.cell_id).difference(intersect_xenium_index)
    non_overlapping_index_custom = set(masks_custom_.cell_id).difference(intersect_custom_index)
    viz_count = 0
    for ix in non_overlapping_index_xenium:
        if viz_count < 100 and random.uniform(0, 1) > 0.7:
            viz_count += 1
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
            polygon = masks_xenium_[masks_xenium_.cell_id == ix].polygon.tolist()[0]
            bounds = np.array([bound - 10 if i < 2 else bound + 10 for i, bound in enumerate(polygon.bounds)])

            bounds_min = bounds[[0, 1]].astype(int)
            bounds_max = bounds[[2, 3]].astype(int)

            try:
                axs[0].imshow(dapi_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                              extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")
                axs[1].imshow(he_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                              extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")

                [ax.plot(np.array(polygon.exterior.coords)[:, 0], np.array(polygon.exterior.coords)[:, 1], 'g',
                         zorder=2, label="xenium") for ax in axs]

                nearby_index = list(spatial_index_custom.intersection(bounds))
                masks_custom_near = gdf_polygons_custom.iloc[nearby_index]
                for polygon_custom in masks_custom_near.polygon:
                    [ax.plot(np.array(polygon_custom.exterior.coords)[:, 0], np.array(polygon_custom.exterior.coords)[:, 1], 'b',
                             zorder=2, label="custom") for ax in axs]

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()

                plt.savefig(non_overlapping_xenium / f"example_{viz_count}.png")
                plt.close()
            except:
                pass

    # Visualize example of masks that do not overlap with Custom
    viz_count = 0
    for ix in non_overlapping_index_custom:
        if viz_count < 100 and random.uniform(0, 1) > 0.7:
            viz_count += 1
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
            polygon = masks_custom_[masks_custom_.cell_id == ix].polygon.tolist()[0]
            bounds = np.array([bound - 10 if i < 2 else bound + 10 for i, bound in enumerate(polygon.bounds)])
            bounds_min = bounds[[0, 1]].astype(int)
            bounds_max = bounds[[2, 3]].astype(int)
            try:
                axs[0].imshow(dapi_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                              extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")
                axs[1].imshow(he_img[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                              extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]), origin="lower")

                [ax.plot(np.array(polygon.exterior.coords)[:, 0], np.array(polygon.exterior.coords)[:, 1], 'b',
                         zorder=2, label="custom") for ax in axs]

                nearby_index = list(spatial_index_xenium.intersection(bounds))
                masks_xenium_near = gdf_polygons_xenium.iloc[nearby_index]
                for polygon_custom in masks_xenium_near.polygon:
                    [ax.plot(np.array(polygon_custom.exterior.coords)[:, 0], np.array(polygon_custom.exterior.coords)[:, 1], 'g',
                             zorder=2, label="xenium") for ax in axs]

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                plt.savefig(non_overlapping_custom / f"example_{viz_count}.png")
                plt.close()
            except:
                pass


def test_nucleus_features_extraction(img_, masks_outline, adata_, results_dir_: Path, segmentation_method_: str, img_type: str):

    save_dir = results_dir_ / segmentation_method_
    os.makedirs(save_dir, exist_ok=True)

    # Get Labels
    adata_ = preprocess_transcriptomics(adata_, filter_=True)
    adata_label = compute_ref_labels(adata_)
    diff = set(masks_outline["cell_id"].tolist()).difference(adata_label.obs["cell_id"].tolist())

    # Create Masks map
    masks_outline = masks_outline[[False if id_ in diff else True for id_ in masks_outline.cell_id]]
    indices = get_polygon_indices(masks_outline)

    # Compute Masks features
    print("Extracting - nuclear features")
    progress_bar = tqdm(total=len(indices), desc="Processing")
    nucleus_features = []
    fos_name = []
    for ix in indices:

        # First order statistic
        dapi_signal = img_[ix[:, 0], ix[:, 1]]
        if len(dapi_signal.shape) > 1:
            dapi_signal = dapi_signal.flatten()
        fos_value, fos_name = compute_fos(dapi_signal)
        cell_area = get_cell_area(dapi_signal)
        nucleus_features.append(fos_value + [cell_area])

        progress_bar.update(1)

    nucleus_features = np.array(nucleus_features)

    np.nan_to_num(nucleus_features, copy=False)  # some cells only have 0 pixels and

    save_dir = save_dir / "umap_pca" / img_type
    os.makedirs(save_dir, exist_ok=True)

    for feature_, feature_name_ in zip(nucleus_features.T, fos_name + ["cell area"]):
        plt.figure()
        plt.hist(feature_, bins=100)
        plt.xlabel(f"{feature_name_}")
        plt.ylabel("counts")
        plt.title(f"{img_type} - {feature_name_}")
        plt.tight_layout()
        plt.savefig(save_dir / f"feature_hist_{feature_name_}_{img_type}.png")
        plt.close()

    visualize_umap_pca(nucleus_features, adata_label.obs["leiden"].astype(int).tolist(), save_dir, img_type)

    return 0


def test_nucleus_type_classification(adata_: sc.AnnData, save_dir_: Path, filename_: str, method_="leiden",
                                     visualize_: bool = True):

    leiden_dir = save_dir_ / "leiden_clustering"
    os.makedirs(leiden_dir, exist_ok=True)

    initial = len(adata_)
    print("Initial Number:", initial)

    adata_filtered = preprocess_transcriptomics(adata_, filter_=True)

    filtered = len(adata_filtered)
    print("Filtered:", filtered)

    n_neighbors = [5, 15, 30, 40, 50]
    n_pca = [25, 50, 100, 200]
    cmap = plt.get_cmap("tab20")

    print("Testing Clustering for Transcriptomics")
    for neigh in n_neighbors:
        for n_comp in n_pca:
            print(f"\tPCA {n_comp} - Neighbor {neigh}")
            adata_labels = compute_ref_labels(adata_filtered, n_comp=n_comp, n_neighbors=neigh)
            labels_ = adata_labels.obs["leiden"].astype(int).tolist()
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=5, label=label)
                for i, label in enumerate(np.sort(np.unique(labels_)))]

            plt.scatter(adata_labels.obsm["X_umap"][:, 0], adata_labels.obsm["X_umap"][:, 1], c=cmap(labels_), s=0.2)
            plt.xlabel("U-map 1")
            plt.ylabel("U-map 2")
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
                       title_fontsize="large")
            plt.tight_layout()
            plt.savefig(leiden_dir / f"pca_{n_comp}-neigh_{neigh}.png")

    print()
# -------------------------------------------------------------------------------------------------------------------- #
# MAIN METHODS


def nucleus_transcripts_matrix_creation(masks_nucleus, transcripts_df, background_masks, results_dir_, filename_,
                                        sample_: bool = False):
    """ Creates the count matrix (nucleus vs genes) necessary for further downstream analysis

    :param masks_nucleus: the masks from nuclei as df containing cell id and polygons
    :param transcripts_df:
    :param background_masks:
    :param results_dir_:
    :param filename_:
    :return: np.ndarray, Annotation Data objects
    """

    results_dir_ = results_dir_ / "adata"
    os.makedirs(results_dir_, exist_ok=True)

    if os.path.isfile(results_dir_ / (filename_ + ".h5ad")):
        if sample_:
            return sc.read_h5ad(results_dir_ / (filename_ + ".h5ad"))[0:10000]
        else:
            return sc.read_h5ad(results_dir_ / (filename_ + ".h5ad"))

    feature_name = np.unique(transcripts_df.feature_name)
    feature_name = [feature for feature in feature_name if not "antisense" in feature]
    feature_name_index = {gene: ix for ix, gene in enumerate(feature_name)}
    count_matrix = np.empty((len(masks_nucleus), len(feature_name)))

    # Convert masks_xenium DataFrame to a GeoDataFrame of Polygons
    gdf_polygons = gpd.GeoDataFrame(masks_nucleus, geometry=masks_nucleus.polygon)

    # Convert transcripts DataFrame to a GeoDataFrame of Points
    gdf_points = gpd.GeoDataFrame(transcripts_df, geometry=[Point(x, y) for x, y in zip(transcripts_df.x_pixels, transcripts_df.y_pixels)])

    # Create spatial index for the points GeoDataFrame
    spatial_index = gdf_points.sindex

    # Update count matrix and observations
    progress_bar = tqdm(total=len(gdf_polygons), desc=f"Processing Masks for: {filename_}")
    observations_dict = {"cell_id": [], "transcripts_count": [], "x_centroid": [], "y_centroid": [], "nucleus_area": []}
    non_background = []
    for i, polygon in enumerate(gdf_polygons.geometry):

        # Check if the masks is not in background
        if background_masks.contains(polygon):

            # Use spatial index to narrow down the list of points to check
            possible_points_index = list(spatial_index.intersection(polygon.bounds))
            possible_points = gdf_points.iloc[possible_points_index]

            # Use within method to update count matrix
            match_index = possible_points[possible_points.within(polygon)]
            gene_counts = match_index.feature_name.value_counts()
            for ix in gene_counts.index:
                if ix in feature_name_index:
                    count_matrix[i, feature_name_index[ix]] += gene_counts.loc[ix]

            # Use value to update observations
            observations_dict["cell_id"].append(gdf_polygons.iloc[i]["cell_id"])
            observations_dict["transcripts_count"].append(len(match_index))
            observations_dict["nucleus_area"].append(polygon.area * (0.2125**2))
            observations_dict["x_centroid"].append(polygon.centroid.x)
            observations_dict["y_centroid"].append(polygon.centroid.y)

        else:
            non_background.append(i)

        progress_bar.update(1)

    # Observations:
    #   - cell_id, x_centroid, y_centroid, transcript_counts, cell_area, nucleus_area
    #   - index = cell_id
    # Variables:
    #   - feature_types, SYMBOL (additional downstream elements, k_means_clustering)
    #   - index = Ensemble of gene

    variables_dict = {"SYMBOL": feature_name, "feature_type": ["Gene Expression" for _ in feature_name]}
    variables_df = pd.DataFrame.from_dict(variables_dict)
    variables_df.index = feature_name
    observations_df = pd.DataFrame.from_dict(observations_dict)

    count_matrix = np.delete(count_matrix, non_background, axis=0)

    # Create new Annotation Data object with count observation and variables
    adata_new = sc.AnnData(count_matrix, obs=observations_df, var=variables_df)

    # Add masks and corresponding resolution
    masks_nucleus = convert_masks_to_df(masks_nucleus, observations_dict["cell_id"])
    adata_new.uns["nucleus_boundaries"] = masks_nucleus
    adata_new.uns["resolution"] = 0.2125

    # Add transcripts information used for matrix counts
    adata_new.uns["spots"] = transcripts_df

    # Add the number of masks that were filtered with background masks
    adata_new.uns["non_background_mask"] = non_background

    adata_new.write_h5ad(results_dir_ / (filename_+".h5ad"))
    print(f"File saved at: {results_dir_ / (filename_+ '.h5ad')}")
    print(f"Number of masks assigned to background: {len(non_background)}")

    return adata_new


def nucleus_type_classification(adata_: sc.AnnData, save_dir_: Path, filename_: str, method_="leiden",
                                visualize_: bool = True):
    """

    :param adata_:
    :param save_dir_:
    :param filename_:
    :param method_:
    :param visualize_:
    :return: adata filtered with cell label
    """

    adata_filtered = preprocess_transcriptomics(adata_, filter_=True)
    adata_labeled = compute_ref_labels(adata_filtered, n_comp=50, n_neighbors=13)


def run_transcripts_assignments():
    """

    :return:
    """


def run_nucleus_features_extraction():
    """

    :return:
    """


def build_dir():
    results = get_results_path() / "nuclear_features"
    os.makedirs(results, exist_ok=True)
    return results


if __name__ == "__main__":
    # -------------------------------------------------------- #
    # Script Parameters
    segmentation_method = "stardist_qupath"  # alt: "stardist_python_pmin0-1_pmin99-9_scale0-5"
    run_nucleus_features_analysis = True
    test_nucleus_features_analysis = True
    sample = False
    # -------------------------------------------------------- #
    # Loading and Set Up

    # Setup
    results_dir = build_dir()
    data_path = get_human_breast_he_path()

    # Load DAPI Image
    aligned_image_path = get_human_breast_dapi_aligned_path()
    image_dapi = load_xenium_he_ome_tiff(aligned_image_path, level_=0)
    aligned_image_path = get_human_breast_he_aligned_path()
    image_he = load_xenium_he_ome_tiff(aligned_image_path, level_=0)

    path_background = get_geojson_masks("background_masks")
    background = load_geojson_background(path_background)

    # Load Custom Masks
    masks = load_geojson_masks(get_geojson_masks(segmentation_method), background)

    # Load Formatted Xenium Default Segmentation Masks
    masks_xenium = load_xenium_masks(data_path, format="pixel", resolution=0.2125)

    masks["nucleus_area"] = masks.polygon.apply(lambda c: c.area)
    # Load Formatted Transcripts from Xenium
    transcripts = load_xenium_transcriptomics(data_path)

    # Create adata object for nucleus segmentation analysis with different segmentation
    adata_filename_custom = f"Xenium-BreastCancer1-nucleus_{segmentation_method}"
    adata_custom = nucleus_transcripts_matrix_creation(masks, transcripts, background, results_dir, adata_filename_custom, sample)
    adata_filename_xenium = "Xenium-BreastCancer1-nucleus_xenium_segmentation"
    adata_xenium = nucleus_transcripts_matrix_creation(masks_xenium, transcripts, background, results_dir, adata_filename_xenium, sample)

    # -------------------------------------------------------- #
    # Main Analysis
    if run_nucleus_features_analysis:
        pass

    # -------------------------------------------------------- #
    # Run Tests

    if test_nucleus_features_analysis:
        test_nucleus_type_classification(adata_custom, save_dir_=results_dir, filename_=adata_filename_custom+"_labeled")

        # Tests mask differences
        test_masks_area_distribution(masks_xenium, masks, results_dir)
        # test_masks_transcripts_difference(adata_xenium, adata_custom, results_dir, segmentation_method)
        # test_masks_overlap(masks_xenium, masks, image_dapi, image_he, results_dir, segmentation_method)

        # Tests nucleus features extraction
        # test_nucleus_features_extraction(image_dapi, masks, adata_custom, results_dir_=results_dir,
        #                                  segmentation_method_=segmentation_method, img_type="dapi")
        # test_nucleus_features_extraction(image_he, masks, adata_custom, results_dir_=results_dir,
        #                                  segmentation_method_=segmentation_method, img_type="h&e")
