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
        ** strategy 1: start with 2D clustering algorithm
            - using Stardist on H&E from QuPath extracted with GeoJson as a starter
        - strategy 2: build a good performing 3D algorithm

[ x ](2) Extract Masks from QuPath to a usable version in Python

[ x ](3) Extract nucleus individually for analysis
        - store masks.pkl (containing the label for each mask) can be used to extract the pixel for each mask

[ x ](4) Create adata objects from masks and transcripts

[ x ](5) Visualization and Comparison between Masks

Last Revision: 12.02.24
"""
import json
import random
# Std
from pathlib import Path
import os
import pickle

import cv2
import geojson
# Third party
import numpy as np
import pandas as pd
import shapely.geometry
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from shapely import Point
import geopandas as gpd
from collections import OrderedDict
import scanpy as sc
import seaborn as sns

# Relative import
from src.utils import (load_xenium_data, get_results_path, get_human_breast_dapi_aligned_path, load_xenium_he_ome_tiff,
                       load_xenium_masks, load_xenium_transcriptomics, get_human_breast_he_path, load_geojson_masks,
                       preprocess_transcriptomics, get_human_breast_he_aligned_path)

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


def get_cell_label(path_replicate_: Path, label_name_: str, masks: np.ndarray):
    """ This will require some work to recover cell label (obtained through transcriptomics)
        with their location on the plane """

    adata = load_xenium_data(path_replicate_)

    # Two Strategy:
    # - take previous labels from graph clustering and find overlapping nucleus boundaries
    # - compute cell boundaries based on new nucleus boundaries and perform cell transcriptomics association
    # Second Strategy seems better since it avoids issues with nucleus boundaries that are extremely different

    return adata.obs[label_name_].astype(int).tolist()


def get_cell_area(dapi_signal):

    cell_per_micron = 0.2125
    return len(dapi_signal) * (cell_per_micron ** 2)


def visualize_umap(nucleus_features, masks_type, results_dir_):
    # Visualize U-map with first order-moment
    embedding = umap.UMAP(n_neighbors=15, n_components=2).fit_transform(nucleus_features)

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.xlabel("umap-1")
    plt.ylabel("umap-2")
    plt.savefig(results_dir_ / f"umap_fos_{masks_type}.png")


def compute_ref_labels(adata, n_comp: int = 50, n_neighbors: int = 13):
    adata_ = adata.copy()
    sc.pp.pca(adata_, n_comps=n_comp)
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors)  # necessary for UMAP (k-neighbors with weights)
    sc.tl.umap(adata_)
    sc.tl.leiden(adata_)

    return adata_.obs["leiden"].values.tolist()


def convert_masks_to_df(masks_):
    """

    :param masks_: a collection of polygon
    :return: a dataframe containing vertex x vertex y and cell_id
    """

    df_dict = {"cell_id": [], "vertex_x": [], "vertex_y": []}
    for i, mask in enumerate(masks_):
        pts = np.array(mask.exterior.coords, np.int32)
        for pt in pts:
            df_dict["cell_id"].append(i+1)
            df_dict["vertex_x"].append(pt[0])
            df_dict["vertex_y"].append(pt[1])

    return pd.DataFrame.from_dict(df_dict)


def create_masks_image(masks_outline, width, height):
    """ Takes a collection of polygon, and draws coresponding masks with
        width and height.

    :param masks_outline:
    :param width:
    :param height:
    :return:
    """

    mask = np.zeros((height, width), dtype=np.int32)

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


def load_geojson_background(path: Path):
    """ Create a MultiPolygon shapely to remove background mask"""
    with open(path, "r") as file:
        dict_ = json.load(file)
    multipolygon = shapely.geometry.shape(dict_["features"][0]["geometry"])
    return multipolygon


# -------------------------------------------------------------------------------------------------------------------- #
# TEST METHODS


def test_masks_difference(masks_xenium_: pd.DataFrame, masks_custom_: pd.DataFrame, results_dir_: Path):

    # Step 1 Compare Surface Distribution
    xenium_areas = []
    for polygon in masks_xenium_["polygon"]:
        xenium_areas.append(polygon.area * 0.2125**2)
    print("Max Area (xeniums):", np.sort(xenium_areas)[-10:])

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
    plt.xlabel("nucleus area [um]")
    plt.ylabel("density [a.u.]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir_ / "nucleus_areas_histogram.png")

    return masks_xenium_, masks_custom_


def test_masks_transcripts_difference(adata_xenium_, adata_custom_, results_dir_, segmentation_method_):

    # Filter
    #filter_index = np.argsort(adata_custom_.obs["nucleus_area"])[-3:].tolist()
    #adata_custom_ = adata_custom_[~adata_custom_.obs.index.astype(int).isin(filter_index)]

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
    plt.hist(number_of_genes_custom, bins=np.max(number_of_genes_custom).astype(int), density=True, fill=False, ec='r',
             label="custom")
    plt.hist(number_of_genes_xenium, bins=np.max(number_of_genes_custom).astype(int), density=True, fill=False, ec='b',
             label="xenium")
    plt.xlabel("number of unique genes")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir_ / f"number_of_genes.png")

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
    #       - for both if nuclei are neither contained in another or matched with another -> not segmented
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

    # Iterate over all the masks
    progress_bar = tqdm(total=int(len(masks_xenium_)), desc="Processing...")
    viz_count = 0
    viz_max = 100
    for i, mask_xenium in enumerate(masks_xenium_.polygon):

        masks_custom_near_index = list(spatial_index_custom.intersection(mask_xenium.bounds))
        masks_custom_near = masks_custom_.iloc[masks_custom_near_index]

        # Randomly Visualize
        check_viz = True if random.uniform(0, 1) > 0.7 else False
        if check_viz:
            viz_count += 1

        # Visualize Xenium and Nearby Xenium
        if viz_count < viz_max and check_viz:
            bounds = np.array(mask_xenium.bounds)
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
            masks_xenium_near_index = list(spatial_index_xenium.intersection(mask_xenium.bounds))
            masks_xenium_near = masks_xenium_.iloc[masks_xenium_near_index]
            for j, mask_xenium_n in enumerate(masks_xenium_near.polygon):
                if mask_xenium_n != mask_xenium:
                    bounds = np.vstack((bounds, np.array(mask_xenium_n.bounds)))
                    bounds = np.vstack((bounds, np.array(mask_xenium_n.bounds)))
                    [ax.plot(np.array(mask_xenium_n.exterior.coords)[:, 0], np.array(mask_xenium_n.exterior.coords)[:, 1], 'g', zorder=2, label="xenium nearby") for ax in axs]

            [ax.plot(np.array(mask_xenium.exterior.coords)[:, 0], np.array(mask_xenium.exterior.coords)[:, 1], 'r', zorder=2, label="xenium") for ax in axs]

        for j, mask_custom in enumerate(masks_custom_near.polygon):

            # Visualize nearby Custom masks
            if viz_count < viz_max and check_viz:
                bounds = np.vstack((bounds, np.array(mask_custom.bounds)))
                [ax.plot(np.array(mask_custom.exterior.coords)[:, 0], np.array(mask_custom.exterior.coords)[:, 1], 'b', zorder=2, label="custom") for ax in axs]

            try:
                if mask_xenium.intersects(mask_custom):
                    intersect_custom_index.append(j)
                    intersect_xenium_index.append(i)

                    # Masks that are almost the same in both segmentation
                    iou = mask_xenium.intersection(mask_custom).area / mask_xenium.union(mask_custom).area
                    iou_distribution.append(iou)
                    if iou > 0.6:
                        matched_pairs.append((i, j))
                        matched += 1
                        matched_xenium[i] += 1
                        matched_custom[j] += 1
                    elif mask_xenium.intersection(mask_custom).area / mask_xenium.area > 0.6 > iou:
                        contained_in_custom[j] += 1
                    elif mask_xenium.intersection(mask_custom).area / mask_custom.area > 0.6 > iou:
                        contained_in_xenium[i] += 1
                    elif iou > 0.1:
                        overlap_pairs.append((i, j))
                        overlap += 1

            except Exception as e:
                print(f"error: {e}")

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
    plt.hist(contained_values, bins=len(np.unique(contained_values)), fill=False, ec="r")
    plt.savefig(results_dir_ / segmentation_method_ / f"xenium_contained_distribution.png")

    # Visualize the number of Custom mask that contains other Xenium
    plt.figure()
    contained_values = np.array(list(contained_in_custom.values()))
    contained_values = contained_values[contained_values > 0]
    plt.yscale("log")
    plt.hist(contained_values, bins=len(np.unique(contained_values)), fill=False, ec="r")
    plt.savefig(results_dir_ / segmentation_method_ / f"custom_contained_distribution.png")


    # Visualize example of masks that overlap partially

    # Visualize example of masks that do not overlap with Xenium

    # Visualize example of masks that do not overlap with Custom


def test_nucleus_features_extraction(img_, masks_outline, adata_, masks_type: str, results_dir_: Path):

    # Get Labels
    # adata_ = preprocess_transcriptomics(adata_, filter_=False)
    # labels = compute_ref_labels(adata_)

    # Create Masks map
    # Remark: the masks label corresponds to label because cell_id are ordered from 1 to n
    masks_image = create_masks_image(masks_outline, img_.shape[0], img_.shape[1])

    # Compute Masks features
    print("Extracting - nuclear features")

    progress_bar = tqdm(total=len(np.unique(masks_image)), desc="Processing")
    nucleus_features = []
    for label in np.unique(masks_image)[1:]:

        # masks_copy = masks.copy()
        # masks_copy[np.where(masks != label)] = 0
        # masks_copy[np.where(masks == label)] = 1

        # First order statistic
        mask_coordinate = np.where(masks_image == label)
        dapi_signal = img_[mask_coordinate]
        fos_value, fos_name = compute_fos(dapi_signal)
        cell_area = get_cell_area(dapi_signal)
        nucleus_features.append(fos_value + [cell_area])

        progress_bar.update(1)

    nucleus_features = np.array(nucleus_features)

    visualize_umap(nucleus_features, masks_type, results_dir_)

    return 0

# -------------------------------------------------------------------------------------------------------------------- #
# MAIN METHODS


def nucleus_transcripts_matrix_creation(masks_nucleus, transcripts_df, background_masks, results_dir_, filename_):
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
        return sc.read_h5ad(results_dir_ / (filename_ + ".h5ad"))

    feature_name = np.unique(transcripts_df.feature_name)
    feature_name = [feature for feature in feature_name if not "antisense" in feature]
    feature_name_index = {gene: ix for ix, gene in enumerate(feature_name)}
    count_matrix = np.empty((len(masks_nucleus), len(feature_name)))

    # Convert masks_xenium DataFrame to a GeoDataFrame of Polygons
    gdf_polygons = gpd.GeoDataFrame(masks_nucleus, geometry=masks_nucleus.polygon)

    # Convert transcripts DataFrame to a GeoDataFrame of Points
    gdf_points = gpd.GeoDataFrame(transcripts, geometry=[Point(x, y) for x, y in zip(transcripts.x_pixels, transcripts.y_pixels)])

    # Create spatial index for the points GeoDataFrame
    spatial_index = gdf_points.sindex

    # Update count matrix and observations
    progress_bar = tqdm(total=len(gdf_polygons), desc="Processing Xenium Masks")
    observations_dict = {"cell_id": [], "transcripts_count": [], "x_centroid": [], "y_centroid": [], "nucleus_area": []}
    non_background = 0
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
            observations_dict["cell_id"].append(i+1)
            observations_dict["transcripts_count"].append(len(match_index))
            observations_dict["nucleus_area"].append(polygon.area * (0.2125**2))
            observations_dict["x_centroid"].append(polygon.centroid.x)
            observations_dict["y_centroid"].append(polygon.centroid.y)

        else:
            non_background += 1

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

    # Create new Annotation Data object with count observation and variables
    adata_new = sc.AnnData(count_matrix, obs=observations_df, var=variables_df)

    # Add masks and corresponding resolution
    masks_nucleus = convert_masks_to_df(masks_nucleus.polygon)
    adata_new.uns["nucleus_boundaries"] = masks_nucleus
    adata_new.uns["resolution"] = 0.2125

    # Add transcripts information used for matrix counts
    adata_new.uns["spots"] = transcripts_df
    adata_new.uns["non_background_mask"] += 1

    adata_new.write_h5ad(results_dir_ / (filename_+".h5ad"))
    print(f"File saved at: {results_dir_ / (filename_+ '.h5ad')}")
    print(f"Number of masks assigned to background: {background_masks}")

    return adata_new


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
    # segmentation_method = "stardist_python_pmin0-1_pmin99-9_scale0-5"
    segmentation_method = "stardist_qupath"
    run_nucleus_features_analysis = True
    test_nucleus_features_analysis = True

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

    # path_background = get_path_he_registered_background_mask()
    path_background = Path("/Users/lbrunsch/Desktop/QuPath_custom/Xenium_FFPE_Human_Breast_Cancer_DAPI_registered_background.geojson")
    background = load_geojson_background(path_background)

    # Load Custom Masks
    if segmentation_method == "stardist_qupath":
        masks = load_geojson_masks(Path("/Users/lbrunsch/Desktop/QuPath_custom/Xenium_FFPE_Human_Breast_Cancer_DAPI_registered.geojson"))
    elif segmentation_method == "stardist_python_pmin0-1_pmin99-9_scale0-5":
        masks = load_geojson_masks(Path("/Users/lbrunsch/Desktop/QuPath_custom/pmin0-1_pmin99-9_scale0-5.geojson"))
    else:
        raise ValueError("Not Implemented")

    # Load Formatted Xenium Default Segmentation Masks
    masks_xenium = load_xenium_masks(data_path, format="pixel", resolution=0.2125)

    # Load Formatted Transcripts from Xenium
    transcripts = load_xenium_transcriptomics(data_path)

    # Load adata formatted (original xenium experiments)
    adata = load_xenium_data(data_path)

    # Create adata object for nucleus segmentation analysis with different segmentation
    adata_filename_custom = f"Xenium-BreastCancer1-nucleus_{segmentation_method}"
    adata_custom = nucleus_transcripts_matrix_creation(masks, transcripts, background, results_dir, adata_filename_custom)
    adata_filename_xenium = "Xenium-BreastCancer1-nucleus_xenium_segmentation"
    adata_xenium = nucleus_transcripts_matrix_creation(masks_xenium, transcripts, background, results_dir, adata_filename_xenium)

    # -------------------------------------------------------- #
    # Main Analysis
    if run_nucleus_features_analysis:
        pass

    # -------------------------------------------------------- #
    # Run Tests

    if test_nucleus_features_analysis:
        # Tests mask differences
        test_masks_difference(masks_xenium, masks, results_dir)
        test_masks_transcripts_difference(adata_xenium, adata_custom, results_dir, segmentation_method)
        test_masks_overlap(masks_xenium, masks, image_dapi, image_he, results_dir, segmentation_method)
        test_nucleus_features_extraction(image_dapi, masks_xenium, adata, masks_type="xenium", results_dir_=results_dir)
