"""
Description: This file will convert GeoJson masks to Polygon and optimize the parameters of the local normalization


Parameters:
    - tile size: 1024, 2048
    - normalize percentile: 0.1/99.9, 0.5, 99.5
    - resize: 0.4, 0.5, 0.6


"""

import os
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from src.utils import (load_xenium_he_ome_tiff, get_results_path, get_data_path,
                       get_human_breast_he_aligned_path, get_human_breast_dapi_aligned_path)


def test_normalize_percentile_distribution(dapi_image, results_dir_):
    dapi_image = dapi_image.flatten()
    percents = [0.1, 0.5, 1.0, 5.0, 10, 95, 99.0, 99.5, 99.9, 99.99]
    percentiles = np.percentile(dapi_image, percents)

    # Plot histogram on a logarithmic scale
    plt.figure(figsize=(12, 8))
    plt.hist(dapi_image, bins=120, log=True, color='blue', alpha=0.7)
    plt.title('Histogram of DAPI Intensities on Log Scale')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency (Log Scale)')

    # Annotate percentiles on the plot
    for percentile, percent in zip(percentiles, percents):
        plt.axvline(x=percentile, color='red', linestyle='--', linewidth=1)
        plt.text(percentile, 10, f'{percentile:.1f} - {percent} - ', rotation=90, color='red')

    plt.savefig(results_dir_ / "percentile_distribution")
    plt.close()


def boxplot(keys, values, y_label, title, measure, save_file: Path = None):
    if len(keys) == 3:
        df = pd.DataFrame({
            'Category': [keys[0]] * len(values[0]) + [keys[1]] * len(values[1]) + [keys[2]] * len(values[2]),
            'Value': values[0] + values[1] + values[2]
        })
        means = [df[df["Category"] == keys[0]]['Value'].median(), df[df["Category"] == keys[1]]['Value'].median(),  df[df["Category"] == keys[2]]['Value'].median()]
        categories = [keys[0], keys[1], keys[2]]
    elif len(keys) == 2:
        df = pd.DataFrame({
            'Category': [keys[0]] * len(values[0]) + [keys[1]] * len(values[1]),
            'Value': values[0] + values[1]
        })
        means = [df[df["Category"] == keys[0]]['Value'].median(), df[df["Category"] == keys[1]]['Value'].median()]
        categories = [keys[0], keys[1]]
    elif len(keys) == 1:
        return

    # Create boxplot
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x='Category', y='Value', data=df, order=categories)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel(y_label)
    plt.ylim(-0.1*max(df.Value), 1.2*max(df.Value))

    # Perform statistical test (e.g., t-test) to determine significance
    if len(keys) == 2:
        stat, p_value = stats.mannwhitneyu(values[0], values[1], alternative="greater")
        plt.text(0.3, 0.95, f"median: {means[0]}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.6, 0.95, f"median: {means[1]}", ha='center', va='center', transform=ax.transAxes)

        if p_value < 0.05:
            plt.text(0.3, 0.9, 'p-value={:.3f} (Significant)'.format(p_value), ha='center', va='center',
                     transform=ax.transAxes)
        else:
            plt.text(0.6, 0.9, 'p-value={:.3f} (Not Significant)'.format(p_value), ha='center', va='center',
                     transform=ax.transAxes)

    elif len(keys) == 3:
        # You can perform pairwise comparisons if you have 3 groups
        # For example, compare group 1 with group 2, group 2 with group 3, and group 1 with group 3
        p_values = []
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                group1 = df[df['Category'] == categories[i]]['Value']
                group2 = df[df['Category'] == categories[j]]['Value']
                if measure == "solidity":
                    stat, p_val = stats.mannwhitneyu(group1, group2, alternative="less")
                elif measure in ["size", "circularity"]:
                    stat, p_val = stats.mannwhitneyu(group1, group2, alternative="greater")
                else:
                    stat, p_val = stats.mannwhitneyu(group1, group2)

                p_values.append(p_val)

        # Bonferroni correction for multiple comparisons
        alpha = 0.05
        adjusted_alpha = alpha / len(p_values)
        for i, p_value in enumerate(p_values):
            plt.text(0.2 + 0.3*i, 0.95, 'p-value={:.3f} (p_adj:{:.2f})'.format(p_value, adjusted_alpha), ha='center', va='center', transform=ax.transAxes)

        plt.text(0.2, 0.9, f"median: {means[0]}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.5, 0.9, f"median: {means[1]}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.8, 0.9, f"median: {means[2]}", ha='center', va='center', transform=ax.transAxes)

    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()

    plt.close()


def test_number_cells(masks_dict, params_cat, str_cat, results_dir_: Path):

    results_dir_ = results_dir_ / "count_matrix"
    os.makedirs(results_dir_, exist_ok=True)

    fixed_params = params_cat.keys()
    other_indices = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    for i, fixed in enumerate(fixed_params):
        param_option = params_cat[fixed]
        others = other_indices[i]
        for option in param_option:
            # have different categories

            cat_name_1 = list(params_cat.keys())[others[0]]
            cat_name_2 = list(params_cat.keys())[others[1]]
            params_cat_1 = params_cat[cat_name_1]
            params_cat_2 = params_cat[cat_name_2]

            count_matrix = np.zeros((len(params_cat_1), len(params_cat_2)))
            # build count matrix:
            for cat in str_cat:
                tmp = cat.split("_")
                if tmp[i] == option:
                    x = np.where(np.array(params_cat_1) == tmp[others[0]])[0][0]
                    y = np.where(np.array(params_cat_2) == tmp[others[1]])[0][0]
                    count_matrix[x, y] = len(masks_dict[cat])

            df = pd.DataFrame(count_matrix, index=params_cat_1, columns=params_cat_2)
            plt.figure(figsize=(10, 6))
            sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.1f')  # annot=True to display the count values
            plt.title(f'Number of Cells - {fixed} = {option}')
            plt.xlabel(cat_name_2)
            plt.ylabel(cat_name_1)
            plt.savefig(results_dir_ / f"{fixed}_{option}.png")


def test_feature_distribution(feature_name: str, masks_dict: dict, params_cat: dict, str_cat: list, image: np.ndarray, results_dir_: Path,):
    assert feature_name in ["size", "mean", "solidity", "circularity"]
    ylabel = {"size": "nucleus area", "mean": "mean intensity", "solidity": "solidity", "circularity": "circularity"}
    results_dir_ = results_dir_ / feature_name
    os.makedirs(results_dir_, exist_ok=True)

    fixed_params = params_cat.keys()

    for i, fixed_1 in enumerate(fixed_params):

        params_options_1 = params_cat[fixed_1]

        # Fixed a category
        for option_1 in params_options_1:
            for j, fixed_2 in enumerate(fixed_params):
                if i != j:
                    params_options_2 = params_cat[fixed_2]
                    for option_2 in params_options_2:

                        sub_cats = []
                        for cat in str_cat:
                            if str(option_1) == cat.split("_")[i] and str(option_2) == cat.split("_")[j]:
                                sub_cats.append(cat)

                        feature_by_cat = {}
                        for cat in sub_cats:
                            feature_by_cat[cat] = []
                            for mask in masks_dict[cat]:
                                if feature_name == "size":
                                    feature_by_cat[cat].append(mask.area * 0.2125**2)
                                elif feature_name == "mean":
                                    minx, miny, maxx, maxy = mask.bounds
                                    x = np.arange(int(minx), int(maxx) + 1)
                                    y = np.arange(int(miny), int(maxy) + 1)
                                    xx, yy = np.meshgrid(x, y)
                                    ix = np.vstack((yy.flatten(), xx.flatten())).T
                                    signal = image[ix[:, 0], ix[:, 1]]
                                    feature_by_cat[cat].append(np.mean(signal))
                                elif feature_name == "solidity":
                                    solidity = mask.area * 0.2125**2 / (mask.convex_hull.area * 0.2125**2)
                                    feature_by_cat[cat].append(solidity)
                                elif feature_name == "circularity":
                                    circularity =(mask.length ** 2 * 0.2125) / 4 * np.pi * mask.area * 0.2125**2
                                    feature_by_cat[cat].append(circularity)

                        values = list(feature_by_cat.values())
                        keys = list(feature_by_cat.keys())
                        save_file = results_dir_ / f"{fixed_1}{option_1}_{fixed_2}{option_2}.png"
                        boxplot(keys, values, ylabel[feature_name], f"{feature_name} Distribution", feature_name, save_file)


def test_matching(masks_he_, masks_dapi_, dapi_img_, he_img_, results_dir_):
    """

    :return:
    """
    results_dir_ = results_dir_
    os.makedirs(results_dir_ / "0_3-0_4", exist_ok=True)
    os.makedirs(results_dir_ / "0_4-0_5", exist_ok=True)
    os.makedirs(results_dir_ / "0_5-0_6", exist_ok=True)
    os.makedirs(results_dir_ / "0_6-0_7", exist_ok=True)
    os.makedirs(results_dir_ / "0_7-0_8", exist_ok=True)
    os.makedirs(results_dir_ / "0_8-0_9", exist_ok=True)
    os.makedirs(results_dir_ / "0_9-1_0", exist_ok=True)
    os.makedirs(results_dir_ / "no_match", exist_ok=True)

    min_dapi = np.min(dapi_img_)
    max_dapi = np.max(dapi_img_)

    masks_he_df = pd.DataFrame(masks_he_, columns=["polygon"])

    masks_dapi_df = pd.DataFrame(masks_dapi_, columns=["polygon"])
    dapi_gpd = gpd.GeoDataFrame(masks_dapi_df, geometry=masks_dapi_df.polygon)
    dapi_sindex = dapi_gpd.sindex
    viz_count = 0
    area_nucleus = {"<0.3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    no_match = []
    for i, mask_he in masks_he_df.iterrows():
        mask_he = mask_he[0]
        intersection = list(dapi_sindex.intersection(mask_he.bounds))
        if len(intersection) > 0:
            ious = []
            intersection_mask = dapi_gpd.iloc[intersection]
            for j, mask_dapi in intersection_mask.iterrows():
                mask_dapi = mask_dapi.polygon
                if mask_dapi.intersection(mask_he):
                    iou = mask_dapi.intersection(mask_he).area / mask_dapi.union(mask_he).area
                    ious.append(iou)
                    if viz_count < 500:
                        viz_count += 1
                        bounds = np.vstack((np.array(mask_dapi.bounds), np.array(mask_he.bounds)))
                        bounds_min = np.min(bounds[:, [0, 1]], axis=0).astype(int)
                        bounds_max = np.max(bounds[:, [2, 3]], axis=0).astype(int)

                        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

                        axs[0].imshow(dapi_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                                      origin="lower", vmin=min_dapi, vmax=max_dapi)
                        axs[1].imshow(he_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                                      origin="lower")
                        [ax.plot(np.array(mask_he.exterior.coords)[:, 0], np.array(mask_he.exterior.coords)[:, 1],
                                 'g', zorder=2, label="he segment") for ax in axs]
                        [ax.plot(np.array(mask_dapi.exterior.coords)[:, 0], np.array(mask_dapi.exterior.coords)[:, 1],
                                 'b', zorder=2, label="dapi segment") for ax in axs]
                        if 0.3 < iou <= 0.4:
                            plt.savefig(results_dir_ / "0_3-0_4" / f"example_{i}.png")
                        elif 0.4 < iou <= 0.5:
                            plt.savefig(results_dir_ / "0_4-0_5" / f"example_{i}.png")
                        elif 0.5 < iou <= 0.6:
                            plt.savefig(results_dir_ / "0_5-0_6" / f"example_{i}.png")
                        elif 0.6 < iou <= 0.7:
                            plt.savefig(results_dir_ / "0_6-0_7" / f"example_{i}.png")
                        elif 0.7 < iou <= 0.8:
                            plt.savefig(results_dir_ / "0_7-0_8" / f"example_{i}.png")
                        elif 0.8 < iou <= 0.9:
                            plt.savefig(results_dir_ / "0_8-0_9" / f"example_{i}.png")
                        elif 0.9 < iou <= 1.0:
                            plt.savefig(results_dir_ / "0_9-1_0" / f"example_{i}.png")
                        plt.close()

            if len(ious) > 0:
                iou = max(ious)
                if iou > 0.3:
                    if 0.3 < iou <= 0.4:
                        area_nucleus["0_3-0_4"].append(mask_he.area)
                    elif 0.4 < iou <= 0.5:
                        area_nucleus["0_4-0_5"].append(mask_he.area)
                    elif 0.5 < iou <= 0.6:
                        area_nucleus["0_5-0_6"].append(mask_he.area)
                    elif 0.6 < iou <= 0.7:
                        area_nucleus["0_6-0_7"].append(mask_he.area)
                    elif 0.7 < iou <= 0.8:
                        area_nucleus["0_7-0_8"].append(mask_he.area)
                    elif 0.8 < iou <= 0.9:
                        area_nucleus["0_8-0_9"].append(mask_he.area)
                    elif 0.9 < iou <= 1.0:
                        area_nucleus["0_9-1_0"].append(mask_he.area)
                else:
                    area_nucleus["<0.3"].append(mask_he.area)
            else:
                no_match.append(i)

    for i in no_match:
        mask = masks_he_df.iloc[i].polygon
        bounds = np.array(mask.bounds)
        bounds_min = bounds[[0, 1]].astype(int)
        bounds_max = bounds[[2, 3]].astype(int)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

        axs[0].imshow(dapi_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                      origin="lower", vmin=min_dapi, vmax=max_dapi)
        axs[1].imshow(he_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                      origin="lower")
        [ax.plot(np.array(mask.exterior.coords)[:, 0], np.array(mask.exterior.coords)[:, 1],
                 'g', zorder=2, label="he segment") for ax in axs]

        plt.savefig(results_dir_ / "no_match" / f"example_{i}.png")
        plt.close()

    # 4314 -> would be nice to keep only the highest values in the count

    dfs = []
    for key, values in area_nucleus.items():
        df = pd.DataFrame({"IOU bin": key + f"_{len(values)}", "Data": values})
        dfs.append(df)

    # Concatenate the data frames
    data_df = pd.concat(dfs)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='IOU bin', y='Data', data=data_df)
    plt.xlabel('IOU bin')
    plt.ylabel('Nucleus Area')
    plt.title('Violin Plot of Data by Keys')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir_ / "iou_bin_dapi.png")

    total_length = [len(values) for i, values in enumerate(area_nucleus.values()) if i > 0]
    total_length = np.sum(total_length)
    print("Number of intersection: ", total_length)
    print("Percentage of intersection: ", total_length / len(masks_he_))



def load_geojson(path: Path, shift):
    """ Expect as input a geojson object

        Format:
            type = Feature
            geometry -> type = Polygon
            properties -> object_type = detection
            properties -> classification -> name = tilesize_norm1_norm2_resolution
    """

    # Step 0: load masks
    with open(path, "r") as file:
        features_dict = json.load(file)["features"]

    # Step 1: iterate through all object and extract category:
    str_cat = []
    tile_size = []
    resolution = []
    normalization = []
    img_coordinates = []
    masks_dict = {}
    for feature in features_dict:
        if feature['type'] == 'Feature' and feature['geometry']['type'] == 'Polygon':
            properties = feature['properties']
            if properties['objectType'] == 'detection':
                params_ = feature['properties']["classification"]['name']
                tmp = params_.split("_")
                params_ = f"{tmp[0]}_{tmp[1]}-{tmp[2]}_{tmp[3]}"

                if params_ not in str_cat:

                    str_cat.append(params_)
                    tmp = params_.split("_")

                    if tmp[0] not in tile_size:
                        tile_size.append(tmp[0])
                    if tmp[1] not in normalization:
                        normalization.append(tmp[1])
                    if tmp[2] not in resolution:
                        resolution.append(tmp[2])

                if not masks_dict.get(params_, None):
                    masks_dict[params_] = []

                if shift: # tmp fix
                    tmp = []
                    for coord in feature['geometry']['coordinates'][0]:
                        tmp.append([coord[0]+10000, coord[1]])
                else:
                    tmp = feature['geometry']['coordinates'][0]
                masks_dict[params_].append(Polygon(tmp))

            elif properties['objectType'] == 'annotation' and len(feature['geometry']['coordinates'][0]) == 5:
                img_coordinates.append(feature['geometry']['coordinates'][0])

    params_cat = {"tile_size": tile_size, "normalization": normalization, "resolution": resolution}

    return masks_dict, str_cat, params_cat


def build_dir():
    results_path = get_results_path()
    results_path = results_path / "segment_qupath"
    os.makedirs(results_path, exist_ok=True)
    return results_path


if __name__ == "__main__":
    # ---------------------------------------------------- #
    # Run Parameters

    run_segment_main = False
    run_segment_test = True
    run_type = "MATCHING"

    # ---------------------------------------------------- #
    # Set up the run
    results_dir = build_dir()

    masks, str_category, params_category = None, None, None
    masks_he, masks_dapi = None, None
    if run_type == "DAPI":
        # Segmentation for DAPI Images
        segmentation_path = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "morphology_mip.ome.tif - Image0_v2.geojson"
        results_dir = results_dir / "DAPI"
        os.makedirs(results_dir, exist_ok=True)
        masks, str_category, params_category = load_geojson(segmentation_path, shift=False)

    elif run_type == "H&E":
        # Segmentation for H&E Images
        segmentation_path = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "morphology_he_v2.geojson"
        results_dir = results_dir / "H&E_v2"
        os.makedirs(results_dir, exist_ok=True)
        masks, str_category, params_category = load_geojson(segmentation_path, shift=True)

    elif run_type == "MATCHING":
        segmentation_path_he = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "morphology_he.geojson"
        segmentation_path_dapi = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "morphology_mip.ome.tif - Image0_v2.geojson"
        results_dir = results_dir / "matching"
        os.makedirs(results_dir, exist_ok=True)
        masks_he, str_category_he, params_category_he = load_geojson(segmentation_path_he, shift=True)
        masks_dapi, str_category_dapi, params_category_dapi = load_geojson(segmentation_path_dapi, shift=False)

    image_dapi_path = get_human_breast_dapi_aligned_path()
    image_dapi = load_xenium_he_ome_tiff(image_dapi_path, level_=0)
    image_he_path = get_human_breast_he_aligned_path()
    image_he = load_xenium_he_ome_tiff(image_he_path, level_=0)

    # ---------------------------------------------------- #
    # Main functions

    if run_segment_main:
        pass
    elif run_segment_test:
        if run_type in ["DAPI", "H&E"]:
            print("Tests: segmentation QuPath")
            test_normalize_percentile_distribution(image_dapi, results_dir)
            test_number_cells(masks, params_category, str_category, results_dir)
            test_feature_distribution("size", masks, params_category, str_category, image_dapi, results_dir)
            test_feature_distribution("mean", masks, params_category, str_category, image_dapi, results_dir)
            test_feature_distribution("solidity", masks, params_category, str_category, image_dapi, results_dir)
            test_feature_distribution("circularity", masks, params_category, str_category, image_dapi, results_dir)
        else:
            he_params = {"tile_size": 4096, "resolution": 0.3, "normalization": "3.0-97.0"}
            he_str = '4096_1.0-99.0_0.3'
            dapi_params = {"tile_size": 4096, "resolution": 0.4, "normalization": "97.0-3.0"}
            dapi_str = '4096_1.0-99.0_0.4'

            test_matching(masks_dapi[dapi_str], masks_he[he_str], image_dapi, image_he, results_dir)
